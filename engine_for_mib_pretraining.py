# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

from cgitb import enable
import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import all_reduce_mean, is_main_process, get_rank
import sts_eval
from transformers.tokenization_utils import PreTrainedTokenizer
import logging
logger = logging.getLogger(__name__)
def train_one_epoch(model: torch.nn.Module, condenser_model: torch.nn.Module,condenser_model_without_ddp: torch.nn.Module,
                    vqkd: torch.nn.Module, tokenizer: PreTrainedTokenizer ,
                    data_loader_paired: Iterable,data_loader_text: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, best_stsb: float,loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, steps_per_epoch=None,args=None,paired_sample_step=-1):
    model.eval()
    condenser_model.train()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if get_rank() % 8==0 else logging.ERROR)
    header = 'Epoch: [{}]'.format(epoch)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    optimizer.zero_grad()

    for step in range(steps_per_epoch):
        complete_step = step + epoch * steps_per_epoch
        if args.only_wiki1m:
            batch = next(data_loader_text)
            best_stsb=train_for_text(condenser_model, condenser_model_without_ddp,tokenizer, optimizer, batch, \
                                complete_step, device, best_stsb, loss_scaler, max_norm, log_writer, args, step)
        else:
            if step % paired_sample_step==0:
                batch = next(data_loader_paired)
                best_stsb=train_for_pair(model, condenser_model, condenser_model_without_ddp,vqkd, tokenizer, optimizer, batch, \
                                    complete_step, device, best_stsb, loss_scaler, max_norm, log_writer, args, step)
            else:
                batch = next(data_loader_text)
                best_stsb=train_for_text(condenser_model, condenser_model_without_ddp,tokenizer, optimizer, batch, \
                                    complete_step, device, best_stsb, loss_scaler, max_norm, log_writer, args, step)


        if (step+1) % args.accum_iter == 0:
            lr_scheduler.step()

    return best_stsb

def train_for_pair(model: torch.nn.Module, condenser_model: torch.nn.Module,condenser_model_without_ddp: torch.nn.Module,
                    vqkd: torch.nn.Module, tokenizer: PreTrainedTokenizer ,optimizer: torch.optim.Optimizer,batch,
                    complete_step:int,device: torch.device, best_stsb: float,loss_scaler, max_norm: float = 0,log_writer=None,args=None,step=-1):
    print_freq = 25
    samples, images, bool_masked_pos = batch["samples"], batch["images"], batch["bool_masked_pos"]

    txt_input_ids, txt_mlm_input_ids,txt_labels, attention_mask,type_ids = batch["input_ids"],batch["mlm_input_ids"], batch["mlm_labels"], batch["attention_mask"],batch["token_type_ids"]
    txt_input_ids, txt_mlm_input_ids,txt_labels, attention_mask,type_ids = txt_input_ids.to(device, non_blocking=True), txt_mlm_input_ids.to(device, non_blocking=True), \
                                                txt_labels.to(device, non_blocking=True), attention_mask.to(device,non_blocking=True),type_ids.to(device, non_blocking=True)

    images = images.to(device, non_blocking=True)
    samples = samples.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
    txt_input_ids, txt_mlm_input_ids,txt_labels, attention_mask = \
        txt_input_ids.to(device, non_blocking=True), txt_mlm_input_ids.to(device, non_blocking=True), \
                                                txt_labels.to(device, non_blocking=True), attention_mask.to(device,non_blocking=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            input_ids = vqkd.get_codebook_indices(images)
        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        labels = input_ids[bool_masked_pos]

    with torch.cuda.amp.autocast():  # enabled=False
        with torch.no_grad():
            beit_cls, beit_feat = model(samples, bool_masked_pos=bool_masked_pos)
            beit_cls, beit_feat = beit_cls.detach(), beit_feat.detach()

        model_input = {"mlm_input_ids": txt_mlm_input_ids,"input_ids": txt_input_ids,
                            "attention_mask": attention_mask,"token_type_ids":type_ids}
        loss, beit_mim_loss, beit_mlm_loss, last_mlm_loss,mid_mlm_loss ,cl_loss= \
            condenser_model(model_input, txt_labels, beit_cls, beit_feat, labels, bool_masked_pos,mode="pair")
        # print("a")

    loss_value = loss.item()
    beit_mim_loss_value = beit_mim_loss.item()
    beit_mlm_loss_value = beit_mlm_loss.item()
    last_mlm_loss_value = last_mlm_loss.item()
    mid_mlm_loss_value = mid_mlm_loss.item()
    cl_loss_value = cl_loss.item()

    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
        sys.exit(1)

    
    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    loss /= args.accum_iter
    grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
            parameters=condenser_model.parameters(), create_graph=is_second_order,update_grad=(step+1)%args.accum_iter==0)
    # for name, param in condenser_model.named_parameters():
    #     if param.grad is None:
    #         print(name)

    if (step+1)%args.accum_iter==0:
        optimizer.zero_grad()
    loss_scale_value = loss_scaler.state_dict()["scale"]

    torch.cuda.synchronize()
    # if complete_step % 1 ==0:
    lr = optimizer.param_groups[0]["lr"]

    if complete_step % (print_freq * 5) == 0:
        logger.info("***** Start evaluation *****")
        condenser_model.eval()
        bert = condenser_model_without_ddp.lm
        metrics = sts_eval.evaluate(bert, tokenizer)
        stsb_spearman = metrics["eval_stsb_spearman"]
        stsb_spearman = all_reduce_mean(stsb_spearman)
        # stsb_spearman = stsb_spearman.item()
        logger.info(f" step {complete_step}: eval_stsb_spearman = {stsb_spearman}")
        if log_writer is not None:
            log_writer.add_scalar("stsb_spearman", stsb_spearman, complete_step)
        if stsb_spearman > best_stsb:
            best_stsb = stsb_spearman
            if args.output_dir:
                logger.info("Saving best model checkpoint to %s", args.output_dir)
                logger.info("best stsb %f", stsb_spearman)
                utils.save_condenser_step(
                    args=args, model=condenser_model, model_without_ddp=condenser_model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, mode="step")
        condenser_model.train()
    # torch.distributed.barrier()

    logger.info(f" step {complete_step}: loss = {loss_value},beit_mim_loss_value = {beit_mim_loss_value}, \
        beit_mlm_loss_value = {beit_mlm_loss_value} , last_mlm_loss_value = {last_mlm_loss_value} , \
        mid_mlm_loss_value = {mid_mlm_loss_value}, cl_loss_value={cl_loss_value},loss_scale_value = {loss_scale_value}, lr_value = {lr}")

    loss_value_reduce = all_reduce_mean(loss_value)
    beit_mim_loss_value_reduce = all_reduce_mean(beit_mim_loss_value)
    beit_mlm_loss_value_reduce = all_reduce_mean(beit_mlm_loss_value)
    last_mlm_loss_value_reduce = all_reduce_mean(last_mlm_loss_value)
    mid_mlm_loss_value_reduce = all_reduce_mean(mid_mlm_loss_value)
    cl_loss_value_reduce = all_reduce_mean(cl_loss_value)
    loss_scale_value_reduce = all_reduce_mean(loss_scale_value)
    if log_writer is not None:
        log_writer.add_scalar("loss",loss_value_reduce, complete_step)
        log_writer.add_scalar("beit_mim_loss", beit_mim_loss_value_reduce, complete_step)
        log_writer.add_scalar("beit_mlm_loss", beit_mlm_loss_value_reduce, complete_step)
        log_writer.add_scalar("last_mlm_loss", last_mlm_loss_value_reduce, complete_step)
        log_writer.add_scalar("mid_mlm_loss", mid_mlm_loss_value_reduce, complete_step)
        log_writer.add_scalar("cl_loss",cl_loss_value_reduce, complete_step)
        log_writer.add_scalar("loss_scale", loss_scale_value_reduce, complete_step)
        log_writer.add_scalar("lr", lr, complete_step)
        # log_writer.set_step()
    return best_stsb

def train_for_text(condenser_model: torch.nn.Module,condenser_model_without_ddp: torch.nn.Module,
                    tokenizer: PreTrainedTokenizer ,optimizer: torch.optim.Optimizer,batch,
                    complete_step:int,device: torch.device, best_stsb: float,loss_scaler, max_norm: float = 0,log_writer=None,args=None,step=-1):
    print_freq = 25
    txt_input_ids, txt_mlm_input_ids,txt_labels, attention_mask,type_ids = batch["input_ids"],batch["mlm_input_ids"], batch["mlm_labels"], batch["attention_mask"],batch["token_type_ids"]
    txt_input_ids, txt_mlm_input_ids,txt_labels, attention_mask,type_ids = txt_input_ids.to(device, non_blocking=True), txt_mlm_input_ids.to(device, non_blocking=True), \
                                                txt_labels.to(device, non_blocking=True), attention_mask.to(device,non_blocking=True),type_ids.to(device, non_blocking=True)

    with torch.cuda.amp.autocast():  # enabled=False
        model_input = {"mlm_input_ids": txt_mlm_input_ids,"input_ids": txt_input_ids,
                            "attention_mask": attention_mask,"token_type_ids":type_ids}
        loss,last_mlm_loss,mid_mlm_loss,cl_loss= condenser_model(model_input, txt_labels, mode="text")

    loss_value = loss.item()
    last_mlm_loss_value = last_mlm_loss.item()
    mid_mlm_loss_value = mid_mlm_loss.item()
    cl_loss_value = cl_loss.item()
    if not math.isfinite(loss_value):
        print(f"Loss is {mlm_loss_value}, stopping training at rank {utils.get_rank()}", force=True)
        sys.exit(1)

    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    loss /=args.accum_iter
    grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
            parameters=condenser_model.parameters(), create_graph=is_second_order,update_grad=(step+1)%args.accum_iter==0)
    # for name, param in condenser_model.named_parameters():
    #     if param.grad is None:
    #         print(name)

    if (step+1)%args.accum_iter==0:
        optimizer.zero_grad()
    loss_scale_value = loss_scaler.state_dict()["scale"]

    torch.cuda.synchronize()
    # if complete_step % 1 ==0:
    lr = optimizer.param_groups[0]["lr"]
    if complete_step % (print_freq * 5) == 0:
        logger.info("***** Start evaluation *****")
        # if is_main_process():
        condenser_model.eval()
        bert = condenser_model_without_ddp.lm
        metrics = sts_eval.evaluate(bert, tokenizer)
        stsb_spearman = metrics["eval_stsb_spearman"]
        stsb_spearman = all_reduce_mean(stsb_spearman)
        # stsb_spearman = stsb_spearman.item()
        logger.info(f" step {complete_step}: eval_stsb_spearman = {stsb_spearman}")
        if log_writer is not None:
            log_writer.add_scalar("stsb_spearman", stsb_spearman, complete_step)
        if stsb_spearman > best_stsb:
            best_stsb = stsb_spearman
            if args.output_dir:
                logger.info("Saving best model checkpoint to %s", args.output_dir)
                logger.info("best stsb %f", stsb_spearman)
                utils.save_condenser_step(
                    args=args, model=condenser_model, model_without_ddp=condenser_model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, mode="step")
        condenser_model.train()
        # torch.distributed.barrier()

    if step % print_freq==0:
        logger.info(f" step {complete_step}: loss={loss_value}, text_last_mlm_loss_value = {last_mlm_loss_value} , \
        text_mid_mlm_loss_value = {mid_mlm_loss_value} ,cl_loss={cl_loss_value},loss_scale_value = {loss_scale_value}, lr_value = {lr}")

    loss_value_reduce = all_reduce_mean(loss_value)
    last_mlm_loss_value_reduce = all_reduce_mean(last_mlm_loss_value)
    mid_mlm_loss_value_reduce = all_reduce_mean(mid_mlm_loss_value)
    cl_loss_value_reduce = all_reduce_mean(cl_loss_value)
    loss_scale_value_reduce = all_reduce_mean(loss_scale_value)
    if log_writer is not None:
        log_writer.add_scalar("wiki_loss", loss_value_reduce, complete_step)
        log_writer.add_scalar("wiki_last_mlm_loss", last_mlm_loss_value_reduce, complete_step)
        log_writer.add_scalar("wiki_mid_mlm_loss", mid_mlm_loss_value_reduce, complete_step)
        log_writer.add_scalar("wiki_cl_loss", cl_loss_value_reduce, complete_step)
        log_writer.add_scalar("loss_scale", loss_scale_value_reduce, complete_step)
        log_writer.add_scalar("lr", lr, complete_step)
        # log_writer.set_step()
    return best_stsb