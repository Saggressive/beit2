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

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys 
sys.path.append("./")
from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from beit_datasets import build_beit_pretraining_dataset
# from engine_for_pretraining import train_one_epoch
from engine_for_mib_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
# import modeling_pretrain
import mibcse_pretrain as modeling_pretrain
import modeling_vqkd
from args import beit_args
# from condenser.modeling_condenser import CondenserForPretraining, RobertaCondenserForPretraining
# from condenser.modeling_condenser_init import CondenserForPretraining, RobertaCondenserForPretraining
from condenser.modeling_condenser_cl import CondenserForPretraining, RobertaCondenserForPretraining
from condenser.arguments import DataTrainingArguments, ModelArguments
from condenser.arguments import CondenserPreTrainingArguments as TrainingArguments
# from flickr_train_dataset import flickr_train
from sts_dataset_cl import paired_dataset,wiki1m_dataset
# from sts_dataset import paired_dataset,wiki1m_dataset
from transformers.optimization import get_scheduler
from transformers.trainer_utils import SchedulerType
import math
from utils import inf_train_gen
from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BertTokenizer,
    HfArgumentParser,
    set_seed, )

CONDENSER_TYPE_MAP = {
    'bert': CondenserForPretraining,
    'roberta': RobertaCondenserForPretraining,
}
from condenser import data,cl_data
def get_model(args, model_args, data_args, training_args):
    print(f"Creating model: {args.model}")
    if 'cls_pt' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size,
            early_layers=args.early_layers,
            head_layers=args.head_layers,
            shared_lm_head=args.shared_lm_head,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size
        )
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()

    _condenser_cls = CONDENSER_TYPE_MAP[model_args.model_type]
    condenser_model = _condenser_cls.from_pretrained(
            model_args, data_args, training_args,args,
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    return model , condenser_model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def init_condenser_weight(condenser,args):

    beit_ckpt = torch.load(args.resume, map_location='cpu')
    mlm_head_ckpt = torch.load(args.mlm_head, map_location='cpu')
    condenser_state_dict=condenser.state_dict()
    condenser_mim_keys=[]
    beit_mim_head=["norm","lm_head","cls_pt_layers"]
    for key in condenser_state_dict.keys():
        if key.split(".")[0] in beit_mim_head:
            condenser_mim_keys.append(key)
    for key in condenser_mim_keys:
        condenser_state_dict[key]=beit_ckpt["model"][key]
    
    
    condenser_keys=[]
    for key in condenser_state_dict.keys():
        if "beit_mlm_head" in key:
            condenser_keys.append(".".join(key.split(".")[1:]))
    for key in condenser_keys:
        key0="beit_mlm_head."+key
        key1="c_head."+key
        condenser_state_dict[key0]=mlm_head_ckpt[key1]

    condenser.load_state_dict(condenser_state_dict)
    return condenser

def main(args , model_args, data_args, training_args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model , condenser_model = get_model(args , model_args, data_args, training_args)
    if args.init_condenser:
        condenser_model = init_condenser_weight(condenser_model,args)
    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(
            model_args.tokenizer_name,use_fast=True
        )
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=True
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    paired_dataset_train = paired_dataset(json_path=args.paired_data_path, tokenizer=tokenizer, args=args)
    text_dataset_train = wiki1m_dataset(txt_path=args.text_data_path, tokenizer=tokenizer, args=args)
    # prepare visual tokenizer
    vqkd = get_visual_tokenizer(args).to(device)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        paired_dataset_steps_per_epoch =math.ceil(len(paired_dataset_train) // args.batch_size // num_tasks)
        text_dataset_steps_per_epoch = math.ceil(len(text_dataset_train) // args.batch_size // num_tasks)
        if args.only_text_cl:
            num_training_steps_per_epoch = text_dataset_steps_per_epoch
        else:
            num_training_steps_per_epoch = paired_dataset_steps_per_epoch+text_dataset_steps_per_epoch
        paired_sampler_train = torch.utils.data.DistributedSampler(
            paired_dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        text_sampler_train = torch.utils.data.DistributedSampler(
            text_dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
    else:
        paired_sampler_train = torch.utils.data.RandomSampler(paired_dataset_train)
        text_sampler_train = torch.utils.data.RandomSampler(text_dataset_train)
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.use_text_cl:
        data_collator = cl_data.CondenserCollator(
            args,
            max_seq_length=data_args.max_seq_length,
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        )
        data_collator_text = cl_data.CondenserCollator_text(
            args,
            max_seq_length=data_args.max_seq_length,
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        )
    else:
        data_collator = data.CondenserCollator(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_seq_length=data_args.max_seq_length,
        )
        data_collator_text = data.CondenserCollator_text(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            max_seq_length=data_args.max_seq_length,
        )
    tokenizer.save_pretrained(args.output_dir + os.sep + "tokenizer")
    data_loader_paired = torch.utils.data.DataLoader(
        paired_dataset_train, sampler=paired_sampler_train,collate_fn=data_collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_text = torch.utils.data.DataLoader(
        text_dataset_train, sampler=text_sampler_train,collate_fn=data_collator_text,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    paired_sample_step = len(data_loader_text) // len(data_loader_paired)
    model.to(device)
    model_without_ddp = model
    condenser_model.to(device)
    condenser_model_without_ddp = condenser_model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    # print("Tokenizer = %s" % str(vqkd))
    total_batch_size = args.batch_size * utils.get_world_size() * args.accum_iter
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        condenser_model = torch.nn.parallel.DistributedDataParallel(condenser_model, device_ids=[args.gpu], find_unused_parameters=True)
        condenser_model_without_ddp = condenser_model.module

    optimizer = create_optimizer(
        args, model_without_ddp, condenser_model_without_ddp)
    loss_scaler = NativeScaler()
    num_training_steps = num_training_steps_per_epoch * args.epochs // args.accum_iter
    num_warmup_steps = num_training_steps*args.warmup_ratio
    lr_scheduler = get_scheduler(
                SchedulerType.LINEAR,
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    #debug
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_stsb = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_paired.sampler.set_epoch(epoch)
            data_loader_text.sampler.set_epoch(epoch)
        loader_paired = inf_train_gen(data_loader_paired)
        loader_text = inf_train_gen(data_loader_text)
        # if log_writer is not None:
        #     log_writer.set_step(epoch * num_training_steps_per_epoch)

        best_stsb = train_one_epoch(
            model, condenser_model,condenser_model_without_ddp,vqkd, tokenizer,loader_paired,
            loader_text,optimizer, device, epoch,best_stsb, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            steps_per_epoch = num_training_steps_per_epoch,
            lr_scheduler=lr_scheduler,
            args=args,
            paired_sample_step=paired_sample_step,
        )

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    if args.output_dir:
        utils.save_condenser_step(
            args=args, model=condenser_model, model_without_ddp=condenser_model_without_ddp, 
            optimizer=optimizer,loss_scaler=loss_scaler , mode="final")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = beit_args.get_args()
    condenser_config = opts.condenser_config
    # condenser_config = "args/condenser_args.json"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    model_args, data_args, training_args = parser.parse_json_file(json_file=condenser_config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts , model_args, data_args, training_args)
