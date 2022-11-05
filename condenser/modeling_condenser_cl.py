# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer
from condenser.bert_model import BertForMaskedLM
from condenser.arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging
# from timm.models.vision_transformer import PatchEmbed, Block
from modeling_finetune import Block
from timm.models.layers import trunc_normal_
from functools import partial
logger = logging.getLogger(__name__)


class mimLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size*2)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size*2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(0.1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features):
        x = self.dense1(features)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.norm(x+features)

        return x

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)

        return x

class ProjectionMLP(nn.Module):
    def __init__(self, size):
        super().__init__()
        in_dim = size
        hidden_dim = size * 2
        out_dim = size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        b,n,d=x.size()
        x=x.view(b*n,d)
        x=self.net(x)
        return x.view(b,n,d)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class CondenserForPretraining(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
        beit_args
    ):
        super(CondenserForPretraining, self).__init__()
        self.beit_args = beit_args
        self.lm = bert
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if beit_args.use_text_cl:
            self.mlp=MLPLayer(768) if not beit_args.batchnorm else ProjectionMLP(768)
            self.sim=Similarity(beit_args.temp)
            self.mlp.apply(self.lm._init_weights)
        if beit_args.use_pair_cl:
            self.mlp_v=MLPLayer(768) if not beit_args.batchnorm else ProjectionMLP(768)
            self.sim_v=Similarity(beit_args.temp_v)
            # self.mlp_v.apply(self.lm._init_weights)
        if beit_args.only_text_cl:
            return

        if beit_args.use_bert_mlm:
            self.c_head = nn.ModuleList(
                [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
            )
            self.c_head.apply(self.lm._init_weights)

        if beit_args.use_beit_mlm:
            self.img2text=MLPLayer(768)
            self.img2text.apply(self.lm._init_weights)
            self.beit_mlm_head = nn.ModuleList(
                [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
            )
            self.beit_mlm_head.apply(self.lm._init_weights)

        if beit_args.use_beit_mim:
            self.text2img=mimLayer(768)
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm=norm_layer(768)
            dpr = [x.item() for x in torch.linspace(0, 0.1, 12)]  # stochastic depth decay rule
            self.cls_pt_layers = nn.ModuleList([
                Block(
                    dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                    drop=0.0, attn_drop=0.0, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=0.1, window_size=None,
                    attn_head_dim=None,
                )
                for i in range(9, 11)])
            self.lm_head = nn.Linear(768, beit_args.codebook_size)
            from modeling_finetune import RelativePositionBias
            self.rel_pos_bias = RelativePositionBias(window_size=(14,14), num_heads=12)

    def forward(self, model_input, labels, beit_cls=None,beit_cls_cl=None, beit_hidden=None, mim_labels=None, mim_mask=None,mode="text"):
        cl_loss = torch.tensor(0,dtype=torch.float,device=model_input["input_ids"].device)
        inter_loss = torch.tensor(0,dtype=torch.float,device=model_input["input_ids"].device)
        cl_out = None
        if self.beit_args.use_text_cl:
            batch_size=model_input["input_ids"].size()[0]//2

            cl_input={"input_ids":model_input["input_ids"].view(2*batch_size,-1),\
                "attention_mask": model_input["attention_mask"].view(2*batch_size,-1),\
                "token_type_ids":model_input["token_type_ids"].view(2*batch_size,-1)}
            cl_out= self.lm(
                **cl_input,
                output_hidden_states=True,
                return_dict=True,
                cl=True
            )
            z=cl_out.hidden_states[-1][:,0]
            z=z.view(batch_size,2,-1)
            pre_z=z.clone()
            z=self.mlp(z)
            z1,z2=z[:,0],z[:,1]
            if dist.is_initialized() and self.lm.training:
                # Dummy vectors for allgather
                z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
                z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
                # Allgather
                dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
                dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                z1_list[dist.get_rank()] = z1
                z2_list[dist.get_rank()] = z2
                # Get full batch embeddings: (bs x N, hidden)
                z1 = torch.cat(z1_list, 0)
                z2 = torch.cat(z2_list, 0)
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            cl_labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
            cl_loss = self.cross_entropy(cos_sim,cl_labels)
        
        if self.beit_args.only_text_cl or (mode=="text" and self.beit_args.use_bert_mlm==False):
            return cl_loss,  torch.tensor(0,dtype=torch.float) , torch.tensor(0,dtype=torch.float), cl_loss

        if self.beit_args.use_pair_cl and self.beit_args.use_text_cl:#开图文对比，必须开文本对比
            batch_size=beit_cls_cl.size()[0]//2
            p=self.mlp_v(pre_z)
            p1,p2=p[:,0],p[:,1]
            v=beit_cls_cl.view(batch_size,2,-1)[:,0]
            if dist.is_initialized() and self.lm.training:
                # Dummy vectors for allgather
                p1_list = [torch.zeros_like(p1) for _ in range(dist.get_world_size())]
                p2_list = [torch.zeros_like(p2) for _ in range(dist.get_world_size())]
                v_list =  [torch.zeros_like(v) for _ in range(dist.get_world_size())]
                # Allgather
                dist.all_gather(tensor_list=p1_list, tensor=p1.contiguous())
                dist.all_gather(tensor_list=p2_list, tensor=p2.contiguous())
                dist.all_gather(tensor_list=v_list, tensor=v.contiguous())
                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                p1_list[dist.get_rank()] = p1
                p2_list[dist.get_rank()] = p2
                v_list[dist.get_rank()] = v
                # Get full batch embeddings: (bs x N, hidden)
                p1 = torch.cat(p1_list, 0)
                p2 = torch.cat(p2_list, 0)
                v = torch.cat(v_list, 0)
            p1,p2=p1 / p1.norm(2, dim=-1, keepdim=True),p2 / p2.norm(2, dim=-1, keepdim=True)
            v= v / v.norm(2,dim=-1,keepdim=True)
            cos_sim_p0 = self.sim_v(p1.unsqueeze(1), v.unsqueeze(0))  # (bs, bs)
            cos_sim_p1 = self.sim_v(p2.unsqueeze(1), v.unsqueeze(0))
            cl_labels = torch.arange(cos_sim_p0.size(0)).long().to(p1.device)
            inter_loss = (self.cross_entropy(cos_sim_p0, cl_labels) + self.cross_entropy(cos_sim_p1, cl_labels)) / 2

        if self.beit_args.use_bert_mlm or self.beit_args.use_beit_mlm:    
            attention_mask = self.lm.get_extended_attention_mask(
                    model_input['attention_mask'],
                    model_input['attention_mask'].shape,
                    model_input['attention_mask'].device
                )
            mlm_input={"input_ids":model_input["mlm_input_ids"],"attention_mask": model_input["attention_mask"]}
            lm_out: MaskedLMOutput = self.lm(
                **mlm_input,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            cls_hiddens = lm_out.hidden_states[-1][:, :1]
            skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
            if self.beit_args.use_bert_mlm:
            
                hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)
                for layer in self.c_head:
                    layer_out = layer(
                        hiddens,
                        attention_mask,
                    )
                    hiddens = layer_out[0]
        
        if mode=="text" and self.beit_args.use_bert_mlm:
            mlm_loss = self.mlm_loss(hiddens, labels)
            if self.model_args.late_mlm:
                loss = self.beit_args.a3*lm_out.loss + self.beit_args.a0*mlm_loss
                loss = self.beit_args.alpha * loss + cl_loss
                return loss, lm_out.loss , mlm_loss, cl_loss
            else:
                loss = self.beit_args.a0*mlm_loss
                loss = self.beit_args.alpha * loss + cl_loss
                return loss, torch.tensor(0,dtype=torch.float) , mlm_loss,cl_loss


        loss = torch.tensor(0,dtype=torch.float,device=mim_labels.device)
        beit_mim_loss , beit_mlm_loss , last_mlm_loss, mlm_loss = torch.tensor(0,dtype=torch.float,device=mim_labels.device), \
            torch.tensor(0,dtype=torch.float,device=mim_labels.device),torch.tensor(0,dtype=torch.float,device=mim_labels.device),\
            torch.tensor(0,dtype=torch.float,device=mim_labels.device)
        
        if self.beit_args.use_beit_mlm:
            # last_hiddens = lm_out.hidden_states[-1][:,1:]
            text_hiddens = skip_hiddens[:, 1:].clone()
            beit_cls = self.img2text(beit_cls)#beit cls 改下
            beit_mlm_hiddens = torch.cat([beit_cls , text_hiddens], dim=1)
            for layer in self.beit_mlm_head:
                layer_out = layer(
                    beit_mlm_hiddens,
                    attention_mask,
                )
                beit_mlm_hiddens = layer_out[0]
            beit_mlm_loss = self.mlm_loss(beit_mlm_hiddens, labels)
            loss += self.beit_args.a1*beit_mlm_loss

        if self.beit_args.use_beit_mim:

            if cl_out is None:
                batch_size=model_input["input_ids"].size()[0]//2
                cl_input={"input_ids":model_input["input_ids"].view(2*batch_size,-1),\
                    "attention_mask": model_input["attention_mask"].view(2*batch_size,-1),\
                    "token_type_ids":model_input["token_type_ids"].view(2*batch_size,-1)}
                cl_out= self.lm(
                    **cl_input,
                    output_hidden_states=True,
                    return_dict=True,
                    cl=True
                )
            last_hidden = cl_out.hidden_states[-1][:,:1]
            last_hidden = self.text2img(last_hidden)

            beit_mean = beit_hidden.mean(dim=-1, keepdim=True)
            beit_var = beit_hidden.var(dim=-1, keepdim=True)
            beit_hidden = (beit_hidden - beit_mean) / (beit_var + 1.e-6)**.5

            rel_pos_bias = self.rel_pos_bias()
            beit_mim_hiddens = torch.cat([last_hidden , beit_hidden], dim=1)
            for blk in self.cls_pt_layers:
                beit_mim_hiddens = blk(beit_mim_hiddens,rel_pos_bias=rel_pos_bias)
            beit_mim_hiddens = self.norm(beit_mim_hiddens)
            beit_mim_hiddens = beit_mim_hiddens[:,1:,:]
            beit_mim_hiddens = beit_mim_hiddens[mim_mask]
            beit_mim_hiddens = self.lm_head(beit_mim_hiddens)
            beit_mim_loss = self.cross_entropy(input=beit_mim_hiddens,target=mim_labels)
            loss+=self.beit_args.a2*beit_mim_loss            

        if self.beit_args.use_bert_mlm:
            mlm_loss = self.mlm_loss(hiddens, labels)
            loss += self.beit_args.a0*mlm_loss

        if self.model_args.late_mlm and self.beit_args.use_bert_mlm:
            last_mlm_loss = lm_out.loss
            loss += self.beit_args.a3*last_mlm_loss
        
        loss = self.beit_args.alpha * loss + cl_loss + self.beit_args.beta * inter_loss
        return loss, beit_mim_loss , beit_mlm_loss , last_mlm_loss, mlm_loss,cl_loss,inter_loss


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            beit_args,*args, **kwargs
    ):
        hf_model = BertForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args,beit_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')) and beit_args.init_condenser and beit_args.use_bert_mlm:
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = BertForMaskedLM.from_config(config)
        model = cls(hf_model, model_args, data_args, train_args)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

class RobertaCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            roberta: RobertaModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = roberta
        self.c_head = nn.ModuleList(
            [RobertaLayer(roberta.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        # self.mlm_head = BertOnlyMLMHead(bert.config)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

class CoCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            bert: BertModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: CoCondenserPreTrainingArguments
    ):
        super(CoCondenserForPretraining, self).__init__(bert, model_args, data_args, train_args)

        effective_bsz = train_args.per_device_train_batch_size * self._world_size() * 2
        target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        self.register_buffer(
            'co_target', target
        )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def forward(self, model_input, labels, grad_cache: Tensor = None, chunk_offset: int = None):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        if self.train_args.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = cls_hiddens.squeeze()

        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
            return loss + co_loss
        else:
            loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
            return loss, surrogate

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def compute_contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
        similarities.fill_diagonal_(float('-inf'))
        co_loss = F.cross_entropy(similarities, self.co_target) * self._world_size()
        return co_loss