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
from condenser.modeling_condenser import CondenserForPretraining, RobertaCondenserForPretraining
from condenser.arguments import DataTrainingArguments, ModelArguments
from condenser.arguments import CondenserPreTrainingArguments as TrainingArguments
# from flickr_train_dataset import flickr_train
from sts_dataset import paired_dataset,wiki1m_dataset
from condenser.data import CondenserCollator,CondenserCollator_text
from transformers.optimization import get_scheduler
from transformers.trainer_utils import SchedulerType
import math
from utils import inf_train_gen
from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )

CONDENSER_TYPE_MAP = {
    'bert': CondenserForPretraining,
    'roberta': RobertaCondenserForPretraining,
}

from torch.utils.tensorboard import SummaryWriter
import torch
from timm.models import create_model
from args import beit_args
if __name__ == "__main__":
    opts = beit_args.get_args()
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
    checkpoint = torch.load("pretrained_model/beitv2_base_patch16_224_pt1k.pth", map_location='cpu')
    model = model.load_state_dict(checkpoint['model'])
    for name,param in model.named_parameters():
        print(name)
    print(checkpoint.keys())
