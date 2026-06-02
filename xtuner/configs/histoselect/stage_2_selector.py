# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel_Selector
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

# Local data paths (relative to repo root).
# Symlink or copy your data into ./data/ — see README for the expected layout.
HISTOSELECT_DATA_ROOT = './data/instruct'
HISTOSELECT_FEATURE_ROOT = './data/features/conch_v1'
HISTOSELECT_WEAK_LABEL_ROOT = './data/weak_labels'
HISTOSELECT_STAGE2_INIT = './data/models/stage2.pth'

# Dataset selection — change this line to switch between datasets.
DATASET = 'wsi-llava'  # 'wsi-llava' or 'slidechat'

if DATASET == 'wsi-llava':
    data_path = f'{HISTOSELECT_DATA_ROOT}/stage_2_vqa_selector_wsi-llava/train_merge_cleaned.json'
    question_embedding_path = './data/embeddings/train_question_embeddings.pt'
elif DATASET == 'slidechat':
    data_path = f'{HISTOSELECT_DATA_ROOT}/stage_2_vqa_selector_slidechat/SlideInstruct_train_stage2_vqa.json'
    question_embedding_path = './data/embeddings/SlideChat_train_question_embeddings.pt'
else:
    raise ValueError(f"Unknown DATASET: {DATASET!r}")

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'

image_path_list = None
pretrained_pth = HISTOSELECT_STAGE2_INIT

prompt_template = PROMPT_TEMPLATE.qwen_chat


max_length = 19600
per_image_length = 196  #None
sample_type='wsi' # 'wsi'or'image'

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 2  # 4 GPU × accum=2 = eff batch 8
dataloader_num_workers = 1
max_epochs = 3
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save — every 1k iters matches the eval watcher cadence
save_steps = 1000
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')


model = dict(
    type=LLaVAModel_Selector,
    freeze_llm=True,
    pretrained_pth=pretrained_pth,
    train_stage='2',
    beta_initial=0.0,
    beta_final=0.2,
    gamma_initial=0.0,
    gamma_final=0.1,
    group_prior_bias=-1.25,
    group_prior_temperature=1.0,
    max_iters=5000,  # IB loss beta/gamma annealing length
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ),
    llm_lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM')
    )

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    per_image_length=per_image_length,
    pad_image_to_square=False,
    sample_num=10000,# max patch number
    identifier='_224x224_b20_t15',
    image_feature_prefix=HISTOSELECT_FEATURE_ROOT,
    image_feature_suffix='.h5',
    weak_label_prefix=HISTOSELECT_WEAK_LABEL_ROOT,
    weak_label_suffix='.h5',
    question_embedding_path = question_embedding_path
    ) 



train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=0,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
 
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
