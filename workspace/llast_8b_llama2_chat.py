# Copyright (c) LLaST. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from torch.optim import AdamW
from peft import LoraConfig

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig, WhisperModel)
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.dataset.llast import LLaSTDataset
from xtuner.dataset.collate_fns import llast_audiomask_mel_collate_fn
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model import LLaSTModel
from xtuner.engine import LLaSTTestLoop
from xtuner.evaluation.metrics import SacreBLEUMetric


#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# Model
llm_name_or_path = 'NousResearch/Llama-2-7b-chat-hf'
speech_encoder_name_or_path = 'openai/whisper-large-v2'

# Data

# data_root = './data/covost2/multilingual_samples/'
tsv_dir = './datasets/covost2/tsv_filtered'
common_voice_folder = './datasets/covost2/audio'
audio_folder='clips_16k'

prompt_template = PROMPT_TEMPLATE.llama2_chat
max_length = 2048

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 2
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Evaluation
evaluation_freq = 200
SYSTEM = ''


#######################################################################
#            PART 2  Model & Tokenizer & Speech Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

model = dict(
    type=LLaSTModel,
    freeze_llm=True,
    freeze_speech_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    speech_encoder=dict(
        type=WhisperModel.from_pretrained,
        pretrained_model_name_or_path=speech_encoder_name_or_path),
    speech_encoder_lora=dict(
        type=LoraConfig, r=128, lora_alpha=64, lora_dropout=0.05, bias='none'),
    projector_depth=3,
)


#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################


llast_train_dataset = dict(
    type=LLaSTDataset,
    offline_processed_text_folder='data/llast_covost2_processed_llama2',
    tsv_dir=tsv_dir,
    tokenizer=tokenizer,
    split='train',
    with_asr=True,
    # debug=True,
    en2x_sample_step=1, # 1: without sampling 
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    cv_dir=common_voice_folder,
    x2en=["fr", "es", "de", "it", "zh-CN", "ja"],
    en2x=["ja", 'de', 'zh-CN'],
    audio_folder=audio_folder,
    input_ids_with_output=True  # True for train
)

llast_test_dataset = dict(
    type=LLaSTDataset,
    tsv_dir=tsv_dir,
    tokenizer=tokenizer,
    split='test',
    x2en=["fr"],
    en2x=[],
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    cv_dir=common_voice_folder,
    audio_folder=audio_folder,
    input_ids_with_output=False  # False for test
)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=llast_train_dataset,
    sampler=dict(
        type=DefaultSampler,
        shuffle=True),
    collate_fn=dict(
        type=llast_audiomask_mel_collate_fn
    ),
)


test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=llast_test_dataset,
    sampler=dict(
        type=DefaultSampler,
        shuffle=False),
    collate_fn=dict(type=llast_audiomask_mel_collate_fn)
)


test_evaluator = dict(
    type=SacreBLEUMetric,
    tokenizer=tokenizer,
    prefix='scarebleu_test',
    dump_dir='work_dirs/llast_8b_llama2',
    epoch_num='_e1_fr_en'
)


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
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]


# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
test_cfg = dict(
    type=LLaSTTestLoop,
    tokenizer=tokenizer,
    system=SYSTEM,
    num_beams=5,
    do_sample=False,
    fp16=True,
    prompt_template=prompt_template,
    max_new_tokens=256)


#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
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