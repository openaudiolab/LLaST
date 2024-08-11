# LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models

<div align="center">
<!-- [🏰[Project Page](https://github.com/open-compass/MathBench/)] -->
[📄[Paper](https://arxiv.org/abs/2407.15415)]
</div>

Training codes is released in [xtuner](https://github.com/InternLM/xtuner/pull/837), and more details will be completed in the near future. Thank you for your attention!

# Introduction
We introduces **LLaST**, a framework for building high-performance **L**arge **La**nguage model based **S**peech-to-text **T**ranslation systems.
We address the limitations of end-to-end speech translation~(E2E ST) models by exploring model architecture design and optimization techniques tailored for LLMs. Our approach includes LLM-based speech translation architecture design, ASR-augmented training, multilingual data augmentation, and dual-LoRA optimization. Our approach demonstrates superior performance on the CoVoST-2 benchmark and showcases exceptional scaling capabilities powered by LLMs.
We believe this effective method will serve as a strong baseline for speech translation and provide insights for future
improvements of the LLM-based speech translation framework

# Model List

| Model         | Speech Encoder       | LLM           | HuggingFace   | ModelScope    |
|---------------|----------------------|---------------|---------------|---------------|
| LLaST-2B      | Whisper-Large        | TinyLlama              | TBD           | TBD           |
| LLaST-8B      | Whisper-Large        | Llama2-7B-Instruct     | TBD          | TBD           |


# Training LLaST

## Data Preparation
- Download data from [CommonVoice](https://commonvoice.mozilla.org/en/datasets)

- Prepare tsv data as follows:

```bash
covost2/tsv
├── covost_v2.de_en.dev.tsv
├── covost_v2.de_en.test.tsv
```

- Prepare the multi-lingual data as the follows

```bash
covost/audio
├── de
├── en
├── es
├── fr
├── it
├── ja
└── zh-CN
```

- Prepare the audio data as the follows:
```bash
covost2/audio/fr/clips_16k
├── common_voice_fr_20241860.wav
├── common_voice_fr_20241864.wav
├── common_voice_fr_20241868.wav
├── common_voice_fr_20241872.wav
└── common_voice_fr_20241875.wav
```

## Training with XTuner
1. Install xtuner
```bash
git clone git@github.com:ChenX17/xtuner.git

cd xtuner

git checkout add_llast
```

2. Training 
```bash
export XTUNER_DATASET_TIMEOUT=120
export HF_EVALUATE_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
python xtuner/tools/train.py worksapce/configs/llast_2b_tinyllama_chat.py  --deepspeed deepspeed_zero2
```

# Evaluation
```bash
export HF_EVALUATE_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
python xtuner/tools/test.py worksapce/configs/llast_2b_tinyllama_chat.py --checkpoint work_dir/xxxx/epoch_1.pth/mp_rank_00_model_states.pt --laucher slurm
```

# Citation

```tex
@inproceedings{chen2024llast,
  title = {LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models},
  author = {Chen, Xi and Zhang, Songyang and Bai, Qibing and Chen, Kai and Nakamura, Satoshi},
  booktitle = {Findings of the Association for Computational Linguistics (ACL),},
  year = {2024}
}
```

# Acknowledgement
