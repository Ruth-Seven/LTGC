"""
集中配置模块
管理本地 VLM (LLaVA/Qwen2-VL)、Stable Diffusion 等所有配置项
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import torch
from pathlib import Path

# ============================================================
# 环境变量加载
# ============================================================
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


# ============================================================
# 本地 VLM 配置（LLaVA / Qwen2-VL 双后端）
# ============================================================
LLAVA_MODEL_ID = "/data/model/llava-hf_llava-1.5-7b-hf"
QWEN2VL_MODEL_ID = "/data/hujunjie/models/Qwen2-VL-7B-Instruct"
LOCAL_VLM_ID = QWEN2VL_MODEL_ID
VLM_MAX_TOKENS = 200
VLM_TEMPERATURE = 0.7
VLM_TOP_P = 0.9

# ============================================================
# Text LLM (描述扩展) 配置
# ============================================================

# local llm 配置
TEXT_LLM_MODEL_ID = "Qwen/Qwen3-8B"
TEXT_LLM_MAX_TOKENS = 10000
TEXT_LLM_TEMPERATURE = 0.8

# ============================================================
# CLIP 配置
# ============================================================
CLIP_BACKEND="huggingface"    # "openai" (原生clip库) 或 "huggingface" (transformers)
# CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
CLIP_MODEL_NAME="openai/clip-vit-large-patch14"
CLIP_MAX_TOKENS = 77 - 2
# ============================================================
# Stable Diffusion 配置
# ============================================================
SD_MODEL_VERSION = "sdxl"       # "v1_5" 或 "sdxl"
SD_V1_5_PATH = "/data/model/runwayml_stable-diffusion-v1-5"
SDXL_PATH = "/data/hujunjie/models/sdxl-base-1.0-runtime-fp16"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_IMAGE_SIZE = 1024
SD_NUM_INFERENCE_STEPS = 30
SD_GUIDANCE_SCALE = 7.5
SD_STYLE_SUFFIX = ", 8k, photorealistic, dslr, ultra detailed, professional photography, natural lighting"

# ============================================================
# 数据路径配置
# ============================================================
DATA_DIR = "/data/hujunjie/generate/ltgc/runtime-data"
PWD=os.path.abspath(os.path.dirname(__file__))
DESCRIPTIONS_DIR = os.path.join(DATA_DIR, "descriptions_data")
GENERATION_EXAMPLE_DIR=os.path.join(PWD, "example/generation_examples")
DESCRIPTION_EXAMPLE_DIR=os.path.join(PWD, "example/description_examples")
EXTENDED_DESCRIPTION_PATH=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv')
GEN_TRAIN_DIR = os.path.join(DATA_DIR, "gen_train")
IMAGENET_DIR = "/data/hujunjie/datasets/imagentlt/imagenet-lt-v2"
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test.jpg")
CLASS_COUNT_FILE = os.path.join(DATA_DIR, "imagenetlt_class_count.json")

os.makedirs(DESCRIPTIONS_DIR, exist_ok=True)
os.makedirs(GEN_TRAIN_DIR, exist_ok=True)
