"""
集中配置模块
管理 DeepSeek API、本地 VLM (LLaVA)、Stable Diffusion 等所有配置项
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
# GPU 分配（LRU 算法）
# ============================================================
class _LRUAllocator:
    def __init__(self):
        
        n = torch.cuda.device_count()
        if n == 0:
            self.id_que = [-1] # CPU 模式
        else:
            self.id_que = [i for i in range(n)]

    def allocate(self):
        gpu_id = self.id_que[0]
        if gpu_id == -1:
            return f"cpu"

        self.id_que.pop(0)
        self.id_que.append(gpu_id)
        print(f"分配到cuda {gpu_id}.\n")
        return f"cuda:{gpu_id}"

_gpu_alloc = _LRUAllocator()

def get_device(model_key=None):
    return _gpu_alloc.allocate()

# ============================================================
# 本地 VLM (LLaVA) 配置
# ============================================================
LOCAL_VLM_ID = "/data/model/llava-hf_llava-1.5-13b-hf"
VLM_MAX_TOKENS = 200

# ============================================================
# Text LLM (描述扩展) 配置
# ============================================================

# local llm 配置
TEXT_LLM_MODEL_ID = "Qwen/Qwen3-8B"
TEXT_LLM_MAX_TOKENS = 10000
TEXT_LLM_TEMPERATURE = 0.8
# api llm 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "Replace with your DeepSeek API Key")
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_VISION_MODEL = "deepseek-chat"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 300
DEEPSEEK_TEMPERATURE = 0.7

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
SDXL_PATH = "/data/model/stable-diffusion-xl-base-1.0"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_IMAGE_SIZE = 1024
SD_NUM_INFERENCE_STEPS = 30
SD_GUIDANCE_SCALE = 7.5

# ============================================================
# 数据路径配置
# ============================================================
DATA_DIR = "/data"
PWD=os.path.abspath(os.path.dirname(__file__))
DESCRIPTIONS_DIR = os.path.join(DATA_DIR, "descriptions_data")
GENERATION_EXAMPLE_DIR=os.path.join(PWD, "example/generation_examples")
DESCRIPTION_EXAMPLE_DIR=os.path.join(PWD, "example/description_examples")
EXTENDED_DESCRIPTION_PATH=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv')
GEN_TRAIN_DIR = os.path.join(DATA_DIR, "gen_train")
IMAGENET_DIR = "/data/dataset/imagenet-lt/torch_image_folder/mnt/volume_sfo3_01/imagenet-lt/ImageDataset"
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test.jpg")
CLASS_COUNT_FILE = os.path.join(DATA_DIR, "imagenetlt_class_count.json")

os.makedirs(DESCRIPTIONS_DIR, exist_ok=True)
os.makedirs(GEN_TRAIN_DIR, exist_ok=True)


def get_deepseek_headers():
    """获取 DeepSeek API 请求头"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }


def get_deepseek_payload(messages, model=None, max_tokens=None):
    """构建 DeepSeek API 请求体"""
    return {
        "model": model or DEEPSEEK_CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or DEEPSEEK_MAX_TOKENS,
    }
