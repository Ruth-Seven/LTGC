"""
集中配置模块
管理 DeepSeek API、本地 VLM (LLaVA)、Stable Diffusion 等所有配置项
"""
import os
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
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "Replace with your DeepSeek API Key")
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 300
DEEPSEEK_TEMPERATURE = 0.7

# ============================================================
# 本地 VLM (LLaVA) 配置
# ============================================================
LOCAL_VLM_ID = "/data/model/llava-hf_llava-1.5-13b-hf"
VLM_MAX_TOKENS = 200

# ============================================================
# Stable Diffusion 配置
# ============================================================
SD_MODEL_VERSION = "sdxl"       # "v1_5" 或 "sdxl"
SD_V1_5_PATH = "/data/model/runwayml_stable-diffusion-v1-5"
SDXL_PATH = "/data/model/stable-diffusion-xl-base-1.0"
SD_IMAGE_SIZE = 1024
SD_NUM_INFERENCE_STEPS = 30
SD_GUIDANCE_SCALE = 7.5

# ============================================================
# 数据路径配置
# ============================================================
DATA_DIR = "/data"
DESCRIPTIONS_DIR = os.path.join(DATA_DIR, "descriptions_data")
GEN_TRAIN_DIR = os.path.join(DATA_DIR, "gen_train")
IMAGENET_DIR = "/data/imagenet-lt/torch_image_folder/mnt/volume_sfo3_01/imagenet-lt/ImageDataset"

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
