"""
LLM统一配置模块
管理DeepSeek API密钥、端点URL、模型名称等配置
替代原来散落在各文件中的OpenAI配置
"""
import os
from pathlib import Path

# 尝试自动加载 .env 文件（如存在）
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

# API密钥 - 请替换为你自己的DeepSeek API Key
# 可以通过环境变量 DEEPSEEK_API_KEY 设置
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "Replace with your DeepSeek API Key"
)

# API端点
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions"

# 模型名称
DEEPSEEK_VISION_MODEL = "deepseek-chat"         # 保留，部分场景回退用
DEEPSEEK_CHAT_MODEL = "deepseek-chat"           # 文本生成模型 (替代GPT-4-turbo)

# API参数
DEEPSEEK_MAX_TOKENS = 300   # 最大输出token数
DEEPSEEK_TEMPERATURE = 0.7  # 生成温度

# ============================================================
# 本地 VLM (LLaVA) 配置
# ============================================================

LOCAL_VLM_ID = "/data/model/llava-hf_llava-1.5-13b-hf"  # 本地 LLaVA 模型路径（HF转换格式）
VLM_MAX_TOKENS = 200                # 视觉模型最大输出 token 数




# ============================================================
# 数据存储路径配置
# ============================================================

# 图像输出根目录
DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)

# 测试图像路径 (gpt4v_observe生成的)
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test.jpg")

# 生成训练数据目录
GEN_TRAIN_DIR = os.path.join(DATA_DIR, "gen_train")
os.makedirs(GEN_TRAIN_DIR, exist_ok=True)

# 描述数据目录
DESCRIPTIONS_DIR = os.path.join(DATA_DIR, "descriptions_data")
os.makedirs(DESCRIPTIONS_DIR, exist_ok=True)

# ============================================================
# Stable Diffusion (HuggingFace) 配置
# ============================================================

# HuggingFace模型ID
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"    # 保留原值，用于 from_pretrained 自动查找缓存
SD_MODEL_PATH = "/data/model/runwayml_stable-diffusion-v1-5"  # 本地缓存路径

# 图像生成参数
SD_IMAGE_SIZE = 512          # 生成图像尺寸 (像素)
SD_NUM_INFERENCE_STEPS = 50  # 推理步数 (越高质量越好，速度越慢)
SD_GUIDANCE_SCALE = 7.5      # 引导尺度 (越高越贴近prompt)

# ============================================================
# 辅助函数
# ============================================================

def get_deepseek_headers():
    """获取DeepSeek API请求头"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }


def get_deepseek_payload(messages, model=None, max_tokens=None):
    """获取DeepSeek API请求体

    Args:
        messages: 消息列表 (OpenAI格式)
        model: 模型名称, 默认用 DEEPSEEK_CHAT_MODEL
        max_tokens: 最大输出token数

    Returns:
        dict: API请求体
    """
    if model is None:
        model = DEEPSEEK_CHAT_MODEL
    if max_tokens is None:
        max_tokens = DEEPSEEK_MAX_TOKENS

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    return payload