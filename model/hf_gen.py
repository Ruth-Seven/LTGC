"""
HuggingFace Stable Diffusion 图像生成模块
替代原OpenAI DALL-E 2方案，使用本地Stable Diffusion模型

功能:
1. hf_gen: Stable Diffusion本地推理生成图像
2. description_refine: 使用DeepSeek Chat润色描述
3. get_cls_index_name: 获取类别索引名称
4. get_cls_template: 获取类别模板
"""
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import requests
from io import BytesIO
import os
import re

from llm_config import (
    SD_MODEL_ID,
    SD_MODEL_PATH,
    SD_IMAGE_SIZE,
    SD_NUM_INFERENCE_STEPS,
    SD_GUIDANCE_SCALE,
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_API_ENDPOINT,
    get_deepseek_headers,
    DATA_DIR,
)
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# 全局组件 (延迟加载，避免每次import都耗时)
# ============================================================
_pipe = None       # Stable Diffusion pipeline
_clip_model = None # CLIP模型 (可选，不强制要求)
_clip_processor = None


def _get_sd_pipeline():
    """获取或初始化Stable Diffusion pipeline (单例模式)"""
    global _pipe
    if _pipe is None:
        model_id = SD_MODEL_PATH if os.path.exists(SD_MODEL_PATH) else SD_MODEL_ID
        print(f"[hf_gen] Loading Stable Diffusion model: {model_id} ...")
        _pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
        # 自动使用GPU (如果可用)
        if torch.cuda.is_available():
            _pipe = _pipe.to("cuda")
        print("[hf_gen] Stable Diffusion model loaded.")
    return _pipe


def hf_gen(saved_path, input_text, saved=False):
    """使用Stable Diffusion生成图像 (替代 DALL-E gen)

    Args:
        saved_path: 保存路径
        input_text: 文本提示词
        saved: 是否保存到本地文件

    Returns:
        str or None: 保存路径 (成功) 或 None (失败)
    """
    try:
        # 截断超长文本
        if len(input_text) > 500:
            input_text = input_text[:500]

        pipe = _get_sd_pipeline()

        # 生成图像
        with torch.no_grad():
            image = pipe(
                input_text,
                height=SD_IMAGE_SIZE,
                width=SD_IMAGE_SIZE,
                num_inference_steps=SD_NUM_INFERENCE_STEPS,
                guidance_scale=SD_GUIDANCE_SCALE,
            ).images[0]

        if saved:
            # 确保目录存在
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            image.save(saved_path, "JPEG", quality=95)
            print(f"[hf_gen] Saved to {saved_path}")
            return saved_path
        else:
            # 如果不需要保存，返回None并释放变量
            print("[hf_gen] Image generated (not saved).")
            return None

    except Exception as e:
        print(f"[hf_gen] An error occurred: {e}")
        return None


def description_refine(input_text, cls_name):
    """使用DeepSeek Chat润色描述 (替代OpenAI GPT-4-turbo)

    Args:
        input_text: 原始描述文本
        cls_name: 类别名称

    Returns:
        str: 润色后的描述，失败时返回原始文本
    """
    headers = get_deepseek_headers()

    user_content = (
        "This description does not seem to be representative of the class "
        f"{cls_name}. Could you refine it to enhance the distinctive "
        f"features of class {cls_name}"
    )

    payload = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "user", "content": input_text},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": DEEPSEEK_MAX_TOKENS
    }

    try:
        response = requests.post(
            DEEPSEEK_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        output = response.json()['choices'][0]['message']['content']
        return output
    except Exception as e:
        print(f"[hf_gen] description_refine failed: {e}")
        return input_text


def get_cls_index_name(label_index):
    """获取类别索引名称 (从ImageNet类别映射文件中读取)

    Args:
        label_index: 类别标签索引 (int)

    Returns:
        str: 类别名称
    """
    # 尝试多个可能的路径
    possible_paths = [
        "data_txt/ImageNet_LT/ImageNet_cls_name.txt",
        os.path.join(DATA_DIR, "data_txt/ImageNet_LT/ImageNet_cls_name.txt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as file:
                labels = [label.strip('",') for label in file.read().splitlines()]

            if 0 <= label_index < len(labels):
                return labels[label_index]
            else:
                return f"class_{label_index}"

    # 找不到文件时返回默认名
    return f"class_{label_index}"


def get_cls_template(cls_name, cls_index, filename="data_txt/ImageNet_LT/class_templates.txt"):
    """获取或生成类别模板 (使用DeepSeek Chat代替GPT-4-turbo)

    Args:
        cls_name: 类别名称
        cls_index: 类别索引 (int)
        filename: 模板文件路径

    Returns:
        str: 类别模板文本
    """
    headers = get_deepseek_headers()

    # 先尝试从文件读取已有模板
    if os.path.exists(filename):
        try:
            with open(filename, "r") as file:
                for line in file:
                    index_str, saved_template = line.strip().split(':', 1)
                    if int(index_str) == cls_index:
                        return saved_template
        except (FileNotFoundError, ValueError):
            pass

    # 文件不存在或索引不匹配，使用DeepSeek生成新模板
    template = (
        "Template: A photo of the class "
        f"{cls_name} with {{feature 1}}{{feature 2}}{{...}}."
    )
    user_content = (
        "Please use the Template to summarize the most distinctive "
        f"features of class {cls_name}"
    )

    payload = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "user", "content": template},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": DEEPSEEK_MAX_TOKENS
    }

    try:
        response = requests.post(
            DEEPSEEK_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        output = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[hf_gen] get_cls_template API failed: {e}")
        return template

    # 保存新生成的模板到文件
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, "a") as file:
        file.write(f"{cls_index}:{output}\n")

    return output