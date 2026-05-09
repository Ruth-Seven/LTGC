"""
图像生成模块
使用 Stable Diffusion v1-5 从文本描述生成图像
"""
import torch
from diffusers import StableDiffusionPipeline
import os

from config import SD_MODEL_ID, SD_MODEL_PATH, SD_IMAGE_SIZE, SD_NUM_INFERENCE_STEPS, SD_GUIDANCE_SCALE


_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    model_id = SD_MODEL_PATH if os.path.exists(SD_MODEL_PATH) else SD_MODEL_ID
    print(f"[image_gen] Loading Stable Diffusion: {model_id} ...")
    _pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        _pipe = _pipe.to("cuda")
    print("[image_gen] SD model loaded.")
    return _pipe


def generate(prompt, save_path=None):
    """从文本描述生成图像

    Args:
        prompt: 文本提示词
        save_path: 保存路径，None 时不保存

    Returns:
        str or None: 保存路径 (成功) 或 None (失败)
    """
    try:
        pipe = _get_pipeline()

        if len(prompt) > 500:
            prompt = prompt[:500]

        with torch.no_grad():
            image = pipe(
                prompt,
                height=SD_IMAGE_SIZE,
                width=SD_IMAGE_SIZE,
                num_inference_steps=SD_NUM_INFERENCE_STEPS,
                guidance_scale=SD_GUIDANCE_SCALE,
            ).images[0]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path, "JPEG", quality=95)
            print(f"[image_gen] Saved: {save_path}")
            return save_path

        return None

    except Exception as e:
        print(f"[image_gen] Failed: {e}")
        return None
