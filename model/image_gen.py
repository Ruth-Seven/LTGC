"""
图像生成模块
支持 SD v1-5 和 SDXL，通过 config.SD_MODEL_VERSION 切换
"""
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import os

from config import SD_MODEL_VERSION, SD_V1_5_PATH, SDXL_PATH, SD_IMAGE_SIZE, SD_NUM_INFERENCE_STEPS, SD_GUIDANCE_SCALE


_pipe = None
_version = None


def _get_pipeline():
    global _pipe, _version
    if _pipe is not None:
        return _pipe, _version

    if SD_MODEL_VERSION == "sdxl":
        model_path = SDXL_PATH
        pipeline_cls = StableDiffusionXLPipeline
    else:
        model_path = SD_V1_5_PATH
        pipeline_cls = StableDiffusionPipeline

    print(f"[image_gen] Loading {SD_MODEL_VERSION}: {model_path} ...")
    _pipe = pipeline_cls.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        _pipe = _pipe.to("cuda")
    _version = SD_MODEL_VERSION
    print(f"[image_gen] {SD_MODEL_VERSION} loaded.")
    return _pipe, _version


def generate(prompt, save_path=None):
    """从文本描述生成图像

    Args:
        prompt: 文本提示词
        save_path: 保存路径，None 时不保存

    Returns:
        str or None: 保存路径 (成功) 或 None (失败)
    """
    try:
        pipe, version = _get_pipeline()

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=SD_NUM_INFERENCE_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE,
        )
        if version == "sdxl":
            kwargs["height"] = SD_IMAGE_SIZE
            kwargs["width"] = SD_IMAGE_SIZE

        with torch.no_grad():
            image = pipe(**kwargs).images[0]

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path, "JPEG", quality=95)
            print(f"[image_gen] Saved: {save_path}")
            return save_path

        return None

    except Exception as e:
        print(f"[image_gen] Failed: {e}")
        return None


if __name__ == "__main__":
    # 测试代码
    test_prompt = "1girl, solo, wearing a beautiful floral kimono, standing under a glowing cherry blossom tree, night sky full of stars, glowing butterflies, studio ghibli style, makoto shinkai, vibrant colors, masterpiece, best quality, ultra-detailed, beautiful lighting."
    test_prompt = "epic majestic landscape, ancient ruins overgrown with bioluminescent plants, floating glowing crystals, misty mountains in the background, golden hour, fantasy concept art, trending on artstation, greg rutkowski, 8k resolution, unreal engine 5 render, volumetric fog."
    test_prompt = "cyberpunk street corner, raining, neon signs reflecting in puddles, a futuristic flying car passing by, moody atmosphere, cinematic lighting, octane render, photorealistic, volumetric fog, neon pink and cyan color palette, high detail."
    generate(test_prompt, "./test_output.jpg")