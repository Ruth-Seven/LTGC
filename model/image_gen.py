"""
图像生成模块
支持 SD v1-5 和 SDXL，通过 config.SD_MODEL_VERSION 切换
"""
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import os

from config import SD_MODEL_VERSION, SD_V1_5_PATH, SDXL_PATH, SD_IMAGE_SIZE, SD_NUM_INFERENCE_STEPS, SD_GUIDANCE_SCALE, get_device


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
        _pipe = _pipe.to(get_device("sd"))
    _version = SD_MODEL_VERSION
    print(f"[image_gen] {SD_MODEL_VERSION} loaded.")
    return _pipe, _version


def generate(prompt, save_path=None):
    """从文本描述生成单张图像"""
    return generate_batch([prompt], [save_path] if save_path else None)[0]


def unload_sd():
    """卸载 SD pipeline 释放显存"""
    global _pipe, _version
    if _pipe is not None:
        _pipe = None
        _version = None
        torch.cuda.empty_cache()
        print("[image_gen] SD unloaded, cache cleared.")


def generate_batch(prompts, save_paths=None):
    """从文本描述批量生成图像，UNet batch 并行

    Args:
        prompts: 文本提示词列表
        save_paths: 保存路径列表，长度与 prompts 一致

    Returns:
        list: 保存路径列表 (成功项为路径，失败项为 None)
    """
    n = len(prompts)
    for attempt in range(2):
        try:
            pipe, version = _get_pipeline()

            kwargs = dict(
                prompt=prompts,
                num_inference_steps=SD_NUM_INFERENCE_STEPS,
                guidance_scale=SD_GUIDANCE_SCALE,
            )
            if version == "sdxl":
                kwargs["height"] = SD_IMAGE_SIZE
                kwargs["width"] = SD_IMAGE_SIZE

            with torch.no_grad():
                images = pipe(**kwargs).images

            results = []
            for i in range(n):
                path = save_paths[i] if save_paths else None
                if path and i < len(images):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    images[i].save(path, "JPEG", quality=95)
                    results.append(path)
                elif path:
                    results.append(None)
                else:
                    results.append(path)

            print(f"[image_gen] Batch generated {len(results)} images")
            return results

        except Exception as e:
            if attempt == 0:
                print(f"[image_gen] Batch failed (attempt 1/2): {e}")
                print(f"[image_gen] Unloading SD and retrying...")
                unload_sd()
            else:
                print(f"[image_gen] Batch failed (attempt 2/2): {e}")
                return [None] * n


if __name__ == "__main__":
    # 测试代码
    test_prompt = "1girl, solo, wearing a beautiful floral kimono, standing under a glowing cherry blossom tree, night sky full of stars, glowing butterflies, studio ghibli style, makoto shinkai, vibrant colors, masterpiece, best quality, ultra-detailed, beautiful lighting."
    test_prompt = "epic majestic landscape, ancient ruins overgrown with bioluminescent plants, floating glowing crystals, misty mountains in the background, golden hour, fantasy concept art, trending on artstation, greg rutkowski, 8k resolution, unreal engine 5 render, volumetric fog."
    test_prompt = "cyberpunk street corner, raining, neon signs reflecting in puddles, a futuristic flying car passing by, moody atmosphere, cinematic lighting, octane render, photorealistic, volumetric fog, neon pink and cyan color palette, high detail."
    generate(test_prompt, "./test_output.jpg")