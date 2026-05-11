"""
视觉语言模型模块 (LLaVA)
负责图像编码和描述生成
"""
import torch
import time
import re
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

from config import LOCAL_VLM_ID, VLM_MAX_TOKENS


_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print(f"[vision_llm] Loading LLaVA model: {LOCAL_VLM_ID} ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        _model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_VLM_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
    except torch.cuda.OutOfMemoryError:
        print("[vision_llm] GPU OOM, falling back to CPU (float32)...")
        torch.cuda.empty_cache()
        _model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_VLM_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    _processor = AutoProcessor.from_pretrained(LOCAL_VLM_ID)
    print("[vision_llm] LLaVA model loaded.")
    return _model, _processor


def _tensor_to_pil(tensor):
    """将 PyTorch 图像张量转为 PIL Image (CxHxW, [0,1])"""
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    img_np = tensor.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(img_np).convert('RGB')


def describe_image(image_tensor, text_prompt, max_retries=2):
    """使用 LLaVA 模型进行图像理解，返回描述文本

    Args:
        image_tensor: PyTorch 张量 (CxHxW), 值范围 [0, 1]
        text_prompt: 文本提示词
        max_retries: 最大重试次数

    Returns:
        str: 生成的描述文本，失败时返回空字符串
    """
    for attempt in range(max_retries):
        try:
            model, processor = _load_model()
            image = _tensor_to_pil(image_tensor)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            input_ids = inputs['input_ids'].to(model.device)
            pixel_values = inputs['pixel_values'].to(model.device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=VLM_MAX_TOKENS,
                    do_sample=False,
                )

            response = processor.decode(
                output[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except torch.cuda.OutOfMemoryError:
            print(f"[vision_llm] CUDA OOM (attempt {attempt + 1}/{max_retries})")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[vision_llm] CUDA OOM (attempt {attempt + 1}/{max_retries})")
                torch.cuda.empty_cache()
            else:
                print(f"[vision_llm] Runtime error: {e}")
                break
        except Exception as e:
            print(f"[vision_llm] Inference failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

