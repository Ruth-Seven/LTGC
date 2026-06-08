"""
视觉语言模型模块
支持 LLaVA / Qwen2-VL 双后端，通过 set_backend() 切换
"""
import torch
import time
import logging
from transformers import AutoProcessor
from PIL import Image

from config import LOCAL_VLM_ID, VLM_MAX_TOKENS, LLAVA_MODEL_ID, QWEN2VL_MODEL_ID

_log = logging.getLogger("vision_lmm")

try:
    from transformers import LlavaForConditionalGeneration
    _HAS_LLAVA = True
except ImportError:
    _HAS_LLAVA = False

try:
    from transformers import Qwen2VLForConditionalGeneration
    _HAS_QWEN2VL = True
except ImportError:
    _HAS_QWEN2VL = False


_model = None
_processor = None
_model_path = LOCAL_VLM_ID
_backend = None


def set_backend(backend):
    """设置 VLM 后端，同时自动选择对应的默认模型路径

    Args:
        backend: "llava" 或 "qwen2vl"
    """
    global _backend, _model_path
    if backend not in ("llava", "qwen2vl"):
        raise ValueError(f"Unknown VLM backend: {backend}, expected 'llava' or 'qwen2vl'")
    _backend = backend
    _model_path = LLAVA_MODEL_ID if backend == "llava" else QWEN2VL_MODEL_ID


def set_model_path(path):
    """设置 VLM 模型路径（必须在首次加载模型前调用）

    Args:
        path: HuggingFace 模型路径或本地目录
    """
    global _model_path, _model, _processor
    if _model is not None:
        _log.info("Unloading existing model (path changed)")
        del _model
        del _processor
        _model = None
        _processor = None
        torch.cuda.empty_cache()
    _model_path = path


def _load_model():
    global _model, _processor, _model_path, _backend
    if _model is not None:
        return _model, _processor

    if _backend is None:
        raise RuntimeError("[vision_llm] set_backend() must be called before first inference")

    if _backend == "qwen2vl":
        if not _HAS_QWEN2VL:
            raise ImportError("Qwen2VLForConditionalGeneration not available")
        ModelClass = Qwen2VLForConditionalGeneration
    elif _backend == "llava":
        if not _HAS_LLAVA:
            raise ImportError("LlavaForConditionalGeneration not available")
        ModelClass = LlavaForConditionalGeneration
    else:
        raise ValueError(f"Unknown backend: {_backend}")

    _log.info(f"Loading VLM model (backend={_backend}): {_model_path} ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        _model = ModelClass.from_pretrained(
            _model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
    except torch.cuda.OutOfMemoryError:
        _log.info("GPU OOM, falling back to CPU (float32)...")
        torch.cuda.empty_cache()
        _model = ModelClass.from_pretrained(
            _model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    _processor = AutoProcessor.from_pretrained(_model_path)
    _log.info("VLM model loaded.")
    return _model, _processor


def _tensor_to_pil(tensor):
    """将 PyTorch 图像张量转为 PIL Image (CxHxW, [0,1])"""
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    img_np = tensor.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(img_np).convert('RGB')


def describe_image(image_tensor, text_prompt, max_retries=2):
    """使用 VLM 模型进行图像理解，返回描述文本

    Args:
        image_tensor: PyTorch 张量 (CxHxW), 值范围 [0, 1]
        text_prompt: 文本提示词
        max_retries: 最大重试次数

    Returns:
        str: 生成的描述文本，失败时返回空字符串
    """
    results = describe_image_batch([image_tensor], [text_prompt], max_retries)
    return results[0] if results else ""


def _build_qwen2vl_messages(image, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def describe_image_batch(image_tensors, text_prompts, max_retries=2):
    """批量图像描述，一次 forward 处理多张图。

    Args:
        image_tensors: list of Tensor, 每个形状 (C,H,W), [0,1]
        text_prompts:  list of str, 与 image_tensors 一一对应
        max_retries:   最大重试次数

    Returns:
        list of str: 生成的描述列表，与输入一一对应，失败位置返回 ""
    """
    B = len(image_tensors)
    if B == 0:
        return []

    for attempt in range(max_retries):
        try:
            model, processor = _load_model()
            pil_images = [_tensor_to_pil(t) for t in image_tensors]

            if _backend == "qwen2vl":
                messages_list = [_build_qwen2vl_messages(img, prompt) for img, prompt in zip(pil_images, text_prompts)]
                texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages_list
                ]
                inputs = processor(
                    text=texts,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                )
            elif _backend == "llava":
                prompts = [f"USER: <image>\n{p}\nASSISTANT:" for p in text_prompts]
                inputs = processor(images=pil_images, text=prompts, return_tensors="pt", padding=True)
            else:
                raise ValueError(f"Unknown backend: {_backend}")

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            input_lengths = inputs["attention_mask"].sum(dim=1)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=VLM_MAX_TOKENS,
                    do_sample=False,
                )

            responses = [
                processor.decode(output[i][input_lengths[i]:], skip_special_tokens=True).strip()
                for i in range(B)
            ]
            return responses

        except torch.cuda.OutOfMemoryError:
            _log.warning(f"CUDA OOM batch (attempt {attempt + 1}/{max_retries})")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                _log.warning(f"CUDA OOM batch (attempt {attempt + 1}/{max_retries})")
                torch.cuda.empty_cache()
            else:
                _log.error(f"Runtime error: {e}")
                break
        except Exception as e:
            _log.warning(f"Batch inference failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    _log.info("All retries exhausted, returning empty batch")
    return [""] * B
