"""
CLIP 质量筛选模块
支持两种后端:
  - "openai": 原生 OpenAI clip 库
  - "huggingface": transformers CLIPModel
"""
import torch
import os
import logging
from PIL import Image

from config import CLIP_BACKEND, CLIP_MODEL_NAME

_log = logging.getLogger("clip_score")


# ── OpenAI backend ─────────────────────────────────────────────

_openai_model = None
_openai_preprocess = None


def _load_openai():
    global _openai_model, _openai_preprocess, _device
    if _openai_model is not None:
        return _openai_model, _openai_preprocess, _device

    import clip
    _device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    _log.info("Loading OpenAI CLIP (ViT-B/32) on %s...", _device)
    _openai_model, _openai_preprocess = clip.load("ViT-B/32", device=_device)
    _log.info("CLIP loaded.")
    return _openai_model, _openai_preprocess, _device


def _score_openai(image_path, text):
    import clip
    model, preprocess, device = _load_openai()

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    return similarity.item()


# ── HuggingFace backend ───────────────────────────────────────

_hf_model = None
_hf_processor = None


def _load_hf():
    global _hf_model, _hf_processor, _device
    if _hf_model is not None:
        return _hf_model, _hf_processor, _device

    from transformers import CLIPModel, CLIPProcessor
    _device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    _log.info("Loading HF CLIP (%s) on %s...", CLIP_MODEL_NAME, _device)
    _hf_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(_device)
    _hf_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
    _log.info("CLIP loaded.")
    return _hf_model, _hf_processor, _device


def _score_hf(image_path, text):
    model, processor, device = _load_hf()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        similarity = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)

    return similarity.item()


def _score_hf_batch(image_paths, texts):
    model, processor, device = _load_hf()

    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        similarities = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
    return similarities.tolist()


# ── Dispatch ───────────────────────────────────────────────────

_device = None


def score(image_path, text):
    """计算生成图像与文本的余弦相似度

    Args:
        image_path: 图像文件路径
        text: 文本描述

    Returns:
        float: 余弦相似度 (0~1)
    """
    if CLIP_BACKEND == "huggingface":
        s = _score_hf(image_path, text)
    else:
        s = _score_openai(image_path, text)

    _log.info("score image=%s text=%r score=%.4f", image_path, text, s)
    return s


def score_batch(image_paths, texts):
    """批量计算生成图像与文本的余弦相似度

    Args:
        image_paths: 图像文件路径列表
        texts: 文本描述列表

    Returns:
        list: 余弦相似度列表 (0~1)
    """
    if CLIP_BACKEND == "huggingface":
        scores = _score_hf_batch(image_paths, texts)
    else:
        raise ValueError("Batch scoring is only supported for HuggingFace backend")

    if scores:
        _log.info(
            "score_batch n=%d min=%.4f max=%.4f avg=%.4f",
            len(scores), min(scores), max(scores), sum(scores) / len(scores),
        )
        _log.info("score_batch scores=%s", [f"{s:.4f}" for s in scores])
    else:
        _log.info("score_batch n=0")
    return scores
