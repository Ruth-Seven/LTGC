"""
CLIP 质量筛选模块
支持两种后端:
  - "openai": 原生 OpenAI clip 库
  - "huggingface": transformers CLIPModel
"""
import torch
import os
from PIL import Image

from config import CLIP_BACKEND, CLIP_LOCAL_MODEL_PATH, CLIP_MODEL_NAME


# ── OpenAI backend ─────────────────────────────────────────────

_openai_model = None
_openai_preprocess = None


def _load_openai():
    global _openai_model, _openai_preprocess, _device
    if _openai_model is not None:
        return _openai_model, _openai_preprocess, _device

    import clip
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip_score] Loading OpenAI CLIP (ViT-B/32) on {_device}...")
    _openai_model, _openai_preprocess = clip.load("ViT-B/32", device=_device)
    print("[clip_score] CLIP loaded.")
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
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip_score] Loading HF CLIP ({CLIP_MODEL_NAME}) on {_device}...")
    _hf_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device)
    _hf_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("[clip_score] CLIP loaded.")
    return _hf_model, _hf_processor, _device


def _score_hf(image_path, text):
    model, processor, device = _load_hf()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = torch.sigmoid(logits_per_image)

    return similarity.item()


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

    print(f"[clip_score] {s:.4f}")
    return s
