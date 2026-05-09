"""
CLIP 质量筛选模块
计算生成图像与文本描述的余弦相似度
"""
import torch
from PIL import Image


_model = None
_preprocess = None
_device = None


def _load():
    global _model, _preprocess, _device
    if _model is not None:
        return _model, _preprocess, _device

    import clip
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip_score] Loading CLIP (ViT-B/32) on {_device}...")
    _model, _preprocess = clip.load("ViT-B/32", device=_device)
    print("[clip_score] CLIP loaded.")
    return _model, _preprocess, _device


def score(image_path, text):
    """计算生成图像与文本的余弦相似度

    Args:
        image_path: 图像文件路径
        text: 文本描述

    Returns:
        float: 余弦相似度 (0~1)
    """
    import clip
    model, preprocess, device = _load()

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    score = similarity.item()
    print(f"[clip_score] {score:.4f}")
    return score
