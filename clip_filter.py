"""
CLIP 质量筛选模块
计算生成图像与文本描述的余弦相似度
CLIP 模型使用全局变量懒加载（单例模式）
"""
import torch
from PIL import Image


_model = None
_preprocess = None
_device = None


def _get_clip():
    global _model, _preprocess, _device
    if _model is None:
        import clip
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[clip_filter] Loading CLIP model (ViT-B/32) on {_device} ...")
        _model, _preprocess = clip.load("ViT-B/32", device=_device)
        print("[clip_filter] CLIP model loaded.")
    return _model, _preprocess, _device


def clip_filter(img_path, text):
    """计算生成图像与文本描述的余弦相似度

    Args:
        img_path: 图像文件路径
        text: 类别描述文本

    Returns:
        float: 余弦相似度分数
    """
    import clip
    model, preprocess, device = _get_clip()

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # Truncate long text for CLIP (max 77 tokens)
    text_tokens = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    score = cosine_similarity.item()
    print(f"[clip_filter] Cosine similarity: {score:.4f}")
    return score
