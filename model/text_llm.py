"""
文本生成模块
使用本地 LLaVA 模型的 LLaMA 13B 骨干进行纯文本生成
替代 DeepSeek Chat API
"""
import torch
import time
import re
from transformers import AutoTokenizer, LlavaForConditionalGeneration

from config import LOCAL_VLM_ID


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"[text_llm] Loading text backbone from {LOCAL_VLM_ID} ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        _model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_VLM_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
    except torch.cuda.OutOfMemoryError:
        print("[text_llm] GPU OOM, falling back to CPU...")
        torch.cuda.empty_cache()
        _model = LlavaForConditionalGeneration.from_pretrained(
            LOCAL_VLM_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    _tokenizer = AutoTokenizer.from_pretrained(LOCAL_VLM_ID)
    print("[text_llm] Model loaded.")
    return _model, _tokenizer


def _unload_model():
    """卸载模型，释放 GPU 内存"""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache()
    print("[text_llm] Model unloaded.")


def extend_descriptions(existing_texts, prompt, max_teken=10000, temperature=0.8):
    """基于已有描述生成新的多样化描述

    Args:
        existing_texts: 已有描述文本列表
        prompt: 生成提示词
        max_new: 最大生成 token 数
        temperature: 生成温度 (越高越多样)

    Returns:
        list[str]: 新生成的描述列表
    """
    model, tokenizer = _load_model()

    existing_block = "\n".join(f"- {t}" for t in existing_texts)
    extend_prompt = (
        f"Existing descriptions:\n{existing_block}\n\n"
        f"{prompt}"
    )

    inputs = tokenizer(extend_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse generated descriptions
    sentences = re.split(r'\n\d+[\.\)]\s*|\n-\s*|\n', response)
    sentences = [s.strip() for s in sentences if s.strip() and s.strip().startswith('A photo')]

    if not sentences:
        sentences = [response.strip()]

    return sentences

def refection_descriptions(texts, prompt, max_token=10000, temperature=0.2):
    model, tokenizer = _load_model()

    refection_prompt =  "\n\nExisting Descriptions:\n" + "\n".join(f"- {t}" for t in texts) + prompt 
    inputs = tokenizer(refection_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_token,
            do_sample=False,
            temperature=temperature,
        )
    reponse = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return reponse.strip().split("\n")

def refine_description(text, class_name):
    """润色描述，使其更符合类别特征

    Args:
        text: 原始描述文本
        class_name: 类别名称

    Returns:
        str: 润色后的描述
    """
    model, tokenizer = _load_model()

    prompt = (
        f"Refine this description to better describe a {class_name}:\n"
        f"'{text}'\n\n"
        f"Make it start with 'A photo of the class {class_name}' and focus on distinctive visual features."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_template(class_name):
    """生成类别模板描述

    Args:
        class_name: 类别名称

    Returns:
        str: 模板描述文本
    """
    model, tokenizer = _load_model()

    prompt = (
        f"Describe the class '{class_name}' using this format:\n"
        f"'A photo of the class {class_name}, with [features], in [setting].'"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()
