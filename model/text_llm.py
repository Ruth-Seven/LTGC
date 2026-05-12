"""
文本生成模块
使用纯文本 LLM (Qwen2.5-7B-Instruct) 进行描述扩展和精炼
替代之前的 LLaVA 视觉模型做纯文本生成
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TEXT_LLM_MODEL_ID, TEXT_LLM_MAX_TOKENS, TEXT_LLM_TEMPERATURE

_model = None
_tokenizer = None

SYSTEM_PROMPT = "You are a helpful assistant that generates diverse and detailed image descriptions for image classification datasets."


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"[text_llm] Loading {TEXT_LLM_MODEL_ID} ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _model = AutoModelForCausalLM.from_pretrained(
        TEXT_LLM_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    _tokenizer = AutoTokenizer.from_pretrained(TEXT_LLM_MODEL_ID)
    print("[text_llm] Model loaded.")
    return _model, _tokenizer


def _unload_model():
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache()
    print("[text_llm] Model unloaded.")


def _generate(messages, max_tokens=TEXT_LLM_MAX_TOKENS, temperature=TEXT_LLM_TEMPERATURE, do_sample=True, top_p=0.9):
    model, tokenizer = _load_model()

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def extend_descriptions(existing_texts, prompt, max_token=TEXT_LLM_MAX_TOKENS, temperature=TEXT_LLM_TEMPERATURE):
    """基于已有描述生成新的多样化描述

    Args:
        existing_texts: 已有描述文本列表
        prompt: 生成提示词
        max_token: 最大生成 token 数
        temperature: 生成温度 (越高越多样)

    Returns:
        list[str]: 新生成的描述列表
    """
    existing_block = "\n".join(f"- {t}" for t in existing_texts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing descriptions:\n{existing_block}\n\n{prompt}"},
    ]
    response = _generate(messages, max_tokens=max_token, temperature=temperature)

    sentences = re.split(r'\n\d+[\.\)]\s*|\n-\s*|\n', response)
    sentences = [s.strip() for s in sentences if s.strip() and s.strip().startswith('A photo')]
    if not sentences:
        sentences = [response]
    return sentences


def refection_descriptions(texts, prompt, max_token=TEXT_LLM_MAX_TOKENS, temperature=0.2):
    """去重/精炼描述列表

    Args:
        texts: 描述文本列表
        prompt: 精炼提示词
        max_token: 最大生成 token 数
        temperature: 生成温度

    Returns:
        list[str]: 精炼后的描述列表
    """
    existing_block = "\n".join(f"- {t}" for t in texts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing descriptions:\n{existing_block}\n\n{prompt}"},
    ]
    response = _generate(messages, max_tokens=max_token, temperature=temperature, do_sample=False)
    return [s.strip() for s in response.split("\n") if s.strip()]


def refine_description(text, class_name):
    """润色描述，使其更符合类别特征

    Args:
        text: 原始描述文本
        class_name: 类别名称

    Returns:
        str: 润色后的描述
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Refine this description to better describe a {class_name}:\n"
                f"'{text}'\n\n"
                f"Make it start with 'A photo of the class {class_name}' "
                f"and focus on distinctive visual features."
            ),
        },
    ]
    return _generate(messages, max_tokens=80, do_sample=False)


def generate_template(class_name):
    """生成类别模板描述

    Args:
        class_name: 类别名称

    Returns:
        str: 模板描述文本
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Describe the class '{class_name}' using this format:\n"
                f"'A photo of the class {class_name}, with [features], in [setting].'"
            ),
        },
    ]
    return _generate(messages, max_tokens=50, do_sample=False)
