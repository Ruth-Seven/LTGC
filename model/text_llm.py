"""
文本生成模块
使用本地 Qwen3-8B 模型进行文本生成。
"""
import torch
import re
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TEXT_LLM_MODEL_ID, TEXT_LLM_MAX_TOKENS, TEXT_LLM_TEMPERATURE,
)

_log = logging.getLogger("text_llm")

_model = None
_tokenizer = None

SYSTEM_PROMPT = "You are a helpful assistant that generates diverse and detailed image descriptions for image classification datasets."


def set_system_prompt(prompt):
    """覆盖默认 SYSTEM_PROMPT（用于 prompt 文件自定义）"""
    global SYSTEM_PROMPT
    if prompt:
        SYSTEM_PROMPT = prompt


# ── Local backend ──────────────────────────────────────────────

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _log.info(f"Loading local model {TEXT_LLM_MODEL_ID} ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _model = AutoModelForCausalLM.from_pretrained(
        TEXT_LLM_MODEL_ID,
        torch_dtype=dtype,
        device_map={"": torch.cuda.current_device()},
    )
    _tokenizer = AutoTokenizer.from_pretrained(TEXT_LLM_MODEL_ID)
    _log.info("Model loaded.")
    return _model, _tokenizer


def _unload_model():
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    torch.cuda.empty_cache()
    _log.info("Model unloaded.")


def _generate_local(messages, max_tokens, temperature, do_sample, top_p,
                    enable_thinking=None):
    model, tokenizer = _load_model()

    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
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


# ── Response 清理 ────────────────────────────────────────────────

def _strip_thinking(response: str) -> str:
    """移除 Qwen3 thinking 模式的 <think>...</think> 推理块"""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


# ── Dispatch ───────────────────────────────────────────────────

def _generate(messages, max_tokens=TEXT_LLM_MAX_TOKENS,
              temperature=TEXT_LLM_TEMPERATURE, do_sample=True, top_p=0.9,
              enable_thinking=None):
    return _generate_local(
        messages, max_tokens, temperature, do_sample, top_p,
        enable_thinking=enable_thinking,
    )


# ── Public API ─────────────────────────────────────────────────

def extend_descriptions(existing_texts, prompt, number, enable_thinking=False, max_token=TEXT_LLM_MAX_TOKENS, temperature=TEXT_LLM_TEMPERATURE):
    """基于已有描述生成新的多样化描述，截断到 number"""
    existing_block = "\n".join(f"- {t}" for t in existing_texts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing descriptions:\n{existing_block}\n\n{prompt}"},
    ]
    response = _generate(
        messages,
        max_tokens=max_token,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    sentences = re.split(r'\n\d+[\.\)]\s*|\n-\s*|\n', response)
    result = []
    for s in sentences:
        s = re.sub(r'^\d+[\.\)]\s*', '', s.strip())
        if s.startswith('A photo'):
            s = s.split('\n')[0].strip()
            if s and s.count('[') == 0 and s.count(']') == 0:
                result.append(s)
    result = list(dict.fromkeys(result))
    _log.info(f"extend: parsed {len(result)} descriptions, returning {min(len(result), number)}")
    if not result: 
        _log.warning("extend: no valid descriptions generated, returning empty list\n Original response was:\n" + response + "\n\n")
        return []
    return result[:number]


def determine_descriptions(existing_texts, prompt, enable_thinking=False, max_token=TEXT_LLM_MAX_TOKENS, temperature=0.2, do_sample=False):
    """返回需要 reflection 的 1-based description 序号。"""
    existing_block = ""
    for idx, text in enumerate(existing_texts):
        existing_block += f"{idx + 1}. {text}\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing descriptions:\n{existing_block}\n\n{prompt}"},
    ]
    response = _generate(
        messages,
        max_tokens=max_token,
        temperature=temperature,
        do_sample=do_sample,
        enable_thinking=enable_thinking,
    )

    result = [int(sentence) for sentence in re.findall(r"\b\d+\b", response)]
    result = list(dict.fromkeys(result))
    result = [num for num in result if 1 <= num <= len(existing_texts)]
    _log.info(f"determine: have {len(result)} to reflect: {result}")

    _log.warning(f"@@@@@@@@@ {response}")

    return result

def reflection_descriptions(texts, prompt, number, enable_thinking=True, max_token=TEXT_LLM_MAX_TOKENS, temperature=0.2, do_sample=False):
    """去重/精炼描述列表，截断到 number"""
    existing_block = "\n".join(f"- {t}" for t in texts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing descriptions:\n{existing_block}\n\n{prompt}"},
    ]
    response = _generate(
        messages,
        max_tokens=max_token,
        temperature=temperature,
        do_sample=do_sample,
        enable_thinking=enable_thinking,
    )
    _log.info(f"reflection: {len(texts)} existing → raw response {len(response)} chars")

    sentences = re.split(r'\n\d+[\.\)]\s*|\n-\s*|\n', response)
    result = []
    for s in sentences:
        s = re.sub(r'^\d+[\.\)]\s*', '', s.strip())
        if s.startswith('A photo'):
            s = s.split('\n')[0].strip()
            result.append(s)
    result = list(dict.fromkeys(result))
    _log.info(f"reflection: parsed {len(result)} descriptions, returning {min(len(result), number)}")
    if not result: 
        _log.warning("reflection: no valid descriptions generated, returning empty list\n Original response was:\n" + response + "\n\n")
        return []
    return result[:number]


def reflect_one_description(description, class_name, prompt, enable_thinking=True, do_sample=False, temperature=0.2, max_token=1000):
    """润色描述

    Args:
        description: 待润色的描述文本
        class_name: 类别名称
        prompt: 自定义 prompt，支持 {class_name} 占位符
    """
    if prompt is None:
        raise Exception("reflect_one_description Empty.")
    user_content = description + "\n" + prompt.format(
        name=class_name,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = _generate(
        messages,
        max_tokens=max_token,
        temperature=temperature,
        do_sample=do_sample,
        enable_thinking=enable_thinking,
    )
    removed_thinking = _strip_thinking(response)
    # 提取第一条 "A photo of..." 描述，允许输出末尾没有换行
    m = re.search(r"(A photo of[^\n]*)(?:\n|$)", removed_thinking)
    if m:
        result = m.group(1).strip()
    else:
        result = ""
    if not result:
        _log.warning("reflect_one_description: no valid descriptions generated, returning empty string. Prompt: %s\nOriginal response was:\n%s\n\n", prompt, response)
        return ""
    _log.warning("reflect_one_description: %s. \n regenerate a description: %s", description, result)
    return result


def generate_template(class_name, prompt=None):
    """生成类别模板描述

    Args:
        class_name: 类别名称
        prompt: 自定义 prompt，支持 {class_name} 占位符。为 None 时使用内置默认
    """

    user_content = prompt.format(name=class_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = _generate(messages, max_tokens=50, do_sample=False)
    response = response.strip()
    if response.startswith('A photo'):
        response = response.split('\n')[0].strip()
    return response
