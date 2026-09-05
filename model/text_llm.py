"""
文本生成模块
使用本地 Qwen3-8B 模型进行文本生成。
"""

import json
import logging
from datetime import datetime, timezone
import os
import re
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TEXT_LLM_MAX_TOKENS,
    TEXT_LLM_MODEL_ID,
    TEXT_LLM_TEMPERATURE,
)

_log = logging.getLogger("text_llm")

_model = None
_tokenizer = None
_deepseek_client = None
_deepseek_context = threading.local()

SYSTEM_PROMPT = "You are a helpful assistant that generates diverse and detailed image descriptions for image classification datasets."


def set_deepseek_sample_id(sample_id):
    """Attach audit metadata to the current API worker thread."""
    _deepseek_context.sample_id = sample_id


def get_deepseek_sample_id():
    return getattr(_deepseek_context, "sample_id", None)


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


def _generate_local(messages, max_tokens, temperature, do_sample, top_p, enable_thinking=None):
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

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


# ── DeepSeek API backend ────────────────────────────────────────


def _load_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            key = Path("~/.config/ltgc/deepseek_api_key").expanduser().read_text().strip()
        _deepseek_client = OpenAI(
            api_key=key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _deepseek_client


def deepseek_usage_cost(usage, model, created):
    """USD estimate from returned tokens and official peak/off-peak rates."""
    peak_rates = {"deepseek-v4-flash": (0.014, 0.44, 1.32),
                  "deepseek-v4-pro": (0.044, 1.32, 3.96)}[model]
    when = datetime.fromtimestamp(created, timezone.utc)
    peak = when.weekday() < 5 and (1 <= when.hour < 4 or 6 <= when.hour < 10)
    rates = tuple(rate if peak else rate / 2 for rate in peak_rates)
    hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if miss is None:
        miss = usage.prompt_tokens - hit
    tokens = (hit, miss, usage.completion_tokens)
    return dict(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens, cache_hit_tokens=hit, cache_miss_tokens=miss,
        period="peak" if peak else "off-peak", created_utc=when.isoformat(),
        rates_usd_per_million=dict(zip(("cache_hit", "cache_miss", "output"), rates)),
        estimated_cost_usd=sum(n * rate for n, rate in zip(tokens, rates)) / 1_000_000,
        conservative_cost_usd=sum(n * rate for n, rate in zip(tokens, peak_rates)) / 1_000_000,
        pricing_source="https://api-docs.deepseek.com/quick_start/pricing/",
        pricing_checked="2026-09-03")


def _record_deepseek_cost(cost):
    ledger_path = os.environ.get("DEEPSEEK_COST_LEDGER")
    ceiling = os.environ.get("DEEPSEEK_MAX_COST_USD")
    if not ledger_path:
        return
    import fcntl

    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.seek(0)
        total = float(output.read().strip() or 0) + cost["conservative_cost_usd"]
        output.seek(0)
        output.truncate()
        output.write(f"{total:.12f}\n")
        output.flush()
        fcntl.flock(output, fcntl.LOCK_UN)
    if ceiling and total > float(ceiling):
        raise RuntimeError(f"DeepSeek cost ceiling exceeded: ${total:.4f} > ${float(ceiling):.2f}")


def _generate_deepseek(messages, max_tokens, temperature, do_sample, top_p, enable_thinking=None):
    client = _load_deepseek_client()
    thinking = "enabled" if enable_thinking else "disabled"
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if do_sample else 0,
        top_p=top_p,
        stream=False,
        extra_body={"thinking": {"type": thinking}},
    )
    usage = response.usage
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
    _log.info(
        "deepseek usage prompt=%s completion=%s total=%s cache_hit=%s cache_miss=%s",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        cache_hit,
        cache_miss,
    )
    cost = deepseek_usage_cost(usage, model, response.created)
    usage_log = os.environ.get("DEEPSEEK_USAGE_LOG")
    if usage_log:
        import fcntl
        record = dict(cost, request_id=response.id, model=model, response_model=response.model,
            sample_id=get_deepseek_sample_id(), messages=messages,
            response_text=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason)
        with open(usage_log, "a") as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            fcntl.flock(output, fcntl.LOCK_UN)
    _record_deepseek_cost(cost)
    return (response.choices[0].message.content or "").strip()


# ── Response 清理 ────────────────────────────────────────────────


def _strip_thinking(response: str) -> str:
    """移除 Qwen3 thinking 模式的 <think>...</think> 推理块"""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _parse_numbered_lines(response: str):
    """Strip only line-leading list markers; preserve content hyphens."""
    result = []
    for line in _strip_thinking(response).splitlines():
        text = re.sub(r"^\s*(?:\d+[.)]\s*|-\s+)", "", line).strip()
        if text:
            result.append(text)
    return result


# ── Dispatch ───────────────────────────────────────────────────


def _generate(
    messages,
    max_tokens=TEXT_LLM_MAX_TOKENS,
    temperature=TEXT_LLM_TEMPERATURE,
    do_sample=True,
    top_p=0.9,
    enable_thinking=None,
):
    if os.environ.get("TEXT_LLM_BACKEND", "local") == "deepseek":
        return _generate_deepseek(
            messages,
            max_tokens,
            temperature,
            do_sample,
            top_p,
            enable_thinking=enable_thinking,
        )
    return _generate_local(
        messages,
        max_tokens,
        temperature,
        do_sample,
        top_p,
        enable_thinking=enable_thinking,
    )


# ── Public API ─────────────────────────────────────────────────


def extend_descriptions(
    existing_texts,
    prompt,
    number,
    prompt_instances=None,
    enable_thinking=False,
    max_token=TEXT_LLM_MAX_TOKENS,
    temperature=TEXT_LLM_TEMPERATURE,
):
    """基于目标类描述和跨类真实描述实例生成新描述，截断到 number。"""
    existing_block = "\n".join(f"- {t}" for t in existing_texts)
    content = f"Target-class descriptions:\n{existing_block}"
    if prompt_instances:
        instance_blocks = []
        for index, instance in enumerate(prompt_instances, 1):
            descriptions = "\n".join(f"- {text}" for text in instance["descriptions"])
            instance_blocks.append(f"— {instance['class_name']}:\n{descriptions}")
        content += "\n\nCross-class prompt:\n" + "\n".join(instance_blocks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task instructions:\n{prompt}\n\n{content}"},
    ]
    response = _generate(
        messages,
        max_tokens=max_token,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    sentences = _parse_numbered_lines(response)
    result = []
    for s in sentences:
        s = s.strip()
        if s.startswith("A photo"):
            s = s.split("\n")[0].strip()
            if s and s.count("[") == 0 and s.count("]") == 0:
                result.append(s)
    result = list(dict.fromkeys(result))
    _log.info(f"extend: parsed {len(result)} descriptions, returning {min(len(result), number)}")
    if not result:
        _log.warning(
            "extend: no valid descriptions generated, returning empty list\n Original response was:\n"
            + response
            + "\n\n"
        )
        return []
    return result[:number]


def determine_descriptions(
    existing_texts,
    prompt,
    enable_thinking=False,
    max_token=TEXT_LLM_MAX_TOKENS,
    temperature=0.2,
    do_sample=False,
):
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


def reflection_descriptions(
    texts,
    prompt,
    number,
    enable_thinking=True,
    max_token=TEXT_LLM_MAX_TOKENS,
    temperature=0.2,
    do_sample=False,
):
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

    sentences = _parse_numbered_lines(response)
    result = []
    for s in sentences:
        s = s.strip()
        if s.startswith("A photo"):
            s = s.split("\n")[0].strip()
            result.append(s)
    result = list(dict.fromkeys(result))
    _log.info(
        f"reflection: parsed {len(result)} descriptions, returning {min(len(result), number)}"
    )
    if not result:
        _log.warning(
            "reflection: no valid descriptions generated, returning empty list\n Original response was:\n"
            + response
            + "\n\n"
        )
        return []
    return result[:number]


def reflect_one_description(
    description,
    class_name,
    prompt,
    enable_thinking=True,
    do_sample=False,
    temperature=0.2,
    max_token=1000,
):
    """润色描述

    Args:
        description: 待润色的描述文本
        class_name: 类别名称
        prompt: 自定义 prompt，支持 {class_name} 占位符
    """
    if prompt is None:
        raise Exception("reflect_one_description Empty.")
    match = re.match(r"^\s*(.+?)\s*\(([^()]+)\)\s*$", class_name)
    if not match:
        raise ValueError(f"class_name must use name (category): {class_name!r}")
    user_content = (
        description
        + "\n"
        + prompt.format(
            name=match.group(1).strip(),
            category=match.group(2).strip(),
        )
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
        _log.warning(
            "reflect_one_description: no valid descriptions generated, returning empty string. Prompt: %s\nOriginal response was:\n%s\n\n",
            prompt,
            response,
        )
        return ""
    _log.warning(
        "reflect_one_description: %s. \n regenerate a description: %s", description, result
    )
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
    if response.startswith("A photo"):
        response = response.split("\n")[0].strip()
    return response
