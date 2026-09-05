"""DeepSeek adapter. API settings, budgeting and request context are isolated here."""
import fcntl
import os
from pathlib import Path
from typing import Protocol
from .configuration import ExtensionConfig
from .records import Sample
from .semantics import (reference_semantic_description,
                        sample_semantic_description, sample_semantic_label)
from .storage import RunPaths


class DescriptionBackend(Protocol):
    def extend(self, sample: Sample,
               target_description: str | None = None) -> tuple[list[str], dict]: ...
    def scene(self, sample: Sample) -> str: ...


class DeepSeekBackend:
    def __init__(self, config: ExtensionConfig, paths: RunPaths, prompts, logs_dir=None):
        from model import text_llm
        self.api, self.config, self.paths, self.prompts = text_llm, config, paths, prompts
        self.logs = Path(logs_dir) if logs_dir is not None else paths.logs
        self.logs.mkdir(parents=True, exist_ok=True)
        os.environ.update(TEXT_LLM_BACKEND='deepseek', DEEPSEEK_MODEL=config.model,
            DEEPSEEK_COST_LEDGER=str(self.logs / 'deepseek_cost_usd.txt'),
            DEEPSEEK_MAX_COST_USD=str(config.max_cost_usd),
            DEEPSEEK_USAGE_LOG=str(self.logs / 'deepseek_usage.jsonl'))
        text_llm._deepseek_client = text_llm._load_deepseek_client().with_options(max_retries=3, timeout=60.0)
        text_llm.set_system_prompt(prompts['extend']['system_prompt'])

    def check_budget(self):
        with (self.logs / 'deepseek_cost_usd.txt').open('a+') as ledger:
            fcntl.flock(ledger, fcntl.LOCK_SH)
            ledger.seek(0)
            spent = float(ledger.read().strip() or '0')
        if spent + .01 * self.config.workers > self.config.max_cost_usd:
            raise RuntimeError(f'DeepSeek budget reached (${self.config.max_cost_usd})')

    def extend(self, sample, target_description=None):
        self.check_budget()
        self.api.set_deepseek_sample_id(sample.sample_id)
        source_description = target_description or sample.source_prompt
        anchor = sample_semantic_description(sample, source_description)
        semantic = sample_semantic_label(sample)
        name, sense = semantic.rsplit(" (", 1)
        reference = dict(sample.father.to_dict(), descriptions=[
            reference_semantic_description(sample.father)])
        result = self.api.extend_descriptions([anchor],
            prompt=self.prompts['extend']['extension_prompt'].format(
                number=1, name=name, category=sense[:-1]),
            number=1, prompt_instances=[reference], enable_thinking=False,
            max_token=self.config.max_tokens, temperature=self.config.temperature)
        return result, dict(target_description=anchor, father_instance=reference)

    def scene(self, sample):
        self.check_budget()
        self.api.set_deepseek_sample_id(sample.sample_id)
        text = self.api._generate([
            {'role':'system', 'content':self.prompts['extend']['system_prompt']},
            {'role':'user', 'content':self.prompts['generate']['context_scene_prompt'].format(description=sample.father.descriptions[0])}],
            max_tokens=self.config.scene_max_tokens, temperature=self.config.scene_temperature,
            do_sample=False, enable_thinking=False)
        return text.strip().splitlines()[0].strip().strip("'\". ")
