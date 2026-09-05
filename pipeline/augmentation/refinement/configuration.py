"""Typed configuration for CLIP-guided regeneration."""
from dataclasses import dataclass, replace
from pathlib import Path
import yaml
from pipeline.augmentation.configuration import PipelineConfig


@dataclass(frozen=True)
class RefinementConfig:
    pipeline: PipelineConfig
    threshold: float = 0.22
    api_batch_size: int = 64
    api_workers: int = 32
    seed_stride: int = 1_000_000
    max_rounds: int = 3
    max_cost_usd: float = 5.0

    @classmethod
    def load(cls, path):
        config_path = Path(path).resolve()
        source = yaml.safe_load(config_path.read_text())
        pipeline_path = Path(source.pop("pipeline_config"))
        if not pipeline_path.is_absolute():
            pipeline_path = config_path.parent / pipeline_path
        result = cls(PipelineConfig.load(pipeline_path), **source)
        result.validate()
        return result

    def validate(self):
        if self.threshold <= 0 or self.api_batch_size <= 0:
            raise ValueError("threshold and api_batch_size must be positive")
        if self.max_cost_usd <= 0 or self.seed_stride <= 0:
            raise ValueError("max_cost_usd and seed_stride must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.api_workers < 1:
            raise ValueError("api_workers must be positive")
    def step2_extension(self):
        """Reuse Step2 settings with a separate sequential-refinement budget."""
        return replace(self.pipeline.extension, workers=max(1, self.api_workers),
                       max_cost_usd=self.max_cost_usd)
