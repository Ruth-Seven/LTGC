"""Strict YAML configuration; stages receive typed settings, never argparse namespaces."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SourceConfig:
    descriptions: str
    progress: str
    class_mapping: str


@dataclass(frozen=True)
class SamplingConfig:
    target_per_class: int = 100
    seed: int = 20260903


@dataclass(frozen=True)
class ExtensionConfig:
    workers: int = 48
    model: str = 'deepseek-v4-flash'
    max_cost_usd: float = 24
    max_attempts: int = 3
    max_tokens: int = 100
    temperature: float = 0.7
    scene_max_tokens: int = 40
    scene_temperature: float = 0.2


@dataclass(frozen=True)
class GenerationConfig:
    model_path: str = '/data/hujunjie/models/flux/klein-4b'
    backend: str = 'flux-klein'
    gpus: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    size: int = 256
    steps: int = 4
    guidance: float = 1.0
    two_references: bool = True
    prompt_style: str = 'short'
    batch_size: int = 1
    clip_threshold: float = 0.22
    clip_batch_size: int = 5
    kontext_blocks_per_group: int = 29


@dataclass(frozen=True)
class ExecutionConfig:
    from_step: int = 1
    through_step: int = 3
    sample_limit: int | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    run_dir: str
    source: SourceConfig
    prompt_file: str
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text())
        result = cls.from_dict(data)
        def resolve(value):
            p = Path(value).expanduser()
            return str((path.parent / p).resolve()) if not p.is_absolute() else str(p)
        data = result.to_dict()
        data['run_dir'] = resolve(result.run_dir)
        data['prompt_file'] = resolve(result.prompt_file)
        data['source'] = {key: resolve(value) for key, value in data['source'].items()}
        data['generation']['model_path'] = resolve(result.generation.model_path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key, kind in [('source', SourceConfig), ('sampling', SamplingConfig),
                          ('extension', ExtensionConfig), ('generation', GenerationConfig),
                          ('execution', ExecutionConfig)]:
            if key in data:
                data[key] = kind(**data[key])
        result = cls(**data)
        if result.extension.workers < 1 or result.extension.max_cost_usd <= 0:
            raise ValueError('extension.workers and max_cost_usd must be positive')
        if result.extension.model != 'deepseek-v4-flash':
            raise ValueError('This pipeline is configured for the approved deepseek-v4-flash backend')
        if result.generation.backend not in ('flux-klein', 'flux-kontext'):
            raise ValueError('generation.backend must be flux-klein or flux-kontext')
        if not result.generation.gpus or len(set(result.generation.gpus)) != len(result.generation.gpus):
            raise ValueError('generation.gpus must be a nonempty list of distinct GPU indices')
        if result.generation.batch_size != 1:
            raise ValueError('Multi-reference inference currently supports batch_size: 1')
        if not 1 <= result.execution.from_step <= result.execution.through_step <= 3:
            raise ValueError('execution requires 1 <= from_step <= through_step <= 3')
        return result

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
