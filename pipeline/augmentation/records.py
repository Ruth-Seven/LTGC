"""Domain records. Disk field names remain compatible with existing checkpoints."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageReference:
    label: str
    class_name: str
    image_path: str
    descriptions: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        keys = ('label', 'class_name', 'image_path', 'descriptions')
        return cls(**{key: value[key] for key in keys},
                   extra={key: val for key, val in value.items() if key not in keys})

    def to_dict(self):
        return dict(self.extra, label=self.label, class_name=self.class_name,
                    image_path=self.image_path, descriptions=self.descriptions)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    label: str
    name: str
    parent_category: str
    class_name: str
    image_path: str
    source_prompt: str
    generation_seed: int
    father: ImageReference
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        keys = ('sample_id', 'label', 'name', 'parent_category', 'class_name',
                'image_path', 'source_prompt', 'generation_seed')
        return cls(**{key: value[key] for key in keys},
                   father=ImageReference.from_dict(value['prompt_instance']),
                   extra={key: val for key, val in value.items() if key not in (*keys, 'prompt_instance')})

    def to_dict(self):
        return dict(self.extra, sample_id=self.sample_id, label=self.label, name=self.name,
                    parent_category=self.parent_category, class_name=self.class_name,
                    image_path=self.image_path, source_prompt=self.source_prompt,
                    generation_seed=self.generation_seed, prompt_instance=self.father.to_dict())


@dataclass(frozen=True)
class ExtendedSample:
    sample: Sample
    description: str
    scene: str

    @classmethod
    def from_dict(cls, value):
        return cls(Sample.from_dict(value), value['extended_prompt'], value.get('context_scene', ''))

    def to_dict(self):
        result = self.sample.to_dict()
        result['extended_prompt'] = self.description
        # Preserve old single-reference manifests that never had a scene field.
        if self.scene or 'context_scene' in result:
            result['context_scene'] = self.scene
        return result

    def csv_row(self):
        source, father = self.sample, self.sample.father
        return (source.label, self.description, father.label, father.class_name,
                father.descriptions[0], father.image_path, source.image_path)


@dataclass(frozen=True)
class GenerationJob:
    record: ExtendedSample
    prompt: str
    output_path: Path
    metadata_path: Path
    two_references: bool

    @property
    def image_paths(self):
        source = self.record.sample
        return [source.image_path, source.father.image_path] if self.two_references else [source.image_path]

    def to_dict(self):
        result = dict(self.record.to_dict(), kontext_prompt=self.prompt,
                      output_path=str(self.output_path), metadata_path=str(self.metadata_path))
        if self.two_references:
            result.update(input_image_paths=self.image_paths, input_image_roles=['target', 'same_parent_context'])
        return result
