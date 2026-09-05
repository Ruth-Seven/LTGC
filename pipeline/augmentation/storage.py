"""Filesystem contract. Auxiliary state stays below logs; public CSV columns are stable."""
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from .records import ExtendedSample, Sample


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root))

    @property
    def logs(self): return self.root / 'logs'
    @property
    def samples(self): return self.logs / 'sample_manifest.json'
    @property
    def extended(self): return self.logs / 'extended_manifest_context_short.json'
    @property
    def checkpoints(self): return self.logs / 'step2_progress'
    @property
    def failures(self): return self.logs / 'step2_failed'
    @property
    def prompt(self): return self.logs / 'prompt.json'
    @property
    def source_csv(self): return self.root / 'imagenet_lt_description_list.csv'
    @property
    def extended_csv(self): return self.root / 'imagenet_lt_extended_description_list.csv'
    @property
    def mapping(self): return self.root / 'class_semantic_mapping.json'
    @property
    def generation_logs(self): return self.logs / 'generation'
    @property
    def shard_dir(self): return self.logs / 'generation_shards'
    @property
    def images(self): return self.root / 'generated_imgs'

    def generation(self, gpu):
        return GenerationPaths(self.images / 'shards' / f'gpu_{gpu}',
                               self.generation_logs / f'gpu_{gpu}')


@dataclass(frozen=True)
class GenerationPaths:
    output: Path
    logs: Path

    def __post_init__(self):
        object.__setattr__(self, 'output', Path(self.output))
        object.__setattr__(self, 'logs', Path(self.logs))

    def image(self, sample): return self.output / 'train' / sample.label / f'{sample.sample_id}.png'
    def metadata(self, sample): return self.logs / 'metadata' / sample.label / f'{sample.sample_id}.json'


class Step2Store:
    def __init__(self, paths: RunPaths):
        self.paths = paths

    def samples(self):
        return [Sample.from_dict(row) for row in read_json(self.paths.samples)]

    def completed(self, samples):
        return [ExtendedSample.from_dict(read_json(path)) for sample in samples
                if (path := self.paths.checkpoints / f'{sample.sample_id}.json').exists()]

    def export(self, records):
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        target = self.paths.extended_csv
        temp = self.paths.logs / (target.name + '.tmp')
        with temp.open('w', newline='') as output:
            csv.writer(output).writerows(record.csv_row() for record in records)
        os.replace(temp, target)
        payload = [record.to_dict() for record in records]
        write_json(self.paths.logs / (target.name + '.provenance.json'), payload)
        with (self.paths.logs / (target.name + '.with_parent.csv')).open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['label', 'class_name', 'extended_description', 'source_description',
                             'source_image_path', 'parent_label', 'parent_class', 'parent_description', 'parent_image_path'])
            for record in records:
                source, father = record.sample, record.sample.father
                writer.writerow([source.label, source.class_name, record.description,
                                 source.source_prompt, source.image_path, father.label,
                                 father.class_name, father.descriptions[0], father.image_path])
        write_json(self.paths.extended, payload)
        write_json(self.paths.logs / 'extended_manifest.json', payload)
