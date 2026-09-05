"""Step1: deterministic same-parent pairing from real-image captions."""
import csv
import random
from collections import defaultdict
from pathlib import Path
from utils import load_class_semantics, parse_semantic_label, validate_description
from .configuration import SamplingConfig, SourceConfig
from .prompts import mapped_description
from .records import ImageReference, Sample
from .storage import RunPaths, write_json


def load_step1_sources(progress):
    parts = sorted(Path(progress).glob('success_gpu*.csv'))
    if not parts:
        raise FileNotFoundError(f'No Step1 success CSVs in {progress}')
    result = {}
    for part in parts:
        with part.open(newline='') as source:
            for label, image_path, caption in csv.reader(source):
                paths = result.setdefault((label, caption), [])
                if image_path not in paths:
                    paths.append(image_path)
    return result


def sample(paths: RunPaths, source: SourceConfig, config: SamplingConfig, limit=None):
    mapping = load_class_semantics(source.class_mapping)
    pool = defaultdict(dict)
    for (label, caption), image_paths in load_step1_sources(source.progress).items():
        if label in mapping and validate_description(mapped_description(caption, mapping[label]), mapping[label]):
            for image_path in image_paths:
                pool[label][image_path] = caption
    records = []
    for label in sorted(pool, key=int):
        name, parent, display = parse_semantic_label(mapping[label])
        siblings = sorted((other for other in pool if other != label and
                           parse_semantic_label(mapping[other])[1] == parent), key=int)
        target_new = max(0, config.target_per_class - len(pool[label]))
        if target_new and not siblings:
            raise ValueError(f'No same-parent reference for {display}')
        items = sorted(pool[label].items())
        rng = random.Random(config.seed + int(label))
        for index in range(target_new):
            image_path, caption = items[index % len(items)]
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
            sibling = rng.choice(siblings)
            father_path, father_caption = rng.choice(sorted(pool[sibling].items()))
            records.append(Sample(sample_id=f'{label}_{index:03d}', label=label,
                name=name, parent_category=parent, class_name=display, image_path=image_path,
                source_prompt=caption, generation_seed=config.seed + len(records),
                father=ImageReference(sibling, mapping[sibling], father_path, [father_caption])))
    records = records[:limit]
    if paths.samples.exists():
        raise FileExistsError(paths.samples)
    write_json(paths.samples, [record.to_dict() for record in records])
    write_json(paths.mapping, {label:mapping[label] for label in sorted(pool, key=int)})
    write_json(paths.logs / 'sampling_config.json', dict(seed=config.seed,
        step1_progress=source.progress, class_mapping=source.class_mapping,
        target_number=config.target_per_class, classes=len(pool), samples=len(records),
        source='existing Step1 successful real-image descriptions',
        eligibility='all classes; fill each to target_number with a strictly same-parent prompt instance'))
    if not paths.source_csv.exists():
        with paths.source_csv.open('w', newline='') as target:
            csv.writer(target).writerows((label, caption) for label in sorted(pool, key=int)
                for _, caption in sorted(pool[label].items()))
    return records
