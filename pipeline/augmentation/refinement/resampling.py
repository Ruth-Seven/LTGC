"""Deterministically resample only father references for each round."""
import hashlib
import random
from dataclasses import dataclass, replace
from pipeline.augmentation.records import ExtendedSample, ImageReference
from pipeline.augmentation.storage import read_json


@dataclass(frozen=True)
class FatherReferencePool:
    fathers_by_parent: dict[str, tuple[ImageReference, ...]]

    @classmethod
    def load(cls, paths):
        source = paths.run / "logs/extended_manifest_context_short.json"
        grouped = {}
        seen = set()
        for value in read_json(source):
            record = ExtendedSample.from_dict(value)
            sample = record.sample
            key = (sample.parent_category, sample.label, sample.image_path)
            if key in seen:
                continue
            seen.add(key)
            father = ImageReference(
                sample.label, sample.class_name, sample.image_path,
                [sample.source_prompt])
            grouped.setdefault(sample.parent_category, []).append(father)
        return cls({
            parent: tuple(fathers) for parent, fathers in grouped.items()
        })

    def draw(self, task, round_number, base_seed):
        current = task.record.sample
        options = [father for father in self.fathers_by_parent[
            current.parent_category] if father.label != current.label]
        if not options:
            raise ValueError(
                f"No father candidates for parent {current.parent_category}")
        alternatives = [
            father for father in options
            if father.image_path != current.father.image_path
        ]
        selected = random.Random(self._seed(
            task.sample_id, round_number, base_seed)).choice(
                alternatives or options)
        sample = replace(current, father=selected)
        return replace(task, record=replace(task.record, sample=sample))

    @staticmethod
    def _seed(sample_id, round_number, base_seed):
        digest = hashlib.sha256(sample_id.encode()).digest()
        return base_seed + round_number * 1_000_003 + int.from_bytes(
            digest[:8], "big")


def redraw_tasks(tasks, paths, base_seed, round_number=1):
    pool = FatherReferencePool.load(paths)
    return [
        pool.draw(task, round_number, base_seed) for task in tasks
    ]
