"""CLIP evaluation and metrics export, independent of image synthesis."""
import csv
from collections import Counter
from .semantics import clip_text
from .storage import write_json


def evaluate(jobs, config, paths, score_batch=None):
    if score_batch is None:
        from model.clip_score import score_batch
    scores = []
    for start in range(0, len(jobs), config.clip_batch_size):
        batch = jobs[start:start + config.clip_batch_size]
        scores.extend(score_batch(
            [str(job.output_path) for job in batch],
            [clip_text(job.record.sample) for job in batch]))
    with (paths.logs / 'generation_results.csv').open('w', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['sample_id', 'label', 'class_name', 'clip_score', 'passes_threshold', 'image_path', 'source_image', 'prompt'])
        writer.writerows((job.record.sample.sample_id, job.record.sample.label, job.record.sample.class_name,
            score, score >= config.clip_threshold, str(job.output_path), job.record.sample.image_path, job.prompt)
            for job, score in zip(jobs, scores))
    if config.two_references:
        with (paths.logs / 'context_pairs.csv').open('w', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(['sample_id', 'target_class', 'parent', 'target_image', 'context_class', 'context_image',
                             'source_context_description', 'used_context_description', 'extended_prompt', 'edit_prompt'])
            for job in jobs:
                source, father = job.record.sample, job.record.sample.father
                writer.writerow((source.sample_id, source.class_name, source.parent_category, source.image_path,
                    father.class_name, father.image_path, father.descriptions[0],
                    job.record.scene if config.prompt_style == 'short' else father.descriptions[0],
                    job.record.description, job.prompt))
    summary = dict(generated=len(jobs), counts=dict(Counter(job.record.sample.label for job in jobs)),
        clip_threshold=config.clip_threshold, clip_passed=sum(score >= config.clip_threshold for score in scores))
    write_json(paths.logs / 'summary.json', summary)
    return summary
