"""One GPU worker: generate candidates, unload Flux, then batch-score CLIP."""
import os
from pathlib import Path
from pipeline.augmentation.image_backend import DiffusersBackend
from pipeline.augmentation.semantics import clip_text
from pipeline.augmentation.storage import read_json, write_json
from .records import CandidateResult, RefinementAttempt, RefinementTask


def generate_candidates(tasks, generation_config):
    backend = DiffusersBackend(generation_config)
    try:
        for task in tasks:
            candidate = Path(task.candidate_path)
            if candidate.exists():
                continue
            candidate.parent.mkdir(parents=True, exist_ok=True)
            result = backend.render(task.generation_job())
            if result.image.size != (generation_config.size, generation_config.size):
                raise ValueError(f"Unexpected output size for {task.sample_id}: {result.image.size}")
            result.image.save(candidate)
    finally:
        backend.close()


def score_candidates(tasks, batch_size, score_batch=None):
    if score_batch is None:
        from model.clip_score import score_batch
    scores = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        scores.extend(score_batch(
            [task.candidate_path for task in batch],
            [clip_text(task.record.sample) for task in batch]))
    return scores


def run_worker(config, paths, gpu, round_number):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    task_file = paths.round_dir(round_number) / f"gpu_{gpu}_tasks.json"
    tasks = [RefinementTask.from_dict(value) for value in read_json(task_file)]
    if not tasks:
        write_json(paths.round_dir(round_number) / f"gpu_{gpu}_results.json", [])
        return
    generate_candidates(tasks, config.pipeline.generation)
    scores = score_candidates(tasks, config.pipeline.generation.clip_batch_size)
    results = []
    for task, score in zip(tasks, scores):
        attempt = RefinementAttempt(task.round_number, task.seed, task.current_description,
                                    task.current_prompt, task.candidate_path, float(score))
        results.append(CandidateResult(task.sample_id, attempt).to_dict())
    write_json(paths.round_dir(round_number) / f"gpu_{gpu}_results.json", results)
