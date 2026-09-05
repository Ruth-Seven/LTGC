"""Top-level CLIP refinement batch pipeline."""
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from pipeline.augmentation.storage import write_json
from .api import refine_batch
from .resampling import redraw_tasks
from .storage import (RefinementPaths, accept_candidate, load_initial_tasks,
    load_round_results, publish_scores, save_failed_state, write_task_shards)


def estimate_cost(low_count, expected_pass_rate=.797, max_rounds=3):
    failure_rate = 1 - expected_pass_rate
    expected_by_round = [
        round(low_count * failure_rate ** round_index)
        for round_index in range(1, max_rounds)
    ]
    expected_api_calls = sum(expected_by_round)
    peak_per_call = (300 * .44 + 60 * 1.32) / 1_000_000
    return dict(low_score_images=low_count,
        expected_api_calls_by_round=expected_by_round,
        estimated_api_calls=expected_api_calls,
        estimated_peak_usd=expected_api_calls * peak_per_call,
        pessimistic_peak_usd=low_count * (max_rounds - 1) * peak_per_call)


def plan(config):
    paths = RefinementPaths(Path(config.pipeline.run_dir))
    tasks = load_initial_tasks(paths, config.threshold)
    report = estimate_cost(len(tasks), max_rounds=config.max_rounds)
    write_json(paths.root / "plan.json", report)
    return report


def run_gpu_round(config_file, config, paths, tasks, round_number):
    gpus = config.pipeline.generation.gpus
    write_task_shards(paths, tasks, gpus, round_number)
    processes = []
    for gpu in gpus:
        log = (paths.round_dir(round_number) / f"gpu_{gpu}.log").open("a")
        command = [sys.executable, "-m", "pipeline.augmentation.refinement.cli", "worker",
                   "--config", str(config_file), "--gpu", str(gpu), "--round", str(round_number)]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[3])
        processes.append((gpu, log, process))
    failures = []
    for gpu, log, process in processes:
        if process.wait() != 0:
            failures.append(gpu)
        log.close()
    if failures:
        raise RuntimeError(f"GPU refinement workers failed: {failures}")
    return load_round_results(paths, gpus, round_number)


def apply_round(paths, tasks, results, threshold):
    by_sample = {result.sample_id:result for result in results}
    pending = []
    for task in tasks:
        attempt = by_sample[task.sample_id].attempt
        if attempt.score >= threshold:
            accept_candidate(paths, task, attempt)
        else:
            save_failed_state(paths, task, attempt)
            pending.append(replace(task, attempts=(*task.attempts, attempt)))
    return pending


def prepare_round_tasks(tasks, config, paths, round_number):
    redrawn = redraw_tasks(
        tasks, paths, config.pipeline.sampling.seed, round_number)
    prepared = []
    for task in redrawn:
        candidate = paths.candidates / f"round_{round_number:03d}" / task.record.sample.label / f"{task.sample_id}.png"
        seed = task.record.sample.generation_seed
        if task.attempts:
            seed += config.seed_stride
        sample = replace(task.record.sample, generation_seed=seed)
        record = replace(task.record, sample=sample)
        prepared.append(replace(task, record=record, round_number=round_number,
                                candidate_path=str(candidate)))
    return refine_batch(prepared, config, paths)


def process_round(config_file, config, paths, tasks, round_number):
    results = run_gpu_round(config_file, config, paths, tasks, round_number)
    failed = apply_round(paths, tasks, results, config.threshold)
    publish_scores(paths, config.threshold)
    if not failed:
        return []
    if round_number >= config.max_rounds:
        return []
    return prepare_round_tasks(failed, config, paths, round_number + 1)


def assert_dataset_not_training(paths):
    for command_file in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = command_file.read_bytes().replace(bytes([0]), b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError):
            continue
        if "train_multi_crop_paco" in command and str(paths.run) in command:
            raise RuntimeError("The augmented dataset is being trained; stop it before replacing images")


def prepare_pending_rounds(tasks, config, paths):
    grouped = {}
    for task in tasks:
        round_number = (task.attempts[-1].round_number + 1
                        if task.attempts else 1)
        if round_number <= config.max_rounds:
            grouped.setdefault(round_number, []).append(task)
    pending = {}
    for round_number, round_tasks in grouped.items():
        pending[round_number] = prepare_round_tasks(
            round_tasks, config, paths, round_number)
    return pending


def run(config_file, config):
    paths = RefinementPaths(Path(config.pipeline.run_dir))
    assert_dataset_not_training(paths)
    tasks = load_initial_tasks(paths, config.threshold)
    pending_rounds = prepare_pending_rounds(tasks, config, paths)
    while pending_rounds:
        round_number = min(pending_rounds)
        round_tasks = pending_rounds.pop(round_number)
        next_tasks = process_round(
            config_file, config, paths, round_tasks, round_number)
        if next_tasks:
            pending_rounds.setdefault(
                round_number + 1, []).extend(next_tasks)
    return publish_scores(paths, config.threshold)
