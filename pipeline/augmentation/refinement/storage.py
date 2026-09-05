"""Filesystem contract and atomic publication for refinement."""
import csv
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from pipeline.augmentation.records import ExtendedSample
from pipeline.augmentation.storage import read_json, write_json
from .records import CandidateResult, RefinementAttempt, RefinementTask


@dataclass(frozen=True)
class RefinementPaths:
    run: Path

    def __post_init__(self):
        object.__setattr__(self, "run", Path(self.run))

    @property
    def root(self): return self.run / "logs/clip_refinement"
    @property
    def state(self): return self.root / "state"
    @property
    def candidates(self): return self.root / "candidates"
    @property
    def backups(self): return self.root / "backups"
    @property
    def scores(self): return self.run / "logs/generation/generation_results.csv"
    @property
    def inputs(self): return self.run / "logs/generation/generation_inputs.json"

    def state_file(self, sample_id): return self.state / f"{sample_id}.json"
    def round_dir(self, number): return self.root / f"round_{number:03d}"


def read_score_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def task_from_sources(generation_input, score_row, paths, round_number, attempts=()):
    record = ExtendedSample.from_dict(generation_input)
    sample_id = record.sample.sample_id
    candidate = paths.candidates / f"round_{round_number:03d}" / record.sample.label / f"{sample_id}.png"
    seed = record.sample.generation_seed + round_number * 1_000_000
    record = ExtendedSample(replace(record.sample, generation_seed=seed), record.description, record.scene)
    return RefinementTask(record, float(score_row["clip_score"]), record.description,
                          generation_input["kontext_prompt"], round_number, str(candidate), attempts)


def load_initial_tasks(paths, threshold):
    scores = {score["sample_id"]: score for score in read_score_rows(paths.scores)}
    inputs = {generation["sample_id"]: generation for generation in read_json(paths.inputs)}
    if set(scores) != set(inputs):
        raise ValueError("Generation scores and inputs have different sample IDs")
    tasks = []
    for sample_id, score in scores.items():
        state_file = paths.state_file(sample_id)
        state = read_json(state_file) if state_file.exists() else {}
        if state.get("passed"):
            continue
        if float(score["clip_score"]) >= threshold:
            continue
        attempts = tuple(RefinementAttempt.from_dict(value)
                         for value in state.get("attempts", []))
        task = task_from_sources(inputs[sample_id], score, paths, 1, attempts)
        if attempts:
            last = attempts[-1]
            if state.get("record"):
                state_record = ExtendedSample.from_dict(state["record"])
                sample = replace(
                    task.record.sample, generation_seed=last.seed,
                    father=state_record.sample.father)
                record = replace(
                    task.record, sample=sample,
                    description=last.description, scene=state_record.scene)
            else:
                sample = replace(task.record.sample, generation_seed=last.seed)
                record = replace(task.record, sample=sample,
                                 description=last.description)
            task = replace(task, record=record,
                           current_description=last.description,
                           current_prompt=last.prompt,
                           round_number=last.round_number,
                           candidate_path=last.candidate_path)
        tasks.append(task)
    return tasks


def write_task_shards(paths, tasks, gpus, round_number):
    round_dir = paths.round_dir(round_number)
    for index, gpu in enumerate(gpus):
        shard = tasks[index::len(gpus)]
        write_json(round_dir / f"gpu_{gpu}_tasks.json", [task.to_dict() for task in shard])
    return round_dir


def load_round_results(paths, gpus, round_number):
    results = []
    for gpu in gpus:
        source = paths.round_dir(round_number) / f"gpu_{gpu}_results.json"
        results.extend(CandidateResult.from_dict(value) for value in read_json(source))
    return results


def save_failed_state(paths, task, attempt):
    attempts = (*task.attempts, attempt)
    write_json(paths.state_file(task.sample_id), dict(passed=False,
        original_score=task.original_score, record=task.record.to_dict(),
        attempts=[value.to_dict() for value in attempts]))


def backup_original(paths, task):
    label = task.record.sample.label
    image = Path(task.record.sample.extra["output_path"])
    metadata = Path(task.record.sample.extra["metadata_path"])
    backup_dir = paths.backups / label
    backup_dir.mkdir(parents=True, exist_ok=True)
    image_backup = backup_dir / image.name
    metadata_backup = backup_dir / metadata.name
    if not image_backup.exists():
        shutil.copy2(image, image_backup)
    if metadata.exists() and not metadata_backup.exists():
        shutil.copy2(metadata, metadata_backup)
    return image, metadata


def accept_candidate(paths, task, attempt):
    image, metadata = backup_original(paths, task)
    os.replace(attempt.candidate_path, image)
    history = [value.to_dict() for value in (*task.attempts, attempt)]
    if metadata.exists():
        payload = read_json(metadata)
        payload.update(task.record.to_dict())
        payload["clip_refinement"] = dict(original_score=task.original_score,
            accepted_score=attempt.score, attempts=history)
        payload["kontext_prompt"] = attempt.prompt
        payload["generation_seed"] = attempt.seed
        payload["input_image_paths"] = task.generation_job().image_paths
        payload["input_image_roles"] = ["target", "same_parent_context"]
        payload["output_path"] = str(image)
        payload["metadata_path"] = str(metadata)
        write_json(metadata, payload)
    write_json(paths.state_file(task.sample_id), dict(passed=True,
        original_score=task.original_score, accepted_score=attempt.score,
        record=task.record.to_dict(), attempts=history))


def publish_inputs(paths, states):
    records = read_json(paths.inputs)
    for record in records:
        state = states.get(record["sample_id"])
        if not state or not state.get("passed"):
            continue
        refined = state["record"]
        preserved = {
            key: record[key] for key in ("output_path", "metadata_path")
        }
        record.clear()
        record.update(refined, **preserved)
        record["kontext_prompt"] = state["attempts"][-1]["prompt"]
        record["input_image_paths"] = [
            refined["image_path"], refined["prompt_instance"]["image_path"]]
        record["input_image_roles"] = ["target", "same_parent_context"]
    write_json(paths.inputs, records)


def publish_scores(paths, threshold):
    rows = read_score_rows(paths.scores)
    states = {path.stem:read_json(path) for path in paths.state.glob("*.json")}
    for score in rows:
        state = states.get(score["sample_id"])
        if state and state.get("passed"):
            accepted = state["attempts"][-1]
            score["clip_score"] = str(accepted["score"])
            score["passes_threshold"] = "True"
            score["prompt"] = accepted["prompt"]
            record = state["record"]
            score["source_image"] = record["image_path"]
    temp = paths.root / "generation_results.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, paths.scores)
    publish_inputs(paths, states)
    passed = sum(float(score["clip_score"]) >= threshold for score in rows)
    report = dict(total=len(rows), passed=passed, remaining=len(rows)-passed, threshold=threshold)
    write_json(paths.root / "summary.json", report)
    return report
