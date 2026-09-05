"""Batch re-extension of failed Step2 descriptions with parent context."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from utils import load_prompts, validate_description
from pipeline.augmentation.prompts import generation_prompt
from pipeline.augmentation.records import ExtendedSample
from pipeline.augmentation.semantics import sample_semantic_label
from pipeline.augmentation.storage import RunPaths, read_json, write_json
from pipeline.augmentation.text_backend import DeepSeekBackend


def context_cache(task, paths):
    return (paths.root / "refined_prompts" /
            f"round_{task.round_number:03d}" / f"{task.sample_id}.json")


def cached_context(task, cache):
    if not cache.exists():
        return None
    payload = read_json(cache)
    father = payload.get("extension_inputs", {}).get("father_instance", {})
    if father.get("image_path") != task.record.sample.father.image_path:
        return None
    if payload.get("source_description") != task.record.sample.source_prompt:
        return None
    if not payload.get("context_scene"):
        return None
    return payload["description"], payload["context_scene"]


def request_description(task, backend):
    last_valid = None
    for _ in range(backend.config.max_attempts):
        results, extension_inputs = backend.extend(
            task.record.sample,
            target_description=task.record.sample.source_prompt)
        refined = results[0] if results else ""
        if not validate_description(
                refined, sample_semantic_label(task.record.sample)):
            continue
        last_valid = refined, extension_inputs
        if refined != task.current_description:
            return last_valid
    if last_valid:
        return last_valid
    raise RuntimeError(
        f"No valid refinement after {backend.config.max_attempts} attempts: "
        f"{task.sample_id}")


def refine_context(task, backend, paths):
    cache = context_cache(task, paths)
    if cached := cached_context(task, cache):
        return cached
    refined, extension_inputs = request_description(task, backend)
    scene = backend.scene(task.record.sample)
    write_json(cache, dict(
        sample_id=task.sample_id,
        source_description=task.record.sample.source_prompt,
        description=refined,
        description_changed=refined != task.current_description,
        context_scene=scene,
        method="step2_parent_conditioned_extension",
        extension_inputs=extension_inputs))
    return refined, scene


def refine_batch(tasks, config, paths):
    prompts = load_prompts(str(paths.run / "logs/prompt.json"))
    backend = DeepSeekBackend(
        config.step2_extension(), RunPaths(paths.run), prompts,
        logs_dir=paths.root)
    templates = prompts.get("generate", {})

    def refine_one(task):
        description, scene = refine_context(task, backend, paths)
        record = ExtendedSample(
            task.record.sample, description, scene)
        prompt = generation_prompt(
            record, config.pipeline.generation, templates)
        return replace(
            task, record=record, current_description=description,
            current_prompt=prompt)

    refined_tasks = []
    with ThreadPoolExecutor(max_workers=config.api_workers) as pool:
        for refined in pool.map(refine_one, tasks):
            refined_tasks.append(refined)
    return refined_tasks
