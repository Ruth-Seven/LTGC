"""Step3: plan typed jobs, resume image checkpoints, invoke backend, then evaluate."""
import time
from utils import parse_semantic_label
from .configuration import GenerationConfig
from .evaluation import evaluate
from .image_backend import DiffusersBackend
from .prompts import generation_prompt
from .records import GenerationJob
from .storage import GenerationPaths, read_json, signature, write_json


def generation_settings(config):
    settings = dict(model=config.model_path, steps=config.steps, guidance_scale=config.guidance,
                    width=config.size, height=config.size, dtype='bfloat16')
    if config.size < 1024:
        settings.update(reference_max_edge=config.size, reference_auto_resize=False)
    if config.two_references:
        settings['input_mode'] = 'target_plus_same_parent_image_and_description'
        if config.prompt_style == 'short':
            settings['context_prompt_style'] = 'short_exclude_image2_subject_scene_only'
    return settings


def plan_jobs(records, config: GenerationConfig, paths: GenerationPaths, prompts):
    if not records:
        raise ValueError('Empty instance manifest')
    if config.two_references and config.backend != 'flux-klein':
        raise ValueError('Two reference images require the flux-klein backend')
    jobs = []
    for record in records:
        source = record.sample
        if config.two_references:
            assert source.father.label != source.label
            assert parse_semantic_label(source.father.class_name)[1] == source.parent_category
        jobs.append(GenerationJob(record, generation_prompt(record, config, prompts.get('generate', {})),
                                  paths.image(source), paths.metadata(source), config.two_references))
    settings = generation_settings(config)
    digest = signature(dict(settings=settings, jobs=[job.to_dict() for job in jobs]))
    return jobs, dict(settings, signature=digest)


def generate(records, config, paths, prompts, dry_run=False, limit=None, backend_factory=None, scorer=None):
    from PIL import Image
    jobs, signed_settings = plan_jobs(records, config, paths, prompts)
    paths.logs.mkdir(parents=True, exist_ok=True)
    config_path = paths.logs / 'generation_config.json'
    if config_path.exists() and read_json(config_path)['signature'] != signed_settings['signature']:
        raise ValueError('Generation resume settings changed; select a new output directory')
    for job in jobs:
        for image_path in job.image_paths:
            with Image.open(image_path) as image:
                image.verify()
    write_json(paths.logs / 'generation_inputs.json', [job.to_dict() for job in jobs])
    if dry_run:
        print(f'[{config.backend}] dry-run validated {len(jobs)} image pairs')
        return jobs
    write_json(config_path, signed_settings)
    backend = None
    settings = generation_settings(config)
    try:
        for job in jobs[:limit]:
            if job.metadata_path.exists() and job.output_path.exists():
                continue
            if backend is None:
                backend = (backend_factory or DiffusersBackend)(config)
                write_json(paths.logs / 'runtime_config.json', backend.runtime)
            start = time.perf_counter()
            result = backend.render(job)
            if result.image.size != (config.size, config.size):
                raise ValueError(f'Unexpected output size: {result.image.size}')
            if config.size < 1024:
                reference_dir = paths.logs / 'reference_inputs'
                reference_dir.mkdir(exist_ok=True)
                for index, image in enumerate(result.references):
                    suffix = '' if index == 0 else '_context'
                    image.save(reference_dir / f'{job.record.sample.sample_id}{suffix}.png')
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(job.output_path)
            sizes = [image.size for image in result.references]
            elapsed = time.perf_counter() - start
            write_json(job.metadata_path, dict(job.to_dict(), settings=settings, runtime=backend.runtime,
                elapsed_seconds=elapsed, peak_gpu_allocated_gib=result.peak_memory_gib,
                reference_image_size=sizes[0], reference_image_sizes=sizes))
            print(f'[{config.backend}] generated {job.record.sample.sample_id} in {elapsed:.1f}s -> {job.output_path}', flush=True)
    finally:
        if backend is not None:
            backend.close()
    if limit is not None:
        return jobs[:limit]
    return evaluate(jobs, config, paths, scorer)
