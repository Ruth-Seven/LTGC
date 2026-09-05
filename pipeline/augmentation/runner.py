"""Python coordinator: YAML configuration in, stage workers out."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from utils import load_prompts
from .configuration import PipelineConfig
from .storage import RunPaths, read_json
from . import results


def prepare(config: PipelineConfig):
    from .sampling import sample
    paths = RunPaths(config.run_dir)
    paths.logs.mkdir(parents=True, exist_ok=True)
    if not paths.prompt.exists():
        shutil.copyfile(config.prompt_file, paths.prompt)
    if not paths.source_csv.exists():
        shutil.copyfile(config.source.descriptions, paths.source_csv)
    if not paths.samples.exists():
        sample(paths, config.source, config.sampling, config.execution.sample_limit)
    previous = read_json(paths.logs / 'sampling_config.json')
    if previous['seed'] != config.sampling.seed or previous['target_number'] != config.sampling.target_per_class:
        raise ValueError('Sampling settings changed; select a new run_dir')
    rows = read_json(paths.samples)
    mapping = read_json(config.source.class_mapping)
    for row in rows:
        father = row['prompt_instance']
        if row['class_name'] != mapping[row['label']] or father['class_name'] != mapping[father['label']]:
            raise ValueError('Existing manifest uses another class mapping; select a new run_dir')
    if config.execution.sample_limit is not None and len(rows) > config.execution.sample_limit:
        raise ValueError('Existing manifest exceeds sample_limit; select a new run_dir for a small test')
    return paths


def run_workers(config_file, stage, workers, paths):
    processes = []
    try:
        for worker in workers:
            name = f'extend_{worker}.log' if stage == 'extend' else f'generate_gpu_{worker}.log'
            with (paths.logs / name).open('a') as log:
                process = subprocess.Popen([sys.executable, '-m', 'pipeline.augmentation', stage,
                    '--config', str(config_file), '--worker', str(worker)], stdout=log, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[2], env=dict(os.environ, PYTHONUNBUFFERED="1"))
            processes.append((worker, process))
        failures = [worker for worker, process in processes if process.wait() != 0]
        if failures:
            raise RuntimeError(f'{stage} workers failed: {failures}; see {paths.logs}')
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        raise


def run(config: PipelineConfig):
    paths = prepare(config)
    snapshot = paths.logs / 'pipeline.resolved.yaml'
    config.save(snapshot)
    print(results.estimate(paths), flush=True)
    if config.execution.dry_run:
        return
    if config.execution.from_step <= 2 <= config.execution.through_step:
        from .extension import prepare_extension, merge
        from .storage import Step2Store
        prepare_extension(paths, config.extension, load_prompts(str(paths.prompt)), Step2Store(paths).samples())
        run_workers(snapshot, 'extend', range(config.extension.workers), paths)
        merge(paths, config.extension.model)
    if config.execution.from_step <= 3 <= config.execution.through_step:
        plan = results.partition(paths, config.generation.gpus)
        run_workers(snapshot, 'generate', [gpu for gpu in config.generation.gpus if plan[str(gpu)]], paths)
        results.merge_generation(paths, config.generation.clip_threshold)
        print(results.verify(paths), flush=True)


def extend_worker(config, worker):
    from .extension import extend
    paths = RunPaths(config.run_dir)
    return extend(paths, config.extension, load_prompts(str(paths.prompt)),
                  config.source.progress, shard=worker)


def generate_worker(config, gpu):
    if gpu not in config.generation.gpus:
        raise ValueError(f'GPU {gpu} is not in generation.gpus')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    from .generation import generate
    from .records import ExtendedSample
    paths = RunPaths(config.run_dir)
    records = [ExtendedSample.from_dict(row) for row in read_json(paths.shard_dir / f'gpu_{gpu}.json')]
    return generate(records, config.generation, paths.generation(gpu),
                    load_prompts(str(paths.prompt)), dry_run=config.execution.dry_run)
