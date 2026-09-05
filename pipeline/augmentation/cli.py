"""CLI chooses an operation; all pipeline settings come from YAML."""
import argparse
from .configuration import PipelineConfig
from .storage import RunPaths


def main(default_stage='run'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default=default_stage,
        choices=['run', 'sample', 'extend', 'generate', 'merge-step2', 'partition', 'merge-generation', 'verify', 'estimate'])
    parser.add_argument('--config', required=True, help='Pipeline YAML file')
    parser.add_argument('--worker', type=int, help='Internal worker identity; model settings remain in YAML')
    command = parser.parse_args()
    config = PipelineConfig.load(command.config)
    paths = RunPaths(config.run_dir)
    from . import runner, results
    if command.stage == 'run':
        runner.run(config)
    elif command.stage == 'sample':
        runner.prepare(config)
    elif command.stage == 'extend':
        if command.worker is None:
            from dataclasses import replace
            runner.run(replace(config, execution=replace(config.execution, from_step=2, through_step=2)))
        else:
            runner.extend_worker(config, command.worker)
    elif command.stage == 'generate':
        if command.worker is None:
            from dataclasses import replace
            runner.run(replace(config, execution=replace(config.execution, from_step=3, through_step=3)))
        else:
            runner.generate_worker(config, command.worker)
    elif command.stage == 'merge-step2':
        from .extension import merge
        merge(paths, config.extension.model)
    elif command.stage == 'partition':
        results.partition(paths, config.generation.gpus)
    elif command.stage == 'merge-generation':
        print(results.merge_generation(paths, config.generation.clip_threshold))
    elif command.stage == 'verify':
        print(results.verify(paths))
    else:
        print(results.estimate(paths))
