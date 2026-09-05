"""Translate historical command arguments at the boundary, never inside stages."""
import argparse
from pathlib import Path
from utils import load_prompts
from .configuration import ExtensionConfig, GenerationConfig, SamplingConfig, SourceConfig
from .storage import RunPaths, GenerationPaths, read_json
from .records import ExtendedSample


def sample(args):
    from .sampling import sample as execute
    paths = RunPaths(args.run_dir)
    return execute(paths, SourceConfig(str(paths.source_csv), args.step1_progress, args.class_mapping),
                   SamplingConfig(args.target_number, args.seed), args.limit)


def extend(args):
    from .extension import extend as execute
    config = ExtensionConfig(workers=args.num_shards, max_cost_usd=args.max_cost_usd)
    return execute(RunPaths(args.run_dir), config, load_prompts(args.prompt_file),
                   args.step1_progress, args.shard, args.limit)


def merge(args):
    from .extension import merge as execute
    return execute(RunPaths(args.run_dir))


def generate(args):
    from .generation import generate as execute
    klein = args.backend == 'flux-klein'
    config = GenerationConfig(backend=args.backend,
        model_path=args.klein_model if klein else args.kontext_model,
        size=args.klein_size if klein else args.kontext_size,
        steps=args.klein_steps if klein else args.kontext_steps, guidance=1.0 if klein else 2.5,
        two_references=args.same_parent_context_image, prompt_style=args.context_prompt_style,
        clip_threshold=args.thresh, kontext_blocks_per_group=args.kontext_blocks_per_group)
    output = Path(args.data_dir)
    logs = output / 'logs' if args.log_dir == '/tmp' else Path(args.log_dir)
    records = [ExtendedSample.from_dict(row) for row in read_json(args.instance_manifest)]
    return execute(records, config, GenerationPaths(output, logs), load_prompts(args.prompt_file),
                   dry_run=args.dry_run, limit=args.kontext_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['sample', 'extend', 'merge'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--step1-progress')
    parser.add_argument('--class-mapping')
    parser.add_argument('--prompt-file')
    parser.add_argument('--seed', type=int, default=20260903)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--target-number', type=int, default=100)
    parser.add_argument('--max-cost-usd', type=float, default=24.0)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    command = parser.parse_args()
    {'sample': sample, 'extend': extend, 'merge': merge}[command.stage](command)
