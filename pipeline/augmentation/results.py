"""Generation partition/merge/verification. No model imports or process management."""
import csv
import os
from collections import Counter
from pathlib import Path
from .storage import RunPaths, read_json as read, write_json as write


def expected_rows(paths):
    return [row for row in read(paths.samples)
        if not (paths.failures / (row['sample_id'] + '.json')).exists()]

def estimate(paths: RunPaths):
    rows = read(paths.samples)
    run_logs = paths.logs
    count = len(rows)
    report = dict(samples=count, classes=len({r['label'] for r in rows}),
        expected_api_calls=count * 2, approximate_input_tokens=count * 500,
        approximate_output_tokens=count * 80,
        estimated_no_cache_peak_usd=count * (500 * .44 + 80 * 1.32) / 1e6,
        estimated_no_cache_off_peak_usd=count * (500 * .22 + 80 * .66) / 1e6,
        model='deepseek-v4-flash', pricing_source='https://api-docs.deepseek.com/quick_start/pricing/',
        note='Planning estimate; actual tokens, caching and retries are recorded in deepseek_usage.jsonl.')
    write(run_logs / 'api_cost_estimate.json', report)
    return report


def partition(paths: RunPaths, gpus):
    run, run_logs = paths.root, paths.logs
    gen_logs, output = paths.generation_logs, paths.images
    rows = expected_rows(paths)
    records = read(run_logs / 'extended_manifest_context_short.json')
    ids = [r['sample_id'] for r in records]
    assert ids == [r['sample_id'] for r in rows] and len(set(ids)) == len(ids)
    shards = run_logs / 'generation_shards'
    shards.mkdir(exist_ok=True)
    plan = {str(gpu):[r['sample_id'] for r in records[index::len(gpus)]] for index, gpu in enumerate(gpus)}
    plan_path = shards / 'plan.json'
    if plan_path.exists():
        assert read(plan_path) == plan, 'Shard assignment changed; use a new RUN_DIR'
    for index, gpu in enumerate(gpus):
        subset = records[index::len(gpus)]
        if not subset:
            continue
        path = shards / f'gpu_{gpu}.json'
        if path.exists():
            assert read(path) == subset, 'Shard inputs changed; use a new RUN_DIR'
        else:
            write(path, subset)
    write(plan_path, plan)
    print('GPU partition:', {gpu:len(ids) for gpu,ids in plan.items()})
    return plan


def merge_generation(paths: RunPaths, threshold=.22):
    run, run_logs = paths.root, paths.logs
    gen_logs, output = paths.generation_logs, paths.images
    rows = expected_rows(paths)
    plan = read(run_logs / 'generation_shards/plan.json')
    order = {r['sample_id']:i for i,r in enumerate(rows)}
    inputs, runtime, headers = [], {}, {}
    tables = {'generation_results.csv': []}
    if any((gen_logs / f'gpu_{gpu}' / 'context_pairs.csv').exists() for gpu, ids in plan.items() if ids):
        tables['context_pairs.csv'] = []
    for gpu, ids in plan.items():
        if not ids:
            continue
        shard = gen_logs / f'gpu_{gpu}'
        records = read(shard / 'generation_inputs.json')
        assert [r['sample_id'] for r in records] == ids
        assert read(shard / 'summary.json')['generated'] == len(ids)
        inputs.extend(records)
        runtime[gpu] = read(shard / 'runtime_config.json')
        for filename in tables:
            with (shard / filename).open(newline='') as source:
                reader = csv.DictReader(source)
                headers[filename] = reader.fieldnames
                entries = list(reader)
            assert [r['sample_id'] for r in entries] == ids
            tables[filename].extend(entries)
        for record in records:
            source = Path(record['output_path'])
            assert source.is_file(), source
            link = output / 'train' / record['label'] / source.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == source.resolve(), link
            else:
                link.symlink_to(os.path.relpath(source, link.parent))
    inputs.sort(key=lambda r:order[r['sample_id']])
    assert [r['sample_id'] for r in inputs] == list(order)
    write(gen_logs / 'generation_inputs.json', inputs)
    write(gen_logs / 'runtime_config.json', runtime)
    for filename, entries in tables.items():
        entries.sort(key=lambda r:order[r['sample_id']])
        with (gen_logs / filename).open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=headers[filename])
            writer.writeheader()
            writer.writerows(entries)
    summary = dict(generated=len(inputs), counts=dict(Counter(r['label'] for r in inputs)),
        clip_threshold=threshold, clip_passed=sum(float(r['clip_score']) >= threshold for r in tables['generation_results.csv']),
        gpu_counts={gpu:len(ids) for gpu,ids in plan.items()})
    write(gen_logs / 'summary.json', summary)
    return summary


def verify(paths: RunPaths):
    run, run_logs = paths.root, paths.logs
    gen_logs, output = paths.generation_logs, paths.images
    rows = expected_rows(paths)
    from PIL import Image
    extended = read(run_logs / 'extended_manifest_context_short.json')
    inputs = read(gen_logs / 'generation_inputs.json')
    plan = read(run_logs / 'generation_shards/plan.json')
    gpu_by_sample = {sample:gpu for gpu,ids in plan.items() for sample in ids}
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in extended] == [r['sample_id'] for r in inputs]
    for source, record in zip(rows, inputs):
        if 'input_image_paths' in record:
            assert record['input_image_paths'] == [source['image_path'], source['prompt_instance']['image_path']]
        else:
            assert record['image_path'] == source['image_path']
        path = Path(record['output_path'])
        meta = read(Path(record['metadata_path']))
        assert meta['runtime']['cuda_visible_devices'] == gpu_by_sample[record['sample_id']]
        if not meta['runtime']['cpu_offload']:
            assert all(devices == ['cuda:0'] for devices in meta['runtime']['component_devices'].values())
        assert meta['generation_seed'] == source['generation_seed']
        with Image.open(path) as image:
            assert image.size == (meta['settings']['width'], meta['settings']['height'])
        assert (output / 'train' / record['label'] / path.name).resolve() == path.resolve()
    summary = read(gen_logs / 'summary.json')
    assert summary['generated'] == len(rows)
    assert summary['counts'] == dict(Counter(r['label'] for r in rows))
    return summary
