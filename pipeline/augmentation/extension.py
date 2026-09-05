"""Step2 application logic: resume, deduplicate, extend, skip failures and export."""
import logging
from collections import Counter, defaultdict
from pathlib import Path
from utils import parse_semantic_label, validate_description
from .configuration import ExtensionConfig
from .prompts import mapped_description
from .records import ExtendedSample
from .semantics import sample_semantic_label
from .sampling import load_step1_sources
from .storage import RunPaths, Step2Store, read_json, signature, write_json
from .text_backend import DeepSeekBackend, DescriptionBackend

GENERATION_MODE = 'same_parent_single_reference_single_output_with_provenance'


def extension_settings(config):
    # Keep the historical signature payload, including integer/float representation.
    return dict(model=config.model, max_output_tokens=config.max_tokens,
        temperature=config.temperature, thinking=False, generation_mode=GENERATION_MODE,
        max_attempts_per_sample=config.max_attempts, max_cost_usd=float(config.max_cost_usd),
        pairing='reuse the sampled target and same-parent reference for each instance')


def prepare_extension(paths, config, prompts, samples):
    settings = extension_settings(config)
    digest = signature(dict(samples=[sample.to_dict() for sample in samples], prompts=prompts, settings=settings))
    target = paths.logs / 'step2_config.json'
    if target.exists():
        if read_json(target)['signature'] != digest:
            raise ValueError('Step2 inputs/settings changed')
    else:
        write_json(target, dict(settings, signature=digest, scene_max_tokens=config.scene_max_tokens,
                                scene_temperature=config.scene_temperature))
    # Historical runs implicitly used these scene options; retain their signature compatibility.
    scene = dict(scene_max_tokens=config.scene_max_tokens, scene_temperature=config.scene_temperature)
    previous = read_json(target)
    historical = dict(scene_max_tokens=40, scene_temperature=0.2)
    if scene != {key: previous.get(key, value) for key, value in historical.items()}:
        raise ValueError('Step2 scene settings changed')
    if not all(key in previous for key in scene):
        write_json(target, dict(previous, **scene))
    return settings


def cost_summary(paths, model, completed, planned):
    usage = paths.logs / 'deepseek_usage.jsonl'
    import json
    calls = [json.loads(line) for line in usage.read_text().splitlines()] if usage.exists() else []
    totals = {key:sum(call[key] for call in calls) for key in ('prompt_tokens', 'completion_tokens',
        'total_tokens', 'cache_hit_tokens', 'cache_miss_tokens', 'estimated_cost_usd', 'conservative_cost_usd')}
    report = dict(totals, provider='DeepSeek', model=model, api_calls=len(calls),
        successful_samples=completed, planned_samples=planned,
        periods=dict(Counter(call['period'] for call in calls)),
        pricing_source='https://api-docs.deepseek.com/quick_start/pricing/', pricing_checked='2026-09-03')
    write_json(paths.logs / 'api_cost_summary.json', report)
    return report


def extend(paths: RunPaths, config: ExtensionConfig, prompts, source_progress=None,
           shard=None, limit=None, backend: DescriptionBackend | None = None):
    store = Step2Store(paths)
    all_samples = store.samples()
    paths.logs.mkdir(parents=True, exist_ok=True)
    prepare_extension(paths, config, prompts, all_samples)
    samples = [sample for sample in all_samples if shard is None or int(sample.label) % config.workers == shard]
    sources = load_step1_sources(source_progress) if source_progress else None
    seen = defaultdict(set)
    for sample in samples:
        seen[sample.label].add(sample.source_prompt)
        father = sample.father
        assert father.label != sample.label and parse_semantic_label(father.class_name)[1] == sample.parent_category
        for label, caption, image_path in [(sample.label, sample.source_prompt, sample.image_path),
                                         (father.label, father.descriptions[0], father.image_path)]:
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
            if sources is not None:
                assert image_path in sources[(label, caption)], (label, image_path)
    if sources is not None:
        displays = {sample.label:sample.class_name for sample in samples}
        for label, caption in sources:
            seen[label].add(caption)
            if label in displays:
                seen[label].add(mapped_description(caption, displays[label]))
    paths.checkpoints.mkdir(exist_ok=True)
    paths.failures.mkdir(exist_ok=True)
    if shard is None or shard == 0:
        write_json(paths.logs / 'prompt_snapshot.json', prompts)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for sample in samples[:limit]:
        checkpoint = paths.checkpoints / f'{sample.sample_id}.json'
        failure = paths.failures / f'{sample.sample_id}.json'
        if checkpoint.exists():
            seen[sample.label].add(read_json(checkpoint)['extended_prompt'])
            continue
        if failure.exists():
            continue
        if backend is None:
            backend = DeepSeekBackend(config, paths, prompts)
        for _ in range(config.max_attempts):
            result, inputs = backend.extend(sample)
            if (result and result[0] not in seen[sample.label]
                    and validate_description(
                        result[0], sample_semantic_label(sample))):
                scene = backend.scene(sample)
                record = dict(sample.to_dict(), extended_prompt=result[0],
                    father_instances=[dict(sample.father.to_dict(), parent_category=sample.parent_category)],
                    extension_inputs=inputs, target_instance=dict(label=sample.label, class_name=sample.class_name,
                        parent_category=sample.parent_category, image_path=sample.image_path, descriptions=[sample.source_prompt]),
                    llm_backend='deepseek', llm_model=config.model, context_scene=scene,
                    context_scene_source='LLM scene extraction from Step2 father caption')
                write_json(checkpoint, record)
                seen[sample.label].add(result[0])
                print(f'[extend] {sample.sample_id}: {result[0]}', flush=True)
                break
        else:
            write_json(failure, dict(sample_id=sample.sample_id,
                reason=f'No valid extension after {config.max_attempts} attempts', action='skip'))
            print(f'[extend] skipped {sample.sample_id} after {config.max_attempts} attempts', flush=True)
    records = store.completed(samples)
    if shard is not None:
        write_json(paths.logs / f'extended_manifest_shard_{shard}.json', [record.to_dict() for record in records])
    else:
        store.export(records)
        cost_summary(paths, config.model, len(records), len(all_samples))
    return records


def merge(paths: RunPaths, model='deepseek-v4-flash'):
    store = Step2Store(paths)
    samples = store.samples()
    shards = sorted(paths.logs.glob('extended_manifest_shard_*.json'))
    if not shards:
        raise FileNotFoundError('No extended_manifest_shard_*.json')
    records = [ExtendedSample.from_dict(row) for shard in shards for row in read_json(shard)]
    order = {sample.sample_id:index for index, sample in enumerate(samples)}
    got = {record.sample.sample_id for record in records}
    assert len(got) == len(records), 'Duplicate sample IDs across Step2 shards'
    skipped = {sample.sample_id for sample in samples if (paths.failures / f'{sample.sample_id}.json').exists()}
    assert not got.intersection(skipped), 'Sample marked both completed and skipped'
    missing = set(order) - got - skipped
    if missing:
        raise RuntimeError(f'merge: {len(missing)} samples missing: {sorted(missing)[:5]}')
    records.sort(key=lambda record:order[record.sample.sample_id])
    write_json(paths.logs / 'step2_summary.json', dict(planned=len(samples), completed=len(records),
        skipped=len(skipped), skipped_sample_ids=sorted(skipped, key=order.get)))
    store.export(records)
    cost_summary(paths, model, len(records), len(samples))
    return records
