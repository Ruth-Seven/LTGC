"""
LTGC 流水线 - Step 2: 描述扩展
读取已有描述 → 本地 LLM 生成多样化变体 → 保存 CSV

--num_gpus N：多卡类级并行（每个 GPU worker 处理独立类子集）
"""
import argparse
import csv
import hashlib
import json
import logging
import os
from random import randint
import shutil
import sys

import pandas as pd
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR
from utils import (
    atomic_json_dump,
    cleanup_stale_parts,
    load_class_semantics,
    load_prompts,
    parse_semantic_label,
    validate_description,
)

# 模块级变量，由 main() 从 prompt 文件初始化
extension_prompt = None
reflection_prompt = None
determine_prompt = None

GENERATION_MODE = "single_reference_single_output"
ATTEMPT_MULTIPLIER = 3


def generation_round_limit(target_num, batch_generate_num):
    if batch_generate_num != 1:
        raise ValueError("single-reference Step2 requires batch_generate_num=1")
    return target_num * ATTEMPT_MULTIPLIER


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_signature(args, class_map, prompts, target):
    payload = {
        "input_sha256": _sha256_file(args.existing_description_path),
        "class_mapping": class_map,
        "prompts": prompts,
        "target": target,
        "batch_generate_num": args.batch_generate_num,
        "generation_mode": GENERATION_MODE,
        "attempt_multiplier": ATTEMPT_MULTIPLIER,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _class_progress_path(progress_dir, label):
    return os.path.join(progress_dir, "classes", f"{label}.csv")


def _write_class_progress(progress_dir, label, descriptions):
    path = _class_progress_path(progress_dir, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows((str(label), text) for text in descriptions)
    os.replace(tmp, path)


def _load_completed_classes(progress_dir, target, class_map):
    classes_dir = os.path.join(progress_dir, "classes")
    completed = {}
    if not os.path.isdir(classes_dir):
        return completed
    for filename in sorted(os.listdir(classes_dir)):
        if not filename.endswith(".csv"):
            continue
        label = filename[:-4]
        if label not in class_map:
            raise RuntimeError(f"Step2 progress has unknown class label: {label}")
        path = os.path.join(classes_dir, filename)
        with open(path, newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        descriptions = [row[1] for row in rows if len(row) == 2 and row[0] == label]
        class_name = class_map[label]
        valid = (
            len(rows) == target
            and len(descriptions) == target
            and len(set(descriptions)) == target
            and all(validate_description(text, class_name) for text in descriptions)
        )
        if not valid:
            raise RuntimeError(f"invalid Step2 class progress: {path}")
        completed[label] = descriptions
    return completed


def _write_class_example(examples_dir, label, class_name, descriptions):
    if not examples_dir:
        return
    os.makedirs(examples_dir, exist_ok=True)
    safe_name = class_name.replace(" ", "_").replace("/", "_")
    path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as output:
        output.write(f"\n## Extended Descriptions ({len(descriptions)})\n\n")
        for index, description in enumerate(descriptions, 1):
            output.write(f"{index}. {description}\n")
        output.write("\n")
    os.replace(tmp, path)


def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    # Route submodule loggers (text_llm, vision_lmm) to the same file
    for sub_name in ("text_llm", "vision_lmm"):
        sub_logger = logging.getLogger(sub_name)
        sub_logger.handlers.clear()
        sub_logger.addHandler(fh)
        sub_logger.setLevel(logging.INFO)
        sub_logger.propagate = False

    return logger


def _get_class_name(label, class_map):
    if class_map is None or str(label) not in class_map:
        raise ValueError(f"missing class semantic mapping for label {label}")
    return class_map[str(label)]


def _extend_worker(rank, class_chunks, max_generate_num, fixed_number,
                   progress_dir, log_dir, class_map, examples_dir,
                   ext_prompt, det_prompt, ref_prompt, batch_generate_num):
    import torch as _torch
    _torch.cuda.set_device(rank)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pipeline_extend_gpu{rank}.log")
    logger = setup_logger(f"extend_gpu{rank}", log_path)

    # 路由 text_llm 模块日志到同一个 GPU 文件
    import model.text_llm as _tllm
    _tllm._log.handlers.clear()
    _tllm._log.addHandler(logger.handlers[0])
    _tllm._log.setLevel(logging.INFO)
    _tllm._log.propagate = False

    from model.text_llm import extend_descriptions, determine_descriptions, reflection_descriptions, _unload_model

    class_list = class_chunks[rank]
    failed_path = os.path.join(progress_dir, f"failed_gpu{rank}.json")
    failures = {}
    total = len(class_list)

    for idx, (label, texts) in enumerate(class_list):
        class_name = _get_class_name(label, class_map)
        name, category, class_name = parse_semantic_label(class_name)
        n_existing = len(texts)
        target_num = fixed_number if fixed_number is not None else max_generate_num
        logger.info(
            "[class %s %s] %d/%d start: existing=%d target=%d fixed=%s",
            label, class_name, idx + 1, total, n_existing, target_num, fixed_number is not None,
        )

        if n_existing == 0:
            logger.warning("[class %s %s] no existing descriptions, skip", label, class_name)
            continue

        per_text = batch_generate_num
        max_attempts = generation_round_limit(target_num, per_text)
        all_new = []
        seen = set(texts)
        # Each attempt uses one random original description. Generated history is
        # deliberately excluded from model context to avoid cross-round copying.
        for ti in range(max_attempts):
            if len(all_new) >= target_num:
                break
            reference_text = texts[randint(0, n_existing - 1)]
            raw = extend_descriptions(
                [reference_text],
                prompt=ext_prompt.format(number=per_text, name=name, category=category),
                number=per_text,
                enable_thinking=False,
                max_token=100 * per_text,
                temperature=0.7,
            )
            # 质量过滤
            unique_raw = list(dict.fromkeys(raw))
            fresh = [d for d in unique_raw if d not in seen and validate_description(d, class_name)]

            reflect_list = []
            if fresh:
                reflect_list = determine_descriptions(
                    fresh,
                    prompt=det_prompt,
                    enable_thinking=False,
                    max_token=10 * len(fresh),
                    temperature=0.1,
                    do_sample=False,
                )
            if reflect_list:
                reflected = reflection_descriptions(
                    [fresh[idx - 1] for idx in reflect_list],
                    prompt=ref_prompt.format(number=len(reflect_list), name=name, category=category),
                    number=len(reflect_list),
                    enable_thinking=True,
                    max_token=200 * len(reflect_list),
                    temperature=0.3,
                    do_sample=True,
                )
                reflected_indices = set(reflect_list)
                new_fresh = [
                    sentence for sentence_idx, sentence in enumerate(fresh, 1)
                    if sentence_idx not in reflected_indices
                ]
                for reflect_idx, sentence in zip(reflect_list, reflected):
                    if validate_description(sentence, class_name) and sentence not in seen:
                        new_fresh.append(sentence)
                    else:
                        logger.info("[class %s %s] reflect %d rejected by validation: %s",
                            label, class_name, reflect_idx, sentence)
                reflected = new_fresh
            else:
                reflected = fresh

            reflected = [d for d in dict.fromkeys(reflected) if d not in seen]

            logger.info("")
            logger.info(
                "[class %s %s] desc %d/%d summary: raw=%d unique=%d fresh=%d reflected=%d",
                label, class_name, ti + 1, max_attempts,
                len(raw), len(unique_raw), len(fresh), len(reflected),
            )
            for desc_idx, desc in enumerate(raw, 1):
                logger.info(
                    "[class %s %s] desc %d/%d raw %d. %s",
                    label, class_name, ti + 1, max_attempts, desc_idx, desc,
                )
            for desc_idx, desc in enumerate(fresh, 1):
                logger.info(
                    "[class %s %s] desc %d/%d fresh %d. %s",
                    label, class_name, ti + 1, max_attempts, desc_idx, desc,
                )
            for desc_idx, desc in enumerate(reflected, 1):
                logger.info(
                    "[class %s %s] desc %d/%d reflected %d. %s",
                    label, class_name, ti + 1, max_attempts, desc_idx, desc,
                )
            for description in reflected:
                if len(all_new) >= target_num:
                    break
                seen.add(description)
                all_new.append(description)

        all_new = list(dict.fromkeys(all_new))[:target_num]
        if len(all_new) < target_num:
            logger.error(
                "[class %s %s] target not reached: output=%d target=%d rounds=%d",
                label, class_name, len(all_new), target_num, max_attempts,
            )
        logger.info("")
        logger.info(
            "[class %s %s] done: output=%d after class-level dedup",
            label, class_name, len(all_new),
        )
        if len(all_new) == target_num:
            _write_class_example(examples_dir, label, class_name, all_new)
            _write_class_progress(progress_dir, label, all_new)
        else:
            failures[str(label)] = {
                "class_name": class_name,
                "count": len(all_new),
                "target": target_num,
            }

    logger.info("GPU %d done. %d classes processed.", rank, total)
    atomic_json_dump(failures, failed_path)
    _unload_model()


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 2: Description → Extended Descriptions')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Input descriptions CSV')
    parser.add_argument('-m', '--max_generate_num', default=50, type=int,
                        help='Max generation attempts per class, or descriptions per class when --fix-number is not set')
    parser.add_argument('--fix-number', '--fix_number', dest='fixed_number', default=None, type=int,
                        help='Generate a fixed number of extended descriptions per class when possible')
    parser.add_argument('-ext', '--extended_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv'),
                        help='Output extended CSV')
    parser.add_argument('--log_dir', type=str, default="/tmp",
                        help='Log file directory')
    parser.add_argument('--class-mapping', type=str, default=None,
                        help='JSON class name mapping file (e.g. {"0":"crazing"})')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for class-level parallelism')
    parser.add_argument('--examples-dir', type=str, default=None,
                        help='Directory to save per-class Markdown (append Extended Descriptions section)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Prompt JSON 配置文件（默认使用内置 prompt）')
    parser.add_argument('--batch-generate-num', type=int, default=1,
                        help='Descriptions requested per attempt; single-reference mode requires 1')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                        help='Resume classes completed by a compatible prior run')
    parser.add_argument('--force', action='store_true',
                        help='Discard Step2 class progress before running')
    return parser.parse_args()


def main():
    global extension_prompt, determine_prompt, reflection_prompt
    args = parse_args()
    if args.fixed_number is not None and args.fixed_number <= 0:
        raise ValueError("--fix-number must be a positive integer")
    if args.batch_generate_num != 1:
        raise ValueError("--batch-generate-num must be 1 in single-reference mode")
    prompts = load_prompts(args.prompt_file)

    ext_cfg = prompts.get("extend", {})
    extension_prompt = ext_cfg.get("extension_prompt") or extension_prompt
    determine_prompt = ext_cfg.get("determine_prompt") or determine_prompt
    reflection_prompt = ext_cfg.get("reflection_prompt") or reflection_prompt
    if not extension_prompt or not determine_prompt or not reflection_prompt:
        raise ValueError("prompt file is missing required Step2 prompts")
    from model.text_llm import SYSTEM_PROMPT, set_system_prompt
    system_prompt = ext_cfg.get("system_prompt") or SYSTEM_PROMPT
    set_system_prompt(system_prompt)
    signature_prompts = {
        "extension_prompt": extension_prompt,
        "determine_prompt": determine_prompt,
        "reflection_prompt": reflection_prompt,
        "system_prompt": system_prompt,
    }

    class_map = load_class_semantics(args.class_mapping)
    print(f"[extend] Loaded class mapping: {len(class_map)} entries")

    os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("extend", os.path.join(args.log_dir, "pipeline_extend.log"))

    cleanup_stale_parts(args.extended_description_path, logger)

    df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
    grouped = sorted(df.groupby('label')['text'].apply(list).items())
    load_class_semantics(args.class_mapping, [label for label, _ in grouped])
    num_classes = len(grouped)
    logger.info("Loaded %d classes from %s", num_classes, args.existing_description_path)

    target = args.fixed_number if args.fixed_number is not None else args.max_generate_num
    if target <= 0:
        raise ValueError("Step2 target must be a positive integer")
    progress_dir = args.extended_description_path + ".progress"
    if args.force or not args.resume:
        shutil.rmtree(progress_dir, ignore_errors=True)
    os.makedirs(progress_dir, exist_ok=True)
    signature = _run_signature(args, class_map, signature_prompts, target)
    signature_path = os.path.join(progress_dir, "signature.json")
    if os.path.exists(signature_path):
        with open(signature_path, encoding="utf-8") as source:
            old_signature = json.load(source).get("signature")
        if old_signature != signature:
            raise RuntimeError("Step2 resume signature mismatch; use --force to restart")
    atomic_json_dump({"signature": signature}, signature_path)
    completed = _load_completed_classes(progress_dir, target, class_map)
    pending = [item for item in grouped if str(item[0]) not in completed]
    logger.info(
        "Step2 resume: completed=%d pending=%d target=%d",
        len(completed), len(pending), target,
    )

    num_gpus = args.num_gpus
    if not pending:
        logger.info("All %d classes restored from progress", num_classes)
    elif num_gpus <= 1:
        _extend_worker(0, [pending], args.max_generate_num, args.fixed_number,
                       progress_dir, args.log_dir, class_map,
                       args.examples_dir,
                       extension_prompt, determine_prompt, reflection_prompt,
                       args.batch_generate_num)
    else:
        logger.info("Multi-GPU mode: %d GPUs, %d pending classes", num_gpus, len(pending))
        chunks = [[] for _ in range(num_gpus)]
        for i, item in enumerate(pending):
            chunks[i % num_gpus].append(item)

        mp.spawn(_extend_worker, args=(chunks, args.max_generate_num, args.fixed_number,
                 progress_dir, args.log_dir, class_map,
                 args.examples_dir,
                 extension_prompt, determine_prompt, reflection_prompt,
                 args.batch_generate_num),
            nprocs=num_gpus, join=True)

    completed = _load_completed_classes(progress_dir, target, class_map)
    incomplete = {
        str(label): len(completed.get(str(label), []))
        for label, _ in grouped
        if str(label) not in completed
    }
    if incomplete:
        failed_path = args.extended_description_path + ".failed.json"
        atomic_json_dump({"target": target, "counts": incomplete}, failed_path)
        raise RuntimeError(f"Step2 incomplete for {len(incomplete)} classes; report: {failed_path}")

    tmp_path = args.extended_description_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        for label, _ in grouped:
            writer.writerows((str(label), text) for text in completed[str(label)])
    os.replace(tmp_path, args.extended_description_path)
    failed_path = args.extended_description_path + ".failed.json"
    if os.path.exists(failed_path):
        os.remove(failed_path)
    for filename in os.listdir(progress_dir):
        if filename.startswith("failed_gpu") and filename.endswith(".json"):
            os.remove(os.path.join(progress_dir, filename))
    logger.info("Done. Output: %s", args.extended_description_path)


if __name__ == "__main__":
    main()
