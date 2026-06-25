"""
LTGC 流水线 - Step 2: 描述扩展
读取已有描述 → 本地 LLM 生成多样化变体 → 保存 CSV

--num_gpus N：多卡类级并行（每个 GPU worker 处理独立类子集）
"""
import os
from random import randint
import sys
import csv
import argparse
import logging
import json
import torch.multiprocessing as mp

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR
from data_txt.imagenet_label_mapping import get_readable_name as _imagenet_class_name
from utils import cleanup_stale_parts, load_prompts, validate_description

# 模块级变量，由 main() 从 prompt 文件初始化
extension_prompt = None
reflection_prompt = None
determine_prompt = None


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
    if class_map is not None:
        return str(class_map.get(str(label), label))
    return _imagenet_class_name(int(label)).split(", ")[0]


def _extend_worker(rank, world_size, class_chunks, max_generate_num,
                   output_base, log_dir, class_map, examples_dir,
                   ext_prompt, det_prompt, ref_prompt):
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
    part_path = f"{output_base}.part{rank}"
    open(part_path, 'w').close()
    total = len(class_list)
    data_to_write = []

    for idx, (label, texts) in enumerate(class_list):
        class_name = _get_class_name(label, class_map)
        n_existing = len(texts)
        logger.info(
            "[class %s %s] %d/%d start: existing=%d target=%d",
            label, class_name, idx + 1, total, n_existing, max_generate_num,
        )

        if n_existing == 0:
            logger.warning("[class %s %s] no existing descriptions, skip", label, class_name)
            continue

        per_text = 1
        all_new = []
        random_texts = []
        while len(random_texts) < max_generate_num:
            random_texts.append(texts[randint(0, n_existing - 1)])
        #对random_texts中的每条文本进行扩展、反思和验证
        for ti, text in enumerate(random_texts):
            raw = extend_descriptions(
                [text],
                prompt=ext_prompt.format(number=per_text, name=class_name),
                number=per_text,
                enable_thinking=False,
                max_token=100 * per_text,
                temperature=0.7,
            )
            # 质量过滤
            unique_raw = list(dict.fromkeys(raw))
            fresh = [d for d in unique_raw if validate_description(d, class_name)]
            # check duplication
            for f in fresh:
                if f in all_new:
                    logger.warnning("[class %s %s] desc %d/%d duplicate with previous, skip: %s",
                        label, class_name, ti + 1, n_existing, f)

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
                    prompt=ref_prompt.format(number=len(reflect_list), name=class_name),
                    number=len(reflect_list),
                    enable_thinking=True,
                    max_token=200 * len(reflect_list),
                    temperature=0.3,
                    do_sample=True,
                )
                new_fresh = fresh[:]
                for idx, sentence in zip(reflect_list, reflected):
                    target_idx = idx - 1
                    if validate_description(sentence, class_name):
                        new_fresh[target_idx] = sentence
                    else:
                        logger.info("[class %s %s] reflect %d rejected by validation: %s",
                            label, class_name, idx, sentence)
                reflected = new_fresh[:per_text]
            else:
                reflected = fresh[:per_text]

            logger.info("")
            logger.info(
                "[class %s %s] desc %d/%d summary: raw=%d unique=%d fresh=%d reflected=%d",
                label, class_name, ti + 1, n_existing,
                len(raw), len(unique_raw), len(fresh), len(reflected),
            )
            for desc_idx, desc in enumerate(raw, 1):
                logger.info(
                    "[class %s %s] desc %d/%d raw %d. %s",
                    label, class_name, ti + 1, n_existing, desc_idx, desc,
                )
            for desc_idx, desc in enumerate(fresh, 1):
                logger.info(
                    "[class %s %s] desc %d/%d fresh %d. %s",
                    label, class_name, ti + 1, n_existing, desc_idx, desc,
                )
            for desc_idx, desc in enumerate(reflected, 1):
                logger.info(
                    "[class %s %s] desc %d/%d reflected %d. %s",
                    label, class_name, ti + 1, n_existing, desc_idx, desc,
                )
            all_new.extend(reflected)

        all_new = list(dict.fromkeys(all_new))  # 全局去重
        logger.info("")
        logger.info(
            "[class %s %s] done: output=%d after class-level dedup",
            label, class_name, len(all_new),
        )
        for desc in all_new:
            data_to_write.append((label, desc))

        if examples_dir and all_new:
            os.makedirs(examples_dir, exist_ok=True)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            with open(md_path, 'w') as f:
                f.write(f"\n## Extended Descriptions ({len(all_new)})\n\n")
                for k, desc in enumerate(all_new, 1):
                    f.write(f"{k}. {desc}\n")
                f.write("\n")

        if len(data_to_write) >= 200:
            with open(part_path, 'a', newline='') as f:
                csv.writer(f).writerows(data_to_write)
            data_to_write = []

    if data_to_write:
        with open(part_path, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    logger.info("GPU %d done. %d classes processed.", rank, total)
    _unload_model()


def _merge_parts(output_path, num_gpus, logger):
    logger.info("Merging %d part files into %s", num_gpus, output_path)
    tmp_path = output_path + ".tmp"
    merged = 0
    with open(tmp_path, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        for gpu_id in range(num_gpus):
            part_path = f"{output_path}.part{gpu_id}"
            if not os.path.exists(part_path):
                logger.warning("Part file not found: %s", part_path)
                continue
            with open(part_path, 'r') as in_f:
                reader = csv.reader(in_f)
                for row in reader:
                    writer.writerow(row)
                    merged += 1
    os.rename(tmp_path, output_path)
    for gpu_id in range(num_gpus):
        part_path = f"{output_path}.part{gpu_id}"
        if os.path.exists(part_path):
            os.remove(part_path)
    logger.info("Merged %d rows into %s", merged, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 2: Description → Extended Descriptions')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Input descriptions CSV')
    parser.add_argument('-m', '--max_generate_num', default=50, type=int,
                        help='Max descriptions per class')
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
    return parser.parse_args()


def main():
    global extension_prompt, determine_prompt, reflection_prompt
    args = parse_args()
    prompts = load_prompts(args.prompt_file)

    ext_cfg = prompts.get("extend", {})
    extension_prompt = ext_cfg.get("extension_prompt") or extension_prompt
    determine_prompt = ext_cfg.get("determine_prompt") or determine_prompt
    reflection_prompt = ext_cfg.get("reflection_prompt") or reflection_prompt
    if ext_cfg.get("system_prompt"):
        from model.text_llm import set_system_prompt
        set_system_prompt(ext_cfg["system_prompt"])

    class_map = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping) as f:
            class_map = json.load(f)
        print(f"[extend] Loaded class mapping: {len(class_map)} entries")

    os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("extend", os.path.join(args.log_dir, "pipeline_extend.log"))

    cleanup_stale_parts(args.extended_description_path, logger)

    df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
    grouped = sorted(df.groupby('label')['text'].apply(list).items())
    num_classes = len(grouped)
    logger.info("Loaded %d classes from %s", num_classes, args.existing_description_path)

    num_gpus = args.num_gpus
    if num_gpus <= 1:
        _extend_worker(0, 1, [grouped], args.max_generate_num,
                       args.extended_description_path, args.log_dir, class_map,
                       args.examples_dir,
                       extension_prompt, determine_prompt, reflection_prompt)
        return

    logger.info("Multi-GPU mode: %d GPUs, %d classes", num_gpus, num_classes)
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(grouped):
        chunks[i % num_gpus].append(item)

    mp.spawn(_extend_worker, args=(num_gpus, chunks, args.max_generate_num,
             args.extended_description_path, args.log_dir, class_map,
             args.examples_dir,
             extension_prompt, determine_prompt, reflection_prompt),
        nprocs=num_gpus, join=True)

    _merge_parts(args.extended_description_path, num_gpus, logger)
    logger.info("Done. Output: %s", args.extended_description_path)


if __name__ == "__main__":
    main()
