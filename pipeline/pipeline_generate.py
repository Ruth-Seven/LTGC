"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 -> SD 生成图像 -> CLIP 筛选 -> 保存
支持 --num_gpus N 自动多卡数据并行
"""
import os
import sys
import shutil
import argparse
import logging
import pandas as pd
from collections import defaultdict
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text -> Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default="/data/descriptions_data/extended_description.csv",
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default="/data", help='Output root')
    parser.add_argument('-t', '--thresh', default=0.28, type=float, help='CLIP score threshold')
    parser.add_argument('-r', '--max_rounds', default=5, type=int, help='Max retry rounds')
    parser.add_argument('-m', '--md', default=None, nargs='?', const="/tmp/gen_examples",
                        help='Markdown example records dir')
    parser.add_argument('-o', '--onepath', action='store_true', help='Save all images to same path')
    parser.add_argument('-b', '--batch', default=10, type=int, help='Batch size for generation')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs (0=auto, 1=single GPU)')
    parser.add_argument('--log_dir', type=str, default="/tmp",
                        help='Log file directory for worker processes')
    parser.add_argument('--class-mapping', type=str, default=None,
                        help='JSON class name mapping file')
    parser.add_argument('--examples-dir', type=str, default=None,
                        help='Directory to save per-class Markdown (append Generated Images section, alias of --md)')
    return parser.parse_args()


def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger


def save_generation_markdown(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "generation_examples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Generation Examples\n\n")
        f.write(f"Total examples: {len(records)}\n\n")
        for i, (class_name, description, img_path, clip_score) in enumerate(records):
            img_filename = f"gen_{i}_{class_name.replace(' ', '_')}.jpg"
            shutil.copy(img_path, os.path.join(output_dir, img_filename))
            f.write(f"## Example {i+1}: {class_name}\n\n")
            f.write(f"**Class:** {class_name}  \n")
            f.write(f"**Description:** {description}  \n")
            f.write(f"![Image]({img_filename})  \n")
            f.write(f"**CLIP Score:** {clip_score:.4f}  \n\n")
            f.write("---\n\n")
    print(f"[save_generation_markdown] Examples saved to {md_path}")


def _worker(rank, world_size, class_chunks, args, examples_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"generate_gpu_{rank}.log")
    logger = setup_logger(f"GPU{rank}", log_path)

    from config import GENERATION_EXAMPLE_DIR
    from model.clip_score import score, score_batch
    from model.image_gen import generate, generate_batch, unload_sd
    from model.text_llm import refine_description, _unload_model as unload_text_llm
    from data_txt.imagenet_label_mapping import get_readable_name as _imagenet_class_name

    _class_map = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        import json
        with open(args.class_mapping) as f:
            _class_map = json.load(f)

    def _class_name(label):
        if _class_map is not None:
            return str(_class_map.get(str(label), label))
        return _imagenet_class_name(int(label)).split(", ")[0]

    # resolve examples_dir: prefer --examples-dir, fallback to --md for backward compat
    examples_dir = examples_dir or args.md

    gen_records = defaultdict(list) if examples_dir else None
    class_list = class_chunks[rank]
    total = len(class_list)

    for label_idx, (label, texts) in enumerate(class_list):
        class_name = _class_name(label)
        dir_path = os.path.join(args.data_dir, str(label))
        os.makedirs(dir_path, exist_ok=True)
        texts = [f'A photo of a {class_name}' for t in texts]
        logger.info("Class %s (%d/%d), %d descriptions", label, label_idx + 1, total, len(texts))

        if args.onepath:
            for text_i, text in enumerate(texts):
                saved_path = os.path.join(args.data_dir, 'gen_train-onepath.JPEG')
                accepted = False
                for attempt in range(args.max_rounds):
                    img_path = generate(text, saved_path)
                    if img_path is None:
                        continue
                    clip_score = score(img_path, text)
                    logger.info("Class %s (%d/%d): Attempt %d/%d, Score: %.4f Class: %s",
                                label, label_idx + 1, total, attempt + 1, args.max_rounds, clip_score, class_name)
                    if clip_score >= args.thresh:
                        logger.info("accepted")
                        if examples_dir:
                            gen_records[label].append((text, clip_score, img_path))
                        accepted = True
                        break
                    else:
                        logger.info("Score %.4f < %s", clip_score, args.thresh)
                        if attempt < args.max_rounds - 1:
                            refined = refine_description(text, class_name)
                            if refined:
                                logger.info("Refined: %s", refined)
                                text = refined
                unload_text_llm()
                if not accepted:
                    logger.info("All attempts failed, skip")
            continue

        n = len(texts)
        save_paths = [os.path.join(dir_path, f"{label}_{i}.JPEG") for i in range(n)]
        accepted = [False] * n
        bs = args.batch

        for chunk_start in range(0, n, bs):
            chunk_end = min(chunk_start + bs, n)
            chunk_ids = list(range(chunk_start, chunk_end))
            logger.info(" Batch chunk [%d:%d]", chunk_start, chunk_end)

            for attempt in range(args.max_rounds):
                pending = [i for i in chunk_ids if not accepted[i]]
                if not pending:
                    break

                batch_prompts = [texts[i] for i in pending]
                batch_paths = [save_paths[i] for i in pending]
                img_paths = generate_batch(batch_prompts, batch_paths)
                unload_sd()

                valid = [(i, p) for i, p in zip(pending, img_paths) if p is not None]
                if not valid:
                    break

                v_idx, v_paths = zip(*valid)
                v_texts = [texts[i] for i in v_idx]
                clip_scores = score_batch(list(v_paths), list(v_texts))

                for idx, s in zip(v_idx, clip_scores):
                    logger.info("Class %s (%d/%d): Attempt %d/%d, Score: %.4f Class: %s",
                                label, label_idx + 1, total, attempt + 1, args.max_rounds, s, class_name)
                    if s >= args.thresh:
                        logger.info("accepted")
                        if examples_dir:
                            gen_records[label].append((texts[idx], s, save_paths[idx]))
                        accepted[idx] = True
                    else:
                        logger.info("Score %.4f < %s", s, args.thresh)

                if attempt < args.max_rounds - 1:
                    for i in chunk_ids:
                        if not accepted[i]:
                            refined = refine_description(texts[i], class_name)
                            if refined:
                                logger.info("Refined: %s", refined)
                                texts[i] = refined
                    unload_text_llm()

                if all(accepted[i] for i in chunk_ids):
                    break

        failed = sum(1 for a in accepted if not a)
        if failed:
            logger.info("%d/%d failed", failed, n)

        # 按 class 写入 Generated Images 段落
        if examples_dir and gen_records.get(label):
            os.makedirs(examples_dir, exist_ok=True)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            records = gen_records[label]
            with open(md_path, 'a') as f:
                f.write(f"\n## Generated Images ({len(records)})\n\n")
                for k, (desc, score, img_path) in enumerate(records, 1):
                    f.write(f"### Image {k}\n\n")
                    f.write(f"![Image {k}](file://{img_path})\n\n")
                    f.write(f"**Description:** {desc}\n\n")
                    f.write(f"**CLIP Score:** {score:.4f}\n\n")
            gen_records.pop(label)

    if examples_dir and gen_records:
        # write any remaining records (e.g. from onepath mode)
        for label, records in gen_records.items():
            class_name = _class_name(label)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            os.makedirs(examples_dir, exist_ok=True)
            with open(md_path, 'a') as f:
                f.write(f"\n## Generated Images ({len(records)})\n\n")
                for k, (desc, score, img_path) in enumerate(records, 1):
                    f.write(f"### Image {k}\n\n")
                    f.write(f"![Image {k}](file://{img_path})\n\n")
                    f.write(f"**Description:** {desc}\n\n")
                    f.write(f"**CLIP Score:** {score:.4f}\n\n")

    logger.info("Done. %d classes processed.", total)


def _detect_gpus():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return len([l for l in result.stdout.strip().split("\n") if l])
    except Exception:
        return 1


def main():
    args = parse_args()

    if args.num_gpus == 0:
        num_gpus = _detect_gpus()
    else:
        num_gpus = args.num_gpus

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = sorted(df.groupby('label')['text'].apply(list).items())

    if num_gpus <= 1:
        _worker(0, 1, [grouped], args, args.examples_dir)
        return

    print(f"[generate] Using {num_gpus} GPUs, {len(grouped)} classes total")
    chunks = [[] for _ in range(num_gpus)]
    for i, item in enumerate(grouped):
        chunks[i % num_gpus].append(item)

    mp.spawn(
        _worker,
        args=(num_gpus, chunks, args, args.examples_dir),
        nprocs=num_gpus,
        join=True,
    )

    print(f"[generate] All GPUs done.")


if __name__ == "__main__":
    main()
