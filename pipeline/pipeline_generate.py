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
from typing import Optional, Any
import pandas as pd
from collections import defaultdict
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, EXTENDED_DESCRIPTION_PATH
from utils import validate_description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text -> Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default=EXTENDED_DESCRIPTION_PATH,
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, help='Output root')
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
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Prompt JSON 配置文件（预留给 generate 阶段扩展）')
    return parser.parse_args()


def setup_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger


def save_generation_markdown(records: list[tuple[str, str, str, float]], output_dir: str) -> None:
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


def _worker(
    rank: int,
    world_size: int,
    class_chunks: list[list[tuple[int, list[str]]]],
    args: argparse.Namespace,
    examples_dir: Optional[str],
) -> None:
    """单 GPU worker：SD 生成 → CLIP 筛选 → 低分 refine 重试"""
    import torch as _torch
    _torch.cuda.set_device(rank)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"generate_gpu_{rank}.log")
    logger = setup_logger(f"GPU{rank}", log_path)

    # ── 延迟导入：在 set_device 后加载模型，确保模型在指定 GPU 上 ──
    from config import GENERATION_EXAMPLE_DIR, SD_STYLE_SUFFIX
    from model.clip_score import score, score_batch
    from model.image_gen import generate, generate_batch, unload_sd
    from model.text_llm import reflect_one_description, _unload_model as unload_text_llm
    from data_txt.imagenet_label_mapping import get_readable_name as _imagenet_class_name

    # ── 类别名映射（支持自定义 JSON mapping）──
    _class_map = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        import json
        with open(args.class_mapping) as f:
            _class_map = json.load(f)

    def _class_name(label: int) -> str:
        if _class_map is not None:
            return str(_class_map.get(str(label), label))
        return _imagenet_class_name(int(label)).split(", ")[0]

    examples_dir = examples_dir or args.md

    gen_records = defaultdict(list) if examples_dir else None
    class_list = class_chunks[rank]
    total = len(class_list)

    # ════════════════════════════════════════════════════════════════
    # 主循环：逐类处理（round 级批处理: SD 全量生成 → CLIP 全量评分 → LLM 全量反思）
    # ════════════════════════════════════════════════════════════════
    import time
    class_times = []  # (label, class_name, elapsed_sec, n, failed)
    failed_descs_path = os.path.join(args.log_dir, f"failed_descs_gpu{rank}.log")
    total_elapsed_start = time.time()

    for label_idx, (label, texts) in enumerate(class_list):
        class_start = time.time()
        class_name = _class_name(label)
        dir_path = os.path.join(args.data_dir, str(label))
        os.makedirs(dir_path, exist_ok=True)

        generation_prompts = [
            str(t).strip() if not pd.isna(t) and str(t).strip() else f'A photo of a {class_name}' for t in texts
        ]
        clip_prompt = f'A photo of a {class_name}'
        n = len(generation_prompts)
        save_paths = [os.path.join(dir_path, f"{label}_{i}.JPEG") for i in range(n)]
        if args.onepath:
            save_paths = [os.path.join(args.data_dir, 'gen_train-onepath.JPEG')] * n

        accepted = [False] * n
        logger.info("[class %s %s] %d descs, thresh=%.2f, max_rounds=%d",
                    label, class_name, n, args.thresh, args.max_rounds)

        bs = args.batch
        for round_idx in range(args.max_rounds):
            round_start = time.time()
            pending = [i for i in range(n) if not accepted[i]]
            if not pending:
                logger.info("[class %s %s] round %d/%d: all accepted, done.",
                            label, class_name, round_idx + 1, args.max_rounds)
                break

            # ── Step A: SD 批量生成图像（SDXL 加载一次，分 chunk 生成）──
            logger.info("[class %s %s] round %d/%d: generating %d images...",
                        label, class_name, round_idx + 1, args.max_rounds, len(pending))
            img_paths_map = {}
            for chunk_start in range(0, len(pending), bs):
                chunk_end = min(chunk_start + bs, len(pending))
                chunk_idx_slice = pending[chunk_start:chunk_end]
                batch_prompts = [generation_prompts[i] + SD_STYLE_SUFFIX for i in chunk_idx_slice]
                batch_paths = [save_paths[i] for i in chunk_idx_slice]
                chunk_paths = generate_batch(batch_prompts, batch_paths)
                for i, p in zip(chunk_idx_slice, chunk_paths):
                    if p is not None:
                        img_paths_map[i] = p
            unload_sd()

            valid_indices = sorted(img_paths_map.keys())
            if not valid_indices:
                logger.warning("[class %s %s] round %d/%d: no valid images generated.",
                               label, class_name, round_idx + 1, args.max_rounds)
                break

            # ── Step B: CLIP 全量评分 ──
            valid_paths = [img_paths_map[i] for i in valid_indices]
            clip_scores = score_batch(valid_paths, [clip_prompt] * len(valid_paths))

            # ── Step C: 按阈值分流 accepted / rejected ──
            n_acc = 0
            n_rej = 0
            for idx, s in zip(valid_indices, clip_scores):
                if s >= args.thresh:
                    if examples_dir:
                        gen_records[label].append((generation_prompts[idx], s, save_paths[idx]))
                    accepted[idx] = True
                    n_acc += 1
                else:
                    n_rej += 1
                    logger.warning("[class %s %s]  round %d/%d: desc %d/%d rejected: %s",
                                   label, class_name, round_idx + 1, args.max_rounds,
                                   idx + 1, n, generation_prompts[idx])
            logger.info("[class %s %s] round %d/%d: accepted=%d rejected=%d",
                        label, class_name, round_idx + 1, args.max_rounds, n_acc, n_rej)

            # ── Step D: 批量反思低分描述（LLM 加载一次，逐个 reflect）──
            if round_idx < args.max_rounds - 1:
                rejected = [i for i in range(n) if not accepted[i]]
                if rejected:
                    n_refined = 0
                    for i in rejected:
                        refined = reflect_one_description(
                            generation_prompts[i], class_name,
                            prompt=args.reflect_one_prompt,
                            enable_thinking=False, do_sample=False,
                            temperature=0.2, max_token=100)
                        if not refined:
                            logger.warning("[class %s %s] round %d/%d: refined an empty description",
                                           label, class_name, round_idx + 1, args.max_rounds)
                            continue
                        if refined and validate_description(refined, class_name):
                            generation_prompts[i] = refined
                            n_refined += 1
                            logger.info("[class %s %s] round %d/%d: refined: %s",
                                        label, class_name, round_idx + 1, args.max_rounds, refined)
                        else:
                            logger.warning("[class %s %s] round %d/%d: fail to validate refined dst: %s",
                                           label, class_name, round_idx + 1, args.max_rounds, refined)
                    unload_text_llm()
                    logger.info("[class %s %s] round %d/%d: refined %d/%d descriptions.",
                                label, class_name, round_idx + 1, args.max_rounds,
                                n_refined, len(rejected))

            if all(accepted):
                logger.info("[class %s %s] round %d/%d: all accepted.",
                            label, class_name, round_idx + 1, args.max_rounds)
                break

        failed = sum(1 for a in accepted if not a)
        elapsed = time.time() - class_start
        class_times.append((label, class_name, elapsed, n, failed))
        logger.info("[class %s %s] done: %d/%d accepted, %d failed | elapsed %.1fs",
                    label, class_name, n - failed, n, failed, elapsed)

        # ── 记录最终未通过（满 max_rounds 仍 rejected）的描述 ──
        if failed > 0:
            with open(failed_descs_path, 'a') as fd:
                fd.write(f"\n[class {label} {class_name}] {failed}/{n} failed after {args.max_rounds} rounds:\n")
                for i in range(n):
                    if not accepted[i]:
                        fd.write(f"  [{i}] {generation_prompts[i]}\n")

        # ── ETA 预估 ──
        if class_times:
            avg_time = sum(t[2] for t in class_times) / len(class_times)
            remaining = total - label_idx - 1
            eta_sec = avg_time * remaining
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            logger.info("[GPU%d progress] %d/%d classes | avg %.1fs/class | ETA %dh%dm",
                        rank, label_idx + 1, total, avg_time, eta_h, eta_m)

        # ── 写入 Markdown ──
        if examples_dir and gen_records.get(label):
            os.makedirs(examples_dir, exist_ok=True)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            records = gen_records[label]
            with open(md_path, 'w') as f:
                f.write(f"\n## Generated Images ({len(records)})\n\n")
                for k, (desc, score, img_path) in enumerate(records, 1):
                    img_rel = f"images/{os.path.relpath(img_path, args.data_dir)}"
                    f.write(f"### Image {k}\n\n")
                    f.write(f"![Image {k}]({img_rel})\n\n")
                    f.write(f"**Description:** {desc}\n\n")
                    f.write(f"**CLIP Score:** {score:.4f}\n\n")
            gen_records.pop(label)

    if examples_dir and gen_records:
        # ── 刷出剩余记录（如 onepath 模式下残留的）──
        for label, records in gen_records.items():
            class_name = _class_name(label)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            os.makedirs(examples_dir, exist_ok=True)
            with open(md_path, 'w') as f:
                f.write(f"\n## Generated Images ({len(records)})\n\n")
                for k, (desc, score, img_path) in enumerate(records, 1):
                    img_rel = f"images/{os.path.relpath(img_path, args.data_dir)}"
                    f.write(f"### Image {k}\n\n")
                    f.write(f"![Image {k}]({img_rel})\n\n")
                    f.write(f"**Description:** {desc}\n\n")
                    f.write(f"**CLIP Score:** {score:.4f}\n\n")

    logger.info("Done. %d classes processed.", total)


def _detect_gpus() -> int:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return len([l for l in result.stdout.strip().split("\n") if l])
    except Exception:
        return 1


def main() -> None:
    from utils import load_prompts
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    args.reflect_one_prompt = prompts.get("generate", {}).get("reflect_one_prompt")

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

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        pool.starmap(_worker, [
            (rank, num_gpus, chunks, args, args.examples_dir)
            for rank in range(num_gpus)
        ])

    print(f"[generate] All GPUs done.")


if __name__ == "__main__":
    main()
