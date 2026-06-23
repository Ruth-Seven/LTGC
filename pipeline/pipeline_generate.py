"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 -> SD 生成图像 -> CLIP 筛选 -> 保存
支持 --num_gpus N 自动多卡数据并行
"""
import os
import sys
import csv
import hashlib
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
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    # Route submodule loggers to the same file
    for sub_name in ("text_llm", "vision_lmm", "clip_score"):
        sub_logger = logging.getLogger(sub_name)
        sub_logger.handlers.clear()
        sub_logger.addHandler(fh)
        sub_logger.setLevel(logging.INFO)
        sub_logger.propagate = False

    return logger


def _hash_filename(label, description: str) -> str:
    h = hashlib.md5(description.encode()).hexdigest()[:12]
    return f"{label}_{h}.JPEG"


def _append_rows(path: str, header: list[str], rows: list[tuple]) -> None:
    """Append rows with a header on first write."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerows(rows)


def _load_success_score_cache(data_dir: str) -> dict[str, float]:
    """Load per-image CLIP scores from prior full or sharded success CSVs."""
    import glob

    cache = {}
    root = os.path.abspath(data_dir)
    paths = [os.path.join(root, "success_list.csv")]
    paths.extend(sorted(glob.glob(os.path.join(root, "parts", "success_gpu*.csv"))))
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.get("img_path")
                score = row.get("clip_score")
                if not img_path or score in (None, ""):
                    continue
                try:
                    cache[os.path.abspath(img_path)] = float(score)
                except ValueError:
                    continue
    return cache


def _load_recorded_paths(path: str) -> set[str]:
    """Load image paths already recorded in a sharded CSV."""
    if not os.path.exists(path):
        return set()
    paths = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("img_path")
            if img_path:
                paths.add(os.path.abspath(img_path))
    return paths


def _move_rejected_image(img_path: str, fail_img_dir: str, label: Any,
                         round_idx: int) -> str:
    """Move a rejected image out of train/ immediately and return its fail path."""
    target_dir = os.path.join(fail_img_dir, str(label))
    os.makedirs(target_dir, exist_ok=True)
    target_name = f"round{round_idx + 1}_{os.path.basename(img_path)}"
    target_path = os.path.join(target_dir, target_name)
    if os.path.abspath(img_path) == os.path.abspath(target_path):
        return target_path
    try:
        shutil.move(img_path, target_path)
    except FileNotFoundError:
        pass
    return target_path


def _unique_path(path: str) -> str:
    """Return a non-conflicting path by adding a numeric suffix if needed."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _reconcile_train_images(
    train_dir: str,
    fail_img_dir: str,
    success_rows: list[tuple],
) -> tuple[list[tuple], list[tuple[str, str]], list[tuple[str, str]]]:
    """Keep train/ aligned with success rows and move extra images to fail/unknown."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    existing_success = []
    missing_success = []
    success_paths = set()

    for row in success_rows:
        img_path = os.path.abspath(row[4])
        if os.path.exists(img_path):
            existing_success.append(row)
            success_paths.add(img_path)
        else:
            missing_success.append((str(row[0]), img_path))

    moved_unknown = []
    unknown_root = os.path.join(fail_img_dir, "unknown")
    if os.path.isdir(train_dir):
        for root, _, files in os.walk(train_dir):
            for name in files:
                src = os.path.abspath(os.path.join(root, name))
                if os.path.splitext(name)[1].lower() not in image_exts:
                    continue
                if src in success_paths:
                    continue
                rel_parent = os.path.relpath(root, train_dir)
                if rel_parent == ".":
                    rel_parent = "root"
                target_dir = os.path.join(unknown_root, rel_parent)
                os.makedirs(target_dir, exist_ok=True)
                target = _unique_path(os.path.join(target_dir, name))
                try:
                    shutil.move(src, target)
                    moved_unknown.append((src, target))
                except FileNotFoundError:
                    pass

    return existing_success, missing_success, moved_unknown


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
    result_queue: Optional[mp.Queue] = None,
) -> None:
    """单 GPU worker：SD 生成 → CLIP 筛选 → 低分 refine 重试"""
    import torch as _torch
    _torch.cuda.set_device(rank)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"pipeline_generate_gpu_{rank}.log")
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
    success_list = []
    fail_list = []
    class_list = class_chunks[rank]
    total = len(class_list)

    # ════════════════════════════════════════════════════════════════
    # 主循环：逐类处理（round 级批处理: SD 全量生成 → CLIP 全量评分 → LLM 全量反思）
    # ════════════════════════════════════════════════════════════════
    import time
    class_times = []  # (label, class_name, elapsed_sec, n, failed)
    failed_descs_path = os.path.join(args.log_dir, f"pipeline_generate_failed_descs_gpu{rank}.log")
    total_elapsed_start = time.time()
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    parts_dir = os.path.join(generated_imgs_dir, "parts")
    csv_header = ['id', 'class_name', 'clip_score', 'description', 'img_path']
    success_part_path = os.path.join(parts_dir, f"success_gpu{rank}.csv")
    score_cache = _load_success_score_cache(generated_imgs_dir)
    recorded_success_paths = _load_recorded_paths(success_part_path)
    logger.info(
        "Resume cache loaded: %d scored success images; shard success already has %d rows at %s",
        len(score_cache), len(recorded_success_paths), success_part_path,
    )

    for label_idx, (label, texts) in enumerate(class_list):
        class_start = time.time()
        class_name = _class_name(label)
        train_dir = os.path.join(args.data_dir, "train")
        dir_path = os.path.join(train_dir, str(label))
        os.makedirs(dir_path, exist_ok=True)

        generation_prompts = [
            str(t).strip() if not pd.isna(t) and str(t).strip() else f'A photo of a {class_name}' for t in texts
        ]
        clip_prompt = f'A photo of a {class_name}'
        n = len(generation_prompts)
        save_paths = [os.path.join(dir_path, _hash_filename(label, generation_prompts[i])) for i in range(n)]
        if args.onepath:
            save_paths = [os.path.join(args.data_dir, 'gen_train-onepath.JPEG')] * n
        current_img_paths = save_paths[:]

        accepted = [False] * n
        last_clip_score = [0.0] * n
        logger.info("[class %s %s] %d descs, thresh=%.2f, max_rounds=%d",
                    label, class_name, n, args.thresh, args.max_rounds)

        # ── 断点续传：检查已有图片 ──
        resumed = 0
        resumed_rows = []
        for i in range(n):
            if os.path.exists(save_paths[i]):
                accepted[i] = True
                cached_score = score_cache.get(os.path.abspath(save_paths[i]), -1.0)
                last_clip_score[i] = cached_score
                row = (label, class_name, cached_score, generation_prompts[i], save_paths[i])
                success_list.append(row)
                abs_path = os.path.abspath(save_paths[i])
                if abs_path not in recorded_success_paths:
                    resumed_rows.append(row)
                    recorded_success_paths.add(abs_path)
                if examples_dir:
                    gen_records[label].append((generation_prompts[i], cached_score, save_paths[i]))
                resumed += 1
        _append_rows(success_part_path, csv_header, resumed_rows)
        if resumed > 0:
            logger.info("[class %s %s] resumed %d/%d from existing images",
                        label, class_name, resumed, n)
            logger.info("[class %s %s] wrote %d resumed rows to shard success CSV",
                        label, class_name, len(resumed_rows))

        if all(accepted):
            logger.info("[class %s %s] all resumed, skip.", label, class_name)
            continue

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
            accepted_rows = []
            for idx, s in zip(valid_indices, clip_scores):
                last_clip_score[idx] = s
                score_row = (label, class_name, s, generation_prompts[idx], save_paths[idx])
                if s >= args.thresh:
                    score_cache[os.path.abspath(save_paths[idx])] = s
                    current_img_paths[idx] = save_paths[idx]
                    if examples_dir:
                        gen_records[label].append((generation_prompts[idx], s, save_paths[idx]))
                    accepted[idx] = True
                    n_acc += 1
                    success_list.append(score_row)
                    abs_path = os.path.abspath(save_paths[idx])
                    if abs_path not in recorded_success_paths:
                        accepted_rows.append(score_row)
                        recorded_success_paths.add(abs_path)
                else:
                    n_rej += 1
                    rejected_path = _move_rejected_image(
                        save_paths[idx], fail_img_dir, label, round_idx
                    )
                    current_img_paths[idx] = rejected_path
                    logger.warning("[class %s %s]  round %d/%d: desc %d/%d  score:%.4f rejected: %s",
                                   label, class_name, round_idx + 1, args.max_rounds,
                                   idx + 1, n, s ,generation_prompts[idx])
                    logger.info("[class %s %s] round %d/%d: moved rejected image to %s",
                                label, class_name, round_idx + 1, args.max_rounds,
                                rejected_path)
            _append_rows(success_part_path, csv_header, accepted_rows)
            if accepted_rows:
                logger.info("[class %s %s] round %d/%d: appended %d accepted rows to %s",
                            label, class_name, round_idx + 1, args.max_rounds,
                            len(accepted_rows), success_part_path)
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

        # ── 收集失败条目 ──
        for i in range(n):
            if not accepted[i]:
                fail_list.append((label, class_name, last_clip_score[i], generation_prompts[i], current_img_paths[i]))

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

    # ── 写入 per-worker CSV ──
    success_path = os.path.join(args.log_dir, f"success_gpu{rank}.csv")
    fail_path = os.path.join(args.log_dir, f"fail_gpu{rank}.csv")
    header = ['id', 'class_name', 'clip_score', 'description', 'img_path']

    with open(success_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(success_list)
    logger.info("Written success list: %d entries to %s", len(success_list), success_path)

    with open(fail_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(fail_list)
    logger.info("Written fail list: %d entries to %s", len(fail_list), fail_path)

    if result_queue is not None:
        result_queue.put((success_list, fail_list))
    return success_list, fail_list


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
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    if os.path.isdir(fail_img_dir):
        shutil.rmtree(fail_img_dir)
        print(f"[generate] Removed stale fail images: {fail_img_dir}")

    if args.num_gpus == 0:
        num_gpus = _detect_gpus()
    else:
        num_gpus = args.num_gpus

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = sorted(df.groupby('label')['text'].apply(list).items())

    if num_gpus <= 1:
        s, f = _worker(0, 1, [grouped], args, args.examples_dir)
        results = [(s, f)]
    else:
        print(f"[generate] Using {num_gpus} GPUs, {len(grouped)} classes total")
        chunks = [[] for _ in range(num_gpus)]
        for i, item in enumerate(grouped):
            chunks[i % num_gpus].append(item)

        result_queue = mp.Queue()
        mp.spawn(_worker, args=(num_gpus, chunks, args, args.examples_dir, result_queue),
                 nprocs=num_gpus, join=True)
        results = [result_queue.get() for _ in range(num_gpus)]

        print(f"[generate] All GPUs done.")

    # ── 合并所有 worker 的 success / fail 列表 ──
    all_success = []
    all_fail = []
    for s, f in results:
        all_success.extend(s)
        all_fail.extend(f)

    # ── 移动失败图片到 data_dir/fail/ ──
    # args.data_dir is the generated image root, e.g. .../generated_imgs.
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    os.makedirs(fail_img_dir, exist_ok=True)

    updated_fail = []
    for class_id, class_name, clip_score, desc, img_path in all_fail:
        target_dir = os.path.join(fail_img_dir, str(class_id))
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, os.path.basename(img_path))
        if os.path.abspath(img_path) != os.path.abspath(target_path):
            try:
                shutil.move(img_path, target_path)
            except FileNotFoundError:
                pass
        updated_fail.append((class_id, class_name, clip_score, desc, target_path))

    train_dir = os.path.join(generated_imgs_dir, "train")
    all_success, missing_success, moved_unknown = _reconcile_train_images(
        train_dir, fail_img_dir, all_success
    )
    if missing_success:
        print(f"[generate] WARNING: {len(missing_success)} success rows missing image files; omitted from success_list.csv")
        for class_id, img_path in missing_success[:20]:
            print(f"[generate] missing success image: class={class_id} path={img_path}")
        if len(missing_success) > 20:
            print(f"[generate] ... {len(missing_success) - 20} more missing success images")
    if moved_unknown:
        print(f"[generate] Moved {len(moved_unknown)} extra train images to {fail_img_dir}/unknown")
        for src, dst in moved_unknown[:20]:
            print(f"[generate] extra train image moved: {src} -> {dst}")
        if len(moved_unknown) > 20:
            print(f"[generate] ... {len(moved_unknown) - 20} more extra train images moved")

    # ── 写入合并 CSV ──
    header = ['id', 'class_name', 'clip_score', 'description', 'img_path']
    os.makedirs(generated_imgs_dir, exist_ok=True)

    success_merged_path = os.path.join(generated_imgs_dir, "success_list.csv")
    with open(success_merged_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_success)

    fail_merged_path = os.path.join(generated_imgs_dir, "fail_list.csv")
    with open(fail_merged_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_fail)

    print(f"[generate] Success: {len(all_success)}, Fail: {len(updated_fail)}")
    print(f"[generate] Merged lists -> {generated_imgs_dir}/")
    print(f"[generate] Fail images moved to {fail_img_dir}")

    # ── 生成 fail examples (单文件) ──
    if args.examples_dir and updated_fail:
        fail_examples_dir = os.path.join(args.examples_dir, "fail")
        os.makedirs(fail_examples_dir, exist_ok=True)

        fail_images_link = os.path.join(fail_examples_dir, "images")
        if not os.path.exists(fail_images_link):
            os.symlink(os.path.relpath(fail_img_dir, fail_examples_dir), fail_images_link)

        fail_md_path = os.path.join(fail_examples_dir, "failed_examples.md")
        with open(fail_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Failed Images ({len(updated_fail)})\n\n")
            for k, (cid, cname, score, desc, path) in enumerate(updated_fail, 1):
                rel = f"images/{os.path.relpath(path, fail_img_dir)}"
                f.write(f"## Fail {k}\n\n")
                f.write(f"**Class ID:** {cid}\n\n")
                f.write(f"**Class Name:** {cname}\n\n")
                f.write(f"**Old Description:** {desc}\n\n")
                f.write(f"![Failed Image {k}]({rel})\n\n")
                f.write(f"**CLIP Score:** {score:.4f}\n\n---\n\n")
        print(f"[generate] Fail examples -> {fail_md_path}")


if __name__ == "__main__":
    main()
