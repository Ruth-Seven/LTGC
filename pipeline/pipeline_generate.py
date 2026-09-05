"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 -> SD 生成图像 -> CLIP 筛选 -> 保存
支持 --num_gpus N 自动多卡数据并行
"""
import argparse
import csv
import faulthandler
import hashlib
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from typing import Any

import pandas as pd
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, EXTENDED_DESCRIPTION_PATH
from utils import (
    atomic_json_dump,
    load_class_semantics,
    validate_description,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text -> Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default=EXTENDED_DESCRIPTION_PATH,
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, help='Output root')
    parser.add_argument('-t', '--thresh', default=0.28, type=float, help='CLIP score threshold')
    parser.add_argument('-r', '--max_rounds', default=5, type=int, help='Max retry rounds')
    parser.add_argument('--max-target-number', '--max_target_number',
                        dest='max_target_number', default=100, type=int,
                        help='Maximum accepted images per class; actual target is capped by the class description count')
    parser.add_argument('--max-sample-attempts', default=1000, type=int,
                        help='Maximum sampled description candidates per class')
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
    try:
        faulthandler.enable(file=fh.stream, all_threads=True)
    except Exception:
        pass
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


def _hash_filename(label, description: str, attempt_id: int = 0) -> str:
    h = hashlib.md5(f"{attempt_id}\0{description}".encode()).hexdigest()[:12]
    return f"{label}_{attempt_id:04d}_{h}.JPEG"


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


def _write_rows_atomic(path: str, header: list[str], rows: list[tuple]) -> None:
    """Atomically replace a CSV, including its header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _deduplicate_success_rows(rows: list[tuple]) -> list[tuple]:
    """Keep one success row per concrete image path."""
    unique = []
    seen_paths = set()
    for row in rows:
        path = os.path.abspath(row[4])
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique.append(row)
    return unique


def _validate_success_counts(
    rows: list[tuple], expected_labels: list[str], target_counts: dict[str, int],
) -> tuple[dict[str, int], list[str]]:
    counts = defaultdict(int)
    for row in rows:
        counts[str(row[0])] += 1
    expected = set(expected_labels)
    incomplete = {
        label: counts.get(label, 0)
        for label in expected_labels
        if counts.get(label, 0) != target_counts[label]
    }
    unexpected = sorted(set(counts) - expected)
    return incomplete, unexpected


def _load_worker_csv(path: str) -> list[tuple]:
    """Load worker result rows from disk instead of passing large lists through mp.Queue."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                label: Any = int(row.get("id", ""))
            except ValueError:
                label = row.get("id", "")
            try:
                clip_score = float(row.get("clip_score", 0.0))
            except ValueError:
                clip_score = 0.0
            rows.append((
                label,
                row.get("class_name", ""),
                clip_score,
                row.get("description", ""),
                row.get("img_path", ""),
            ))
    return rows


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
            f.write(f'<img src="{img_filename}" alt="Image" loading="lazy">  \n')
            f.write(f"**CLIP Score:** {clip_score:.4f}  \n\n")
            f.write("---\n\n")
    print(f"[save_generation_markdown] Examples saved to {md_path}")


def _ensure_relative_symlink(link_path: str, target_dir: str) -> None:
    """Create a relative symlink for Markdown assets if it is absent."""
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    rel_target = os.path.relpath(os.path.abspath(target_dir), os.path.dirname(link_path))
    if os.path.islink(link_path):
        if os.readlink(link_path) != rel_target:
            os.unlink(link_path)
            os.symlink(rel_target, link_path)
        return
    if not os.path.exists(link_path):
        os.symlink(rel_target, link_path)


def _write_class_generation_markdown(
    examples_dir: str,
    data_dir: str,
    label: Any,
    class_name: str,
    records: list[tuple[str, float, str]],
) -> None:
    """Write per-class generated image examples with links rooted at examples_dir/images."""
    os.makedirs(examples_dir, exist_ok=True)
    _ensure_relative_symlink(os.path.join(examples_dir, "images"), data_dir)
    safe_name = class_name.replace(' ', '_').replace('/', '_')
    md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"\n## Generated Images ({len(records)})\n\n")
        for k, (desc, score, img_path) in enumerate(records, 1):
            img_rel = f"images/{os.path.relpath(img_path, data_dir)}"
            f.write(f"### Image {k}\n\n")
            f.write(f'<img src="{img_rel}" alt="Image {k}" loading="lazy">\n\n')
            f.write(f"**Description:** {desc}\n\n")
            f.write(f"**CLIP Score:** {score:.4f}\n\n")


def _worker(
    rank: int,
    world_size: int,
    class_chunks: list[list[tuple[int, list[str]]]],
    args: argparse.Namespace,
    examples_dir: str | None,
    result_queue: Any = None,
) -> None:
    """单 GPU worker：SD 生成 → CLIP 筛选 → 低分 refine 重试"""
    import torch as _torch
    _torch.cuda.set_device(rank)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"pipeline_generate_gpu_{rank}.log")
    logger = setup_logger(f"GPU{rank}", log_path)

    # ── 延迟导入：在 set_device 后加载模型，确保模型在指定 GPU 上 ──
    from data_txt.imagenet_label_mapping import (
        get_readable_name as _imagenet_class_name,
    )
    from model.clip_score import score_batch
    from model.image_gen import generate_batch, unload_sd
    from model.text_llm import _unload_model as unload_text_llm
    from model.text_llm import reflect_one_description

    # ── 类别名映射（支持自定义 JSON mapping）──
    _class_map = load_class_semantics(args.class_mapping)

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
    open(failed_descs_path, "w").close()
    time.time()
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    parts_dir = os.path.join(generated_imgs_dir, "parts")
    csv_header = ['id', 'class_name', 'clip_score', 'description', 'img_path']
    success_part_path = os.path.join(parts_dir, f"success_gpu{rank}.csv")
    prior_success_by_label = defaultdict(list)
    prior_success_paths = set()
    for row in _load_worker_csv(success_part_path):
        abs_path = os.path.abspath(row[4])
        if (
            row[2] >= args.thresh
            and os.path.exists(abs_path)
            and abs_path not in prior_success_paths
        ):
            prior_success_by_label[str(row[0])].append(row)
            prior_success_paths.add(abs_path)
    recorded_success_paths = set(prior_success_paths)
    incomplete_classes = {}
    logger.info(
        "Resume cache loaded: %d valid accepted images from %s",
        len(recorded_success_paths), success_part_path,
    )

    for label_idx, (label, texts) in enumerate(class_list):
        class_start = time.perf_counter()
        class_name = _class_name(label)
        train_dir = os.path.join(args.data_dir, "train")
        dir_path = os.path.join(train_dir, str(label))
        os.makedirs(dir_path, exist_ok=True)

        source_descriptions = [
            str(t).strip() if not pd.isna(t) and str(t).strip() else f'A photo of a {class_name}' for t in texts
        ]
        source_descriptions = [t for t in source_descriptions if t]
        if not source_descriptions:
            logger.error("[class %s %s] empty description pool", label, class_name)
            incomplete_classes[str(label)] = 0
            continue

        clip_prompt = f'A photo of a {class_name}'
        target_num = args.target_counts[str(label)]
        class_success_rows = prior_success_by_label.get(str(label), [])[:target_num]
        success_list.extend(class_success_rows)
        for row in class_success_rows:
            if examples_dir:
                gen_records[label].append((row[3], row[2], row[4]))

        class_success = len(class_success_rows)
        sampled_candidates = 0
        candidate_serial = class_success
        draw_queue = []
        logger.info(
            "[class %s %s] pool=%d resumed=%d target=%d thresh=%.2f max_rounds=%d max_samples=%d",
            label, class_name, len(source_descriptions), class_success, target_num,
            args.thresh, args.max_rounds, args.max_sample_attempts,
        )

        def draw_description() -> str:
            if not draw_queue:
                draw_queue.extend(source_descriptions)
                random.shuffle(draw_queue)
            return draw_queue.pop()

        while class_success < target_num and sampled_candidates < args.max_sample_attempts:
            candidate_n = min(
                target_num - class_success,
                args.max_sample_attempts - sampled_candidates,
            )
            base_prompts = [draw_description() for _ in range(candidate_n)]
            generation_prompts = [text + args.generate_pic_style for text in base_prompts]
            save_paths = []
            for text in base_prompts:
                candidate_serial += 1
                path = os.path.join(
                    dir_path, _hash_filename(label, text, candidate_serial)
                )
                save_paths.append(_unique_path(path))
            current_img_paths = save_paths[:]
            accepted = [False] * candidate_n
            last_clip_score = [0.0] * candidate_n
            sampled_candidates += candidate_n

            logger.info(
                "[class %s %s] refill: success=%d/%d sampled=%d/%d candidates=%d",
                label, class_name, class_success, target_num,
                sampled_candidates, args.max_sample_attempts, candidate_n,
            )

            bs = args.batch
            for round_idx in range(args.max_rounds):
                round_start = time.perf_counter()
                pending = [i for i in range(candidate_n) if not accepted[i]]
                if not pending:
                    break

                logger.info("[class %s %s] round %d/%d: generating %d images...",
                            label, class_name, round_idx + 1, args.max_rounds, len(pending))
                img_paths_map = {}
                sd_start = time.perf_counter()
                for chunk_start in range(0, len(pending), bs):
                    chunk_idx_slice = pending[chunk_start:chunk_start + bs]
                    batch_prompts = [generation_prompts[i] for i in chunk_idx_slice]
                    batch_paths = [save_paths[i] for i in chunk_idx_slice]
                    chunk_paths = generate_batch(batch_prompts, batch_paths)
                    for i, path in zip(chunk_idx_slice, chunk_paths):
                        if path is not None:
                            img_paths_map[i] = path
                unload_sd()
                sd_elapsed = time.perf_counter() - sd_start

                valid_indices = sorted(img_paths_map)
                if not valid_indices:
                    logger.warning("[class %s %s] round %d/%d: no valid images generated.",
                                   label, class_name, round_idx + 1, args.max_rounds)
                    break

                valid_paths = [img_paths_map[i] for i in valid_indices]
                clip_start = time.perf_counter()
                clip_scores = score_batch(valid_paths, [clip_prompt] * len(valid_paths))
                clip_elapsed = time.perf_counter() - clip_start

                split_start = time.perf_counter()
                n_acc = 0
                n_rej = 0
                accepted_rows = []
                for idx, clip_value in zip(valid_indices, clip_scores):
                    last_clip_score[idx] = clip_value
                    score_row = (
                        label, class_name, clip_value,
                        generation_prompts[idx], save_paths[idx],
                    )
                    if clip_value >= args.thresh:
                        current_img_paths[idx] = save_paths[idx]
                        accepted[idx] = True
                        n_acc += 1
                        class_success += 1
                        class_success_rows.append(score_row)
                        success_list.append(score_row)
                        if examples_dir:
                            gen_records[label].append((
                                generation_prompts[idx], clip_value, save_paths[idx]
                            ))
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
                split_elapsed = time.perf_counter() - split_start

                shard_write_start = time.perf_counter()
                _append_rows(success_part_path, csv_header, accepted_rows)
                shard_write_elapsed = time.perf_counter() - shard_write_start
                logger.info(
                    "[class %s %s] round %d/%d: accepted=%d rejected=%d total=%d/%d",
                    label, class_name, round_idx + 1, args.max_rounds,
                    n_acc, n_rej, class_success, target_num,
                )

                refine_elapsed = 0.0
                if round_idx < args.max_rounds - 1:
                    rejected = [i for i in range(candidate_n) if not accepted[i]]
                    if rejected:
                        refine_start = time.perf_counter()
                        n_refined = 0
                        for i in rejected:
                            refined = reflect_one_description(
                                generation_prompts[i], class_name,
                                prompt=args.reflect_one_prompt,
                                enable_thinking=False, do_sample=False,
                                temperature=0.2, max_token=100,
                            )
                            if refined and validate_description(refined, class_name):
                                generation_prompts[i] = refined
                                n_refined += 1
                        unload_text_llm()
                        logger.info(
                            "[class %s %s] round %d/%d: refined %d/%d descriptions.",
                            label, class_name, round_idx + 1, args.max_rounds,
                            n_refined, len(rejected),
                        )
                        refine_elapsed = time.perf_counter() - refine_start

                round_elapsed = time.perf_counter() - round_start
                logger.info(
                    "[class %s %s] round %d/%d timing total=%.3fs sd_generate=%.3fs clip_score=%.3fs split_move=%.3fs shard_success_write=%.3fs refine=%.3fs valid_images=%d",
                    label, class_name, round_idx + 1, args.max_rounds,
                    round_elapsed, sd_elapsed, clip_elapsed, split_elapsed,
                    shard_write_elapsed, refine_elapsed, len(valid_indices),
                )

            batch_failed = sum(1 for value in accepted if not value)
            for i in range(candidate_n):
                if not accepted[i]:
                    fail_list.append((
                        label, class_name, last_clip_score[i],
                        generation_prompts[i], current_img_paths[i],
                    ))
            if batch_failed:
                with open(failed_descs_path, 'a') as fd:
                    fd.write(
                        f"\n[class {label} {class_name}] {batch_failed}/{candidate_n} "
                        f"candidate failures after {args.max_rounds} rounds; "
                        f"success={class_success}/{target_num}:\n"
                    )
                    for i in range(candidate_n):
                        if not accepted[i]:
                            fd.write(f"  [{i}] {generation_prompts[i]}\n")

        failed = target_num - class_success
        elapsed = time.perf_counter() - class_start
        class_times.append((label, class_name, elapsed, target_num, failed))
        logger.info(
            "[class %s %s] done: %d/%d accepted, sampled=%d, missing=%d | elapsed %.1fs",
            label, class_name, class_success, target_num,
            sampled_candidates, failed, elapsed,
        )
        if failed > 0:
            incomplete_classes[str(label)] = class_success

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
            _write_class_generation_markdown(
                examples_dir, args.data_dir, label, class_name, gen_records[label]
            )
            gen_records.pop(label)

    if examples_dir and gen_records:
        # ── 刷出剩余记录（如 onepath 模式下残留的）──
        for label, records in gen_records.items():
            class_name = _class_name(label)
            _write_class_generation_markdown(
                examples_dir, args.data_dir, label, class_name, records
            )

    logger.info("Done. %d classes processed.", total)

    # ── 写入 per-worker CSV ──
    success_path = os.path.join(args.log_dir, f"success_gpu{rank}.csv")
    fail_path = os.path.join(args.log_dir, f"fail_gpu{rank}.csv")
    header = ['id', 'class_name', 'clip_score', 'description', 'img_path']

    success_list = _deduplicate_success_rows(success_list)
    _write_rows_atomic(success_path, header, success_list)
    _write_rows_atomic(success_part_path, header, success_list)
    logger.info("Written success list: %d entries to %s", len(success_list), success_path)

    _write_rows_atomic(fail_path, header, fail_list)
    logger.info("Written fail list: %d entries to %s", len(fail_list), fail_path)

    incomplete_path = os.path.join(args.log_dir, f"incomplete_gpu{rank}.json")
    if incomplete_classes:
        atomic_json_dump(incomplete_classes, incomplete_path)
    elif os.path.exists(incomplete_path):
        os.remove(incomplete_path)

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
        return len([entry for entry in result.stdout.strip().split("\n") if entry])
    except Exception:
        return 1


def main() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    import time

    from utils import load_prompts
    main_start = time.perf_counter()
    args = parse_args()
    if args.max_target_number <= 0:
        raise ValueError("--max-target-number must be positive")
    if args.max_rounds <= 0:
        raise ValueError("--max-rounds must be positive")
    prompts = load_prompts(args.prompt_file)
    args.reflect_one_prompt = prompts.get("generate", {}).get("reflect_one_prompt")
    args.generate_pic_style = prompts.get("generate", {}).get("generate_pic_style") or ""
    if not args.reflect_one_prompt:
        raise ValueError("prompt file is missing generate.reflect_one_prompt")
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    if os.path.isdir(fail_img_dir):
        shutil.rmtree(fail_img_dir)
        print(f"[generate] Removed stale fail images: {fail_img_dir}")

    if args.num_gpus == 0:
        num_gpus = _detect_gpus()
    else:
        num_gpus = args.num_gpus

    load_csv_start = time.perf_counter()
    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = sorted(df.groupby('label')['text'].apply(list).items())
    load_class_semantics(args.class_mapping, [label for label, _ in grouped])
    args.target_counts = {
        str(label): min(args.max_target_number, len(texts))
        for label, texts in grouped
    }
    largest_target = max(args.target_counts.values(), default=0)
    if args.max_sample_attempts < largest_target:
        raise ValueError(
            "--max-sample-attempts must be at least the largest per-class target "
            f"({largest_target})"
        )
    if args.onepath and largest_target > 1:
        raise ValueError("--onepath is incompatible with a per-class target greater than 1")
    print(
        f"[generate] Per-class target=min(max_target_number={args.max_target_number}, "
        "description_count)"
    )
    load_csv_elapsed = time.perf_counter() - load_csv_start

    worker_start = time.perf_counter()
    if num_gpus <= 1:
        s, f = _worker(0, 1, [grouped], args, args.examples_dir)
        results = [(s, f)]
    else:
        print(f"[generate] Using {num_gpus} GPUs, {len(grouped)} classes total")
        chunks = [[] for _ in range(num_gpus)]
        for i, item in enumerate(grouped):
            chunks[i % num_gpus].append(item)

        mp.spawn(_worker, args=(num_gpus, chunks, args, args.examples_dir, None),
                 nprocs=num_gpus, join=True)
        results = [
            (
                _load_worker_csv(os.path.join(args.log_dir, f"success_gpu{rank}.csv")),
                _load_worker_csv(os.path.join(args.log_dir, f"fail_gpu{rank}.csv")),
            )
            for rank in range(num_gpus)
        ]

        print("[generate] All GPUs done.")
    worker_elapsed = time.perf_counter() - worker_start

    # ── 合并所有 worker 的 success / fail 列表 ──
    merge_start = time.perf_counter()
    all_success = []
    all_fail = []
    for s, f in results:
        all_success.extend(s)
        all_fail.extend(f)
    all_success = _deduplicate_success_rows(all_success)
    merge_elapsed = time.perf_counter() - merge_start

    # ── 移动失败图片到 data_dir/fail/ ──
    # args.data_dir is the generated image root, e.g. .../generated_imgs.
    generated_imgs_dir = os.path.abspath(args.data_dir)
    fail_img_dir = os.path.join(generated_imgs_dir, "fail")
    os.makedirs(fail_img_dir, exist_ok=True)

    updated_fail = []
    move_fail_start = time.perf_counter()
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
    move_fail_elapsed = time.perf_counter() - move_fail_start

    train_dir = os.path.join(generated_imgs_dir, "train")
    reconcile_start = time.perf_counter()
    all_success, missing_success, moved_unknown = _reconcile_train_images(
        train_dir, fail_img_dir, all_success
    )
    reconcile_elapsed = time.perf_counter() - reconcile_start
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

    expected_labels = [str(label) for label, _ in grouped]
    incomplete, unexpected_labels = _validate_success_counts(
        all_success, expected_labels, args.target_counts
    )
    if unexpected_labels:
        raise RuntimeError(f"Step3 has unexpected success labels: {unexpected_labels[:20]}")

    # ── 写入合并 CSV；只有全部类别完整时发布正式 success_list.csv ──
    write_csv_start = time.perf_counter()
    header = ['id', 'class_name', 'clip_score', 'description', 'img_path']
    os.makedirs(generated_imgs_dir, exist_ok=True)

    success_merged_path = os.path.join(generated_imgs_dir, "success_list.csv")
    success_partial_path = os.path.join(generated_imgs_dir, "success_list.partial.csv")
    incomplete_path = os.path.join(generated_imgs_dir, "incomplete.json")
    if incomplete:
        _write_rows_atomic(success_partial_path, header, all_success)
        atomic_json_dump({
            "target_counts": args.target_counts,
            "counts": incomplete,
        }, incomplete_path)
        if os.path.exists(success_merged_path):
            os.remove(success_merged_path)
    else:
        _write_rows_atomic(success_merged_path, header, all_success)
        for stale_path in (success_partial_path, incomplete_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    fail_merged_path = os.path.join(generated_imgs_dir, "fail_list.csv")
    _write_rows_atomic(fail_merged_path, header, updated_fail)
    write_csv_elapsed = time.perf_counter() - write_csv_start

    print(f"[generate] Success: {len(all_success)}, Fail: {len(updated_fail)}")
    print(f"[generate] Merged lists -> {generated_imgs_dir}/")
    print(f"[generate] Fail images moved to {fail_img_dir}")

    # ── 生成 fail examples (单文件) ──
    fail_examples_elapsed = 0.0
    if args.examples_dir and updated_fail:
        fail_examples_start = time.perf_counter()
        fail_examples_dir = os.path.join(args.examples_dir, "fail")
        os.makedirs(fail_examples_dir, exist_ok=True)

        fail_images_link = os.path.join(fail_examples_dir, "images")
        _ensure_relative_symlink(fail_images_link, fail_img_dir)

        fail_md_path = os.path.join(fail_examples_dir, "failed_examples.md")
        with open(fail_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Failed Images ({len(updated_fail)})\n\n")
            for k, (cid, cname, score, desc, path) in enumerate(updated_fail, 1):
                rel = f"images/{os.path.relpath(path, fail_img_dir)}"
                f.write(f"## Fail {k}\n\n")
                f.write(f"**Class ID:** {cid}\n\n")
                f.write(f"**Class Name:** {cname}\n\n")
                f.write(f"**Old Description:** {desc}\n\n")
                f.write(f'<img src="{rel}" alt="Failed Image {k}" loading="lazy">\n\n')
                f.write(f"**CLIP Score:** {score:.4f}\n\n---\n\n")
        print(f"[generate] Fail examples -> {fail_md_path}")
        fail_examples_elapsed = time.perf_counter() - fail_examples_start

    total_elapsed = time.perf_counter() - main_start
    print(
        "[generate] timing "
        f"total={total_elapsed:.3f}s "
        f"load_csv={load_csv_elapsed:.3f}s "
        f"workers={worker_elapsed:.3f}s "
        f"merge_results={merge_elapsed:.3f}s "
        f"move_fail={move_fail_elapsed:.3f}s "
        f"reconcile_train={reconcile_elapsed:.3f}s "
        f"write_csv={write_csv_elapsed:.3f}s "
        f"fail_examples={fail_examples_elapsed:.3f}s"
    )
    if incomplete:
        raise RuntimeError(
            f"Step3 incomplete for {len(incomplete)} classes; report: {incomplete_path}"
        )


if __name__ == "__main__":
    main()
