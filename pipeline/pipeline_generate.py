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
    # ── 进程隔离：每 worker 绑定一张 GPU（通过 CUDA_VISIBLE_DEVICES）──
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"generate_gpu_{rank}.log")
    logger = setup_logger(f"GPU{rank}", log_path)

    # ── 延迟导入：在 CUDA_VISIBLE_DEVICES 设置后再加载模型模块 ──
    from config import GENERATION_EXAMPLE_DIR
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
    # 主循环：逐类处理
    # ════════════════════════════════════════════════════════════════
    for label_idx, (label, texts) in enumerate(class_list):
        class_name = _class_name(label)
        dir_path = os.path.join(args.data_dir, str(label))
        os.makedirs(dir_path, exist_ok=True)

        # ── 准备描述列表和 CLIP 目标 prompt ──
        generation_prompts = [
            str(t).strip() if not pd.isna(t) and str(t).strip() else f'A photo of a {class_name}' for t in texts
        ]
        clip_prompt = f'A photo of a {class_name}'
        n = len(generation_prompts)
        logger.info("[class %s %s] %d descs, thresh=%.2f, max_rounds=%d",
                    label, class_name, n, args.thresh, args.max_rounds)

        # ── 单图模式：逐张生成 + 单张 CLIP 评分（不走批量流水线）──
        if args.onepath:
            for text_i, generation_prompt in enumerate(generation_prompts):
                saved_path = os.path.join(args.data_dir, 'gen_train-onepath.JPEG')
                desc_accepted = False
                for attempt in range(args.max_rounds):
                    img_path = generate(generation_prompt, saved_path)
                    if img_path is None:
                        continue
                    s = score(img_path, clip_prompt)
                    if s >= args.thresh:
                        if examples_dir:
                            gen_records[label].append((generation_prompt, s, img_path))
                        desc_accepted = True
                        logger.info("[class %s %s] desc %d/%d accepted in round %d (%.4f)",
                                    label, class_name, text_i + 1, n, attempt + 1, s)
                        break
                    if attempt < args.max_rounds - 1:
                        refined = reflect_one_description(
                            generation_prompt, class_name,
                            prompt=args.reflect_one_prompt,
                        )
                        if refined and validate_description(refined, class_name):
                            generation_prompt = refined
                        elif refined:
                            logger.info("[class %s %s] desc %d/%d refine rejected",
                                        label, class_name, text_i + 1, n)
                unload_text_llm()
                if not desc_accepted:
                    logger.info("[class %s %s] desc %d/%d failed after %d rounds",
                                label, class_name, text_i + 1, n, args.max_rounds)
            continue

        # ── 批量模式：按 batch 分块，每块内 SD 批量生成 → CLIP 批量评分 → 低分 refine 重试 ──
        save_paths = [os.path.join(dir_path, f"{label}_{i}.JPEG") for i in range(n)]
        accepted = [False] * n
        bs = args.batch

        for chunk_start in range(0, n, bs):
            chunk_end = min(chunk_start + bs, n)
            chunk_ids = list(range(chunk_start, chunk_end))

            for attempt in range(args.max_rounds):
                # 只处理本轮仍未接受的描述
                pending = [i for i in chunk_ids if not accepted[i]]
                if not pending:
                    break

                # Step A: SD 批量生成图像
                batch_prompts = [generation_prompts[i] for i in pending]
                batch_paths = [save_paths[i] for i in pending]
                img_paths = generate_batch(batch_prompts, batch_paths)
                unload_sd()

                valid = [(i, p) for i, p in zip(pending, img_paths) if p is not None]
                if not valid:
                    break

                # Step B: CLIP 批量评分
                v_idx, v_paths = zip(*valid)
                clip_scores = score_batch(
                    list(v_paths), [clip_prompt] * len(v_paths)
                )

                # Step C: 按阈值分流 accepted / rejected
                n_acc = 0
                n_rej = 0
                for idx, s in zip(v_idx, clip_scores):
                    if s >= args.thresh:
                        if examples_dir:
                            gen_records[label].append((generation_prompts[idx], s, save_paths[idx]))
                        accepted[idx] = True
                        n_acc += 1
                    else:
                        n_rej += 1
                        logger.warning("[class %s %s] desc %d/%d rejected: %s",
                                    label, class_name, idx + 1, n, batch_prompts[idx])
                logger.info("[class %s %s] round %d/%d: accepted=%d rejected=%d",
                            label, class_name, attempt + 1, args.max_rounds, n_acc, n_rej)

                # Step D: 低分描述 refine（调用 text_llm 润色后下一轮重试）
                if attempt < args.max_rounds - 1:
                    n_refined = 0
                    for i in chunk_ids:
                        if not accepted[i]:
                            refined = reflect_one_description(
                                generation_prompts[i], class_name,
                                prompt=args.reflect_one_prompt,
                                enable_thinking=True, do_sample=False, 
                                temperature=0.2, max_token=1000)
                            if not refined:
                                logger.warn("[class %s %s] round %d/%d: refined an empty description")
                            if refined and validate_description(refined, class_name):
                                generation_prompts[i] = refined
                                n_refined += 1
                                logger.info("[class %s %s] round %d/%d: refined: %s",
                                    label, class_name, attempt + 1, args.max_rounds, refined)
                            else:
                                logger.warn("[class %s %s] round %d/%d: fail to validate refined dst: %s",
                                    label, class_name, attempt + 1, args.max_rounds, refined)
                    unload_text_llm()
                        
                if all(accepted[i] for i in chunk_ids):
                    break

        failed = sum(1 for a in accepted if not a)
        logger.info("[class %s %s] done: %d/%d accepted, %d failed",
                    label, class_name, n - failed, n, failed)

        # ── 写入 Markdown：追加 Generated Images 段落 ──
        if examples_dir and gen_records.get(label):
            os.makedirs(examples_dir, exist_ok=True)
            safe_name = class_name.replace(' ', '_').replace('/', '_')
            md_path = os.path.join(examples_dir, f"{label}_{safe_name}.md")
            records = gen_records[label]
            with open(md_path, 'a') as f:
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
            with open(md_path, 'a') as f:
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

    mp.spawn(
        _worker,
        args=(num_gpus, chunks, args, args.examples_dir),
        nprocs=num_gpus,
        join=True,
    )

    print(f"[generate] All GPUs done.")


if __name__ == "__main__":
    main()
