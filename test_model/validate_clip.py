#!/usr/bin/env python3
"""
CLIP 语义区分能力验证
对 test_class_ids 中每个类，抽取 augmented 和 original 图片进行 CLIP score，
结果记录在 markdown 文档中，图片通过软链接在 md 中直接显示。
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.clip_score import score_batch
from config import CLIP_BACKEND, CLIP_MODEL_NAME, IMAGENET_DIR
from data_txt.imagenet_label_mapping import lable2name


# ── 工具函数 ──────────────────────────────────────────────────────


def get_class_ids_from_dir(path):
    """从目录中提取所有数字类 ID（子目录名为纯数字）"""
    if not os.path.isdir(path):
        return []
    ids = []
    for entry in sorted(os.listdir(path)):
        p = os.path.join(path, entry)
        if os.path.isdir(p):
            try:
                ids.append(int(entry))
            except ValueError:
                continue
    return ids


def get_images_for_class(base_dir, class_id, max_n=None):
    """获取某个类别下的图片路径列表，最多 max_n 张"""
    class_dir = os.path.join(base_dir, str(class_id))
    if not os.path.isdir(class_dir):
        return []
    exts = {'.png', '.jpg', '.jpeg', '.JPEG', '.PNG', '.JPG'}
    imgs = sorted([
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if os.path.splitext(f)[1] in exts
    ])
    if max_n and len(imgs) > max_n:
        imgs = random.sample(imgs, max_n)
    return imgs


def build_symlinks(img_dir, results):
    """在 img_dir 下为所有图片建立软链接，返回 (aug_refs, org_refs) 相对路径映射"""
    os.makedirs(img_dir, exist_ok=True)
    aug_refs = {}  # (class_id, idx) → rel_path
    org_refs = {}

    for r in results:
        cid = r["class_id"]
        for i, (full_path, _score) in enumerate(r["aug_full"]):
            ext = os.path.splitext(full_path)[1]
            link_name = f"aug_{cid}_{i}{ext}"
            link_path = os.path.join(img_dir, link_name)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(full_path), link_path)
            aug_refs[(cid, i)] = f"images/{link_name}"

        for i, (full_path, _score) in enumerate(r["org_full"]):
            ext = os.path.splitext(full_path)[1]
            link_name = f"org_{cid}_{i}{ext}"
            link_path = os.path.join(img_dir, link_name)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(full_path), link_path)
            org_refs[(cid, i)] = f"images/{link_name}"

    return aug_refs, org_refs


# ── 参数解析 ──────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="CLIP score 对比：augmented vs original 图片，输出 markdown 报告"
    )
    p.add_argument("--augment-dir", "-a", required=True,
                   help="增强实验目录（含 generated_imgs/train/）")
    p.add_argument("--original-dir", "-o", default=IMAGENET_DIR,
                   help="原始数据集目录")
    p.add_argument("--output", "-O", default=None,
                   help="Markdown 输出路径（默认 {augment-dir}_validate/clip_validation.md）")
    p.add_argument("--max-samples", "-n", type=int, default=10,
                   help="每类最多采样图片数 (default: 10)")
    p.add_argument("--class-mapping", "-c", default=None,
                   help="JSON 类别映射文件（类ID→类名）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (default: 42)")
    return p.parse_args()


# ── 主逻辑 ────────────────────────────────────────────────────────


def main():
    args = parse_args()
    random.seed(args.seed)

    # 加载类别映射
    class_map = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping) as f:
            class_map = json.load(f)

    aug_train = os.path.join(args.augment_dir, "generated_imgs", "train")
    org_train = os.path.join(args.original_dir, "train")
    output_path = args.output or os.path.join(args.augment_dir + "_validate", "clip_validation.md")

    # 1. 获取 test_class_ids
    test_class_ids = get_class_ids_from_dir(aug_train)
    if not test_class_ids:
        print(f"ERROR: no augmented images found in {aug_train}")
        print("  (实验 Step 3 可能未运行或未完成)")
        return

    print(f"CLIP 后端: {CLIP_BACKEND}, 模型: {CLIP_MODEL_NAME}")
    print(f"增强数据: {aug_train}  ({len(test_class_ids)} classes)")
    print(f"原始数据: {org_train}")
    print(f"每类采样: ≤{args.max_samples} 张\n")

    # 2. 逐类评分
    results = []
    all_aug_scores = []
    all_org_scores = []

    for class_id in test_class_ids:
        if class_map:
            class_name = class_map.get(str(class_id), f"class_{class_id}")
        else:
            class_name = lable2name.get(class_id, f"class_{class_id}").split(",")[0]
        caption = f"A photo of a {class_name.lower()}"

        aug_imgs = get_images_for_class(aug_train, class_id, args.max_samples)
        org_imgs = get_images_for_class(org_train, class_id, args.max_samples)
        n_aug, n_org = len(aug_imgs), len(org_imgs)

        aug_vec = score_batch(aug_imgs, [caption] * n_aug) if aug_imgs else []
        org_vec = score_batch(org_imgs, [caption] * n_org) if org_imgs else []

        all_aug_scores.extend(aug_vec)
        all_org_scores.extend(org_vec)

        # (full_path, basename, score) 三元组
        aug_full = list(zip(aug_imgs, aug_vec))
        org_full = list(zip(org_imgs, org_vec))

        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "n_aug": n_aug,
            "n_org": n_org,
            "aug_mean": float(np.mean(aug_vec)) if aug_vec else None,
            "aug_std":  float(np.std(aug_vec))  if aug_vec else None,
            "org_mean": float(np.mean(org_vec)) if org_vec else None,
            "org_std":  float(np.std(org_vec))  if org_vec else None,
            "aug_full": aug_full,
            "org_full": org_full,
        })
        print(f"  [{class_id:>4}] {class_name:<20s}  aug={n_aug:<2d}  org={n_org:<2d}"
              f"  aug_mean={aug_vec and np.mean(aug_vec):.4f}  org_mean={org_vec and np.mean(org_vec):.4f}")

    # 3. 建立 images/ 软链接
    out_dir = os.path.dirname(output_path)
    img_dir = os.path.join(out_dir, "images")
    aug_refs, org_refs = build_symlinks(img_dir, results)
    print(f"Symlinks: {len(os.listdir(img_dir))} images → {img_dir}")

    # 4. 写入 Markdown
    os.makedirs(out_dir, exist_ok=True)
    aug_arr = np.array(all_aug_scores) if all_aug_scores else np.array([])
    org_arr = np.array(all_org_scores) if all_org_scores else np.array([])

    with open(output_path, "w") as f:
        f.write("# CLIP Score Validation Report\n\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **CLIP 后端**: `{CLIP_BACKEND}`\n")
        f.write(f"- **CLIP 模型**: `{CLIP_MODEL_NAME}`\n")
        f.write(f"- **增强数据**: `{aug_train}`\n")
        f.write(f"- **原始数据**: `{org_train}`\n")
        f.write(f"- **测试类别数**: {len(test_class_ids)}\n")
        f.write(f"- **每类采样上限**: {args.max_samples}\n")
        f.write(f"- **随机种子**: {args.seed}\n\n")

        # ── Summary ──
        f.write("## Summary\n\n")
        f.write("| Source | Count | Mean | Std | Min | Max |\n")
        f.write("|--------|-------|------|-----|-----|-----|\n")
        if len(aug_arr):
            f.write(f"| Augmented | {len(aug_arr)} | {aug_arr.mean():.4f} | {aug_arr.std():.4f} "
                    f"| {aug_arr.min():.4f} | {aug_arr.max():.4f} |\n")
        if len(org_arr):
            f.write(f"| Original   | {len(org_arr)} | {org_arr.mean():.4f} | {org_arr.std():.4f} "
                    f"| {org_arr.min():.4f} | {org_arr.max():.4f} |\n")
        if len(aug_arr) and len(org_arr):
            delta = aug_arr.mean() - org_arr.mean()
            f.write(f"\n**Augmented − Original = {delta:+.4f}**\n")
        f.write("\n")

        # ── Per-Class ──
        f.write("## Per-Class Results\n\n")
        f.write("| ID | Class | #Aug | #Org | Aug Mean | Aug Std | "
                "Org Mean | Org Std | Δ Mean |\n")
        f.write("|----|-------|------|------|----------|---------|"
                "----------|---------|--------|\n")
        for r in results:
            fm = lambda x: f"{x:.4f}" if x is not None else "—"
            d = f"{r['aug_mean'] - r['org_mean']:+.4f}" if r["aug_mean"] and r["org_mean"] else "—"
            f.write(f"| {r['class_id']} | {r['class_name']} "
                    f"| {r['n_aug']} | {r['n_org']} "
                    f"| {fm(r['aug_mean'])} | {fm(r['aug_std'])} "
                    f"| {fm(r['org_mean'])} | {fm(r['org_std'])} | {d} |\n")
        f.write("\n")

        # ── Per-Image ──
        f.write("## Per-Image Scores\n\n")
        for r in results:
            cid = r['class_id']
            f.write(f"### {cid}: {r['class_name']}\n\n")
            f.write("| # | Augmented | Score | Original | Score |\n")
            f.write("|---|-----------|-------|----------|-------|\n")
            mx = max(r['n_aug'], r['n_org'])
            for i in range(mx):
                if i < r['n_aug']:
                    a_ref = aug_refs.get((cid, i), "")
                    a_sc  = f"{r['aug_full'][i][1]:.4f}"
                    a_cell = f"![aug]({a_ref})"
                else:
                    a_cell, a_sc = "—", "—"

                if i < r['n_org']:
                    o_ref = org_refs.get((cid, i), "")
                    o_sc  = f"{r['org_full'][i][1]:.4f}"
                    o_cell = f"![org]({o_ref})"
                else:
                    o_cell, o_sc = "—", "—"

                f.write(f"| {i + 1} | {a_cell} | {a_sc} | {o_cell} | {o_sc} |\n")
            f.write("\n")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
