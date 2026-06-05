#!/usr/bin/env python3
"""
CLIP 语义区分能力验证
对 test_class_ids 中每个类，三组对比：
  - Augmented（生成图）
  - Original（正确类原图）
  - Wrong-Class Original（随机错误类原图）
每组按图片记录 score，输出 markdown 报告，图片通过软链接显示。
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
    """从目录中提取所有数字类 ID"""
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
    """获取某个类别下的图片路径列表"""
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


def resolve_class_name(class_id, class_map=None):
    """解析类名"""
    if class_map:
        return class_map.get(str(class_id), f"class_{class_id}")
    return lable2name.get(class_id, f"class_{class_id}").split(",")[0]


def build_symlinks(img_dir, results):
    """在 img_dir 下为所有图片建立软链接，返回 (aug_refs, org_refs, wrong_refs)"""
    os.makedirs(img_dir, exist_ok=True)
    aug_refs = {}
    org_refs = {}
    wrong_refs = {}

    for r in results:
        cid = r["class_id"]
        for i, (full_path, _score) in enumerate(r["aug_full"]):
            ext = os.path.splitext(full_path)[1]
            name = f"aug_{cid}_{i}{ext}"
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                os.symlink(os.path.abspath(full_path), p)
            aug_refs[(cid, i)] = f"images/{name}"

        for i, (full_path, _score) in enumerate(r["org_full"]):
            ext = os.path.splitext(full_path)[1]
            name = f"org_{cid}_{i}{ext}"
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                os.symlink(os.path.abspath(full_path), p)
            org_refs[(cid, i)] = f"images/{name}"

        for i, (full_path, _score, w_cid) in enumerate(r["wrong_full"]):
            ext = os.path.splitext(full_path)[1]
            name = f"wrong_{cid}_{i}{ext}"
            p = os.path.join(img_dir, name)
            if not os.path.exists(p):
                os.symlink(os.path.abspath(full_path), p)
            wrong_refs[(cid, i)] = f"images/{name}"

    return aug_refs, org_refs, wrong_refs


# ── 参数解析 ──────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="CLIP score 三组对比：augmented vs original vs wrong-class"
    )
    p.add_argument("--augment-dir", "-a", required=True,
                   help="增强实验目录（含 generated_imgs/train/）")
    p.add_argument("--original-dir", "-o", default=IMAGENET_DIR,
                   help="原始数据集目录")
    p.add_argument("--output", "-O", default=None,
                   help="Markdown 输出路径（默认 {augment-dir}_validate/clip_validation.md）")
    p.add_argument("--max-samples", "-n", type=int, default=10,
                   help="每类每组最多采样数 (default: 10)")
    p.add_argument("--class-mapping", "-c", default=None,
                   help="JSON 类别映射文件")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (default: 42)")
    return p.parse_args()


# ── 主逻辑 ────────────────────────────────────────────────────────


def main():
    args = parse_args()
    random.seed(args.seed)

    class_map = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping) as f:
            class_map = json.load(f)

    aug_train = os.path.join(args.augment_dir, "generated_imgs", "train")
    org_train = os.path.join(args.original_dir, "train")
    output_path = args.output or os.path.join(args.augment_dir + "_validate", "clip_validation.md")

    # 1. 获取 test_class_ids (有生成图的类)
    test_class_ids = get_class_ids_from_dir(aug_train)
    if not test_class_ids:
        print(f"ERROR: no augmented images found in {aug_train}")
        return

    # 所有可用的原始类 ID（用于随机选错误类）
    all_org_ids = get_class_ids_from_dir(org_train)

    print(f"CLIP 后端: {CLIP_BACKEND}, 模型: {CLIP_MODEL_NAME}")
    print(f"增强数据: {aug_train}  ({len(test_class_ids)} classes)")
    print(f"原始数据: {org_train}  ({len(all_org_ids)} classes)")
    print(f"每类采样: ≤{args.max_samples} 张\n")

    # 2. 逐类评分
    results = []
    all_aug, all_org, all_wrong = [], [], []

    for class_id in test_class_ids:
        class_name = resolve_class_name(class_id, class_map)
        caption = f"A photo of a {class_name.lower()}"

        # ── Augmented ──
        aug_imgs = get_images_for_class(aug_train, class_id, args.max_samples)
        na = len(aug_imgs)
        aug_vec = score_batch(aug_imgs, [caption] * na) if aug_imgs else []
        all_aug.extend(aug_vec)

        # ── Original (正确类) ──
        org_imgs = get_images_for_class(org_train, class_id, args.max_samples)
        no = len(org_imgs)
        org_vec = score_batch(org_imgs, [caption] * no) if org_imgs else []
        all_org.extend(org_vec)

        # ── Wrong-Class Original ──
        wrong_candidates = [c for c in all_org_ids if c != class_id]
        wrong_class_id = random.choice(wrong_candidates) if wrong_candidates else class_id
        wrong_class_name = resolve_class_name(wrong_class_id, class_map)
        wrong_imgs = get_images_for_class(org_train, wrong_class_id, args.max_samples)
        nw = len(wrong_imgs)
        wrong_vec = score_batch(wrong_imgs, [caption] * nw) if wrong_imgs else []
        all_wrong.extend(wrong_vec)

        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "wrong_class_id": wrong_class_id,
            "wrong_class_name": wrong_class_name,
            "n_aug": na, "n_org": no, "n_wrong": nw,
            "aug_mean": float(np.mean(aug_vec)) if aug_vec else None,
            "aug_std":  float(np.std(aug_vec))  if aug_vec else None,
            "org_mean": float(np.mean(org_vec)) if org_vec else None,
            "org_std":  float(np.std(org_vec))  if org_vec else None,
            "wrong_mean": float(np.mean(wrong_vec)) if wrong_vec else None,
            "wrong_std":  float(np.std(wrong_vec))  if wrong_vec else None,
            "aug_full": list(zip(aug_imgs, aug_vec)),
            "org_full": list(zip(org_imgs, org_vec)),
            "wrong_full": [(p, s, wrong_class_id) for p, s in zip(wrong_imgs, wrong_vec)],
        })
        print(f"  [{class_id:>4}] {class_name:<20s}  aug={na} μ={aug_vec and np.mean(aug_vec):.4f}"
              f"  org={no} μ={org_vec and np.mean(org_vec):.4f}"
              f"  wrong({wrong_class_id}:{wrong_class_name:<15s})={nw} μ={wrong_vec and np.mean(wrong_vec):.4f}")

    # 3. 建立 soft link
    out_dir = os.path.dirname(output_path)
    img_dir = os.path.join(out_dir, "images")
    aug_refs, org_refs, wrong_refs = build_symlinks(img_dir, results)
    print(f"Symlinks: {len(os.listdir(img_dir))} → {img_dir}")

    # 4. 写 Markdown
    os.makedirs(out_dir, exist_ok=True)
    A = np.array(all_aug) if all_aug else np.array([])
    O = np.array(all_org) if all_org else np.array([])
    W = np.array(all_wrong) if all_wrong else np.array([])

    with open(output_path, "w") as f:
        f.write("# CLIP Score Validation Report\n\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **CLIP**: `{CLIP_BACKEND}` / `{CLIP_MODEL_NAME}`\n")
        f.write(f"- **增强数据**: `{aug_train}`\n")
        f.write(f"- **原始数据**: `{org_train}`\n")
        f.write(f"- **测试类别**: {len(test_class_ids)}\n")
        f.write(f"- **每类采样**: ≤{args.max_samples}\n")
        f.write(f"- **Seed**: {args.seed}\n\n")

        # ── Summary ──
        f.write("## Summary\n\n")
        f.write("| Source | Count | Mean | Std | Min | Max |\n")
        f.write("|--------|-------|------|-----|-----|-----|\n")
        for label, arr in [("Augmented", A), ("Original", O), ("Wrong-Class", W)]:
            if len(arr):
                f.write(f"| {label} | {len(arr)} | {arr.mean():.4f} | {arr.std():.4f} "
                        f"| {arr.min():.4f} | {arr.max():.4f} |\n")
        f.write("\n")

        # ── Per-Class ──
        f.write("## Per-Class Results\n\n")
        hdr = ("| ID | Class | Wrong | #Aug | #Org | #Wrong | "
               "Aug μ | Aug σ | Org μ | Org σ | Wrong μ | Wrong σ | Δ(A-O) | Δ(A-W) |")
        f.write(hdr + "\n")
        sep = ("|----|-------|-------|------|------|--------|"
               "-------|-------|-------|-------|---------|---------|--------|--------|")
        f.write(sep + "\n")
        for r in results:
            fm = lambda x: f"{x:.4f}" if x is not None else "—"
            d_ao = f"{r['aug_mean'] - r['org_mean']:+.4f}" if r["aug_mean"] and r["org_mean"] else "—"
            d_aw = f"{r['aug_mean'] - r['wrong_mean']:+.4f}" if r["aug_mean"] and r["wrong_mean"] else "—"
            f.write(f"| {r['class_id']} | {r['class_name']} | {r['wrong_class_id']}:{r['wrong_class_name']} "
                    f"| {r['n_aug']} | {r['n_org']} | {r['n_wrong']} "
                    f"| {fm(r['aug_mean'])} | {fm(r['aug_std'])} "
                    f"| {fm(r['org_mean'])} | {fm(r['org_std'])} "
                    f"| {fm(r['wrong_mean'])} | {fm(r['wrong_std'])} "
                    f"| {d_ao} | {d_aw} |\n")
        f.write("\n")

        # ── Per-Image ──
        f.write("## Per-Image Scores\n\n")
        for r in results:
            cid = r['class_id']
            f.write(f"### {cid}: {r['class_name']}\n\n")
            f.write(f"> Wrong class: **{r['wrong_class_id']}:{r['wrong_class_name']}**\n\n")
            f.write("| # | Augmented | Score | Original | Score | Wrong-Class | Wrong Name | Score |\n")
            f.write("|---|-----------|-------|----------|-------|-------------|------------|-------|\n")
            mx = max(r['n_aug'], r['n_org'], r['n_wrong'])
            for i in range(mx):
                # Augmented
                if i < r['n_aug']:
                    a_cell = f"![aug]({aug_refs.get((cid, i), '')})"
                    a_sc   = f"{r['aug_full'][i][1]:.4f}"
                else:
                    a_cell, a_sc = "—", "—"
                # Original
                if i < r['n_org']:
                    o_cell = f"![org]({org_refs.get((cid, i), '')})"
                    o_sc   = f"{r['org_full'][i][1]:.4f}"
                else:
                    o_cell, o_sc = "—", "—"
                # Wrong-Class
                if i < r['n_wrong']:
                    w_cell = f"![wrong]({wrong_refs.get((cid, i), '')})"
                    w_sc   = f"{r['wrong_full'][i][1]:.4f}"
                    w_name = r['wrong_class_name']
                else:
                    w_cell, w_sc, w_name = "—", "—", "—"

                f.write(f"| {i + 1} | {a_cell} | {a_sc} | {o_cell} | {o_sc} | {w_cell} | {w_name} | {w_sc} |\n")
            f.write("\n")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
