#!/usr/bin/env python3
"""
CLIP 语义区分能力验证
用 ImageNet-LT 真实图片 + 类别名测试 CLIP 能否正确匹配语义
"""
import os
import sys
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.clip_score import score
from config import CLIP_BACKEND, CLIP_MODEL_NAME, IMAGENET_DIR
from data_txt.imagenet_label_mapping import lable2name
from data.data_loader import ImageNetLTDataLoader


def main():
    n = 200
    print(f"CLIP 后端: {CLIP_BACKEND}, 模型: {CLIP_MODEL_NAME}")
    print(f"数据: {IMAGENET_DIR}")
    print(f"测试 {n} 组正确 vs 错误类别名匹配\n")

    loader = ImageNetLTDataLoader(IMAGENET_DIR, split="test", batch_size=1, num_workers=0)
    dataset = loader.dataset
    total = len(dataset)
    indices = random.Random(42).sample(range(total), min(n, total))

    all_class_names = [lable2name.get(i, f"class_{i}").split(", ")[0] for i in range(len(lable2name))]

    correct_scores = []
    wrong_scores = []
    correct_higher = 0

    for idx, sample_idx in enumerate(tqdm(indices, desc="[val] Scoring")):
        img_path = dataset.img_paths[sample_idx]
        label = dataset.labels[sample_idx]

        class_name = lable2name.get(label, f"class_{label}").split(", ")[0]
        correct_caption = f"A photo of a {class_name.lower()}"
        wrong_label = random.Random(sample_idx).choice([c for c in all_class_names if c != class_name])
        wrong_caption = f"A photo of a {wrong_label.lower()}"

        s_correct = score(img_path, correct_caption)
        s_wrong = score(img_path, wrong_caption)

        correct_scores.append(s_correct)
        wrong_scores.append(s_wrong)

        if s_correct > s_wrong:
            correct_higher += 1

        if idx < 5:
            print(f"\n  [{idx}] label={label} ({class_name})")
            print(f"      correct({s_correct:.4f}): {correct_caption}")
            print(f"      wrong  ({s_wrong:.4f}): {wrong_caption}")
            print(f"      {'✓ correct > wrong' if s_correct > s_wrong else '✗ correct <= wrong'}")

    correct_scores = np.array(correct_scores)
    wrong_scores = np.array(wrong_scores)

    print(f"\n{'='*60}")
    print(f"  CLIP 语义区分验证结果 ({n} 组 ImageNet 图片)")
    print(f"{'='*60}")
    print(f"  Acc@1 (correct > wrong): {correct_higher}/{n} = {correct_higher/n*100:.1f}%")
    print(f"  Correct mean ± std:      {correct_scores.mean():.4f} ± {correct_scores.std():.4f}")
    print(f"  Wrong   mean ± std:      {wrong_scores.mean():.4f} ± {wrong_scores.std():.4f}")
    print(f"  Score gap:               {correct_scores.mean() - wrong_scores.mean():.4f}")
    print(f"  Correct min/max:         {correct_scores.min():.4f} / {correct_scores.max():.4f}")
    print(f"  Wrong   min/max:         {wrong_scores.min():.4f} / {wrong_scores.max():.4f}")
    print(f"{'='*60}")

    # 细粒度: goldfish vs tench
    print(f"\n  --- 细粒度: goldfish vs tench ---")
    goldfish = lable2name[1].split(", ")[0]
    tench = lable2name[0].split(", ")[0]
    gf_scores, t_scores = [], []
    total_pairs = 0
    for sample_idx in indices:
        label = dataset.labels[sample_idx]
        if label == 1:
            img_path = dataset.img_paths[sample_idx]
            s_gf = score(img_path, f"A photo of a {goldfish.lower()}")
            s_t = score(img_path, f"A photo of a {tench.lower()}")
            gf_scores.append(s_gf)
            t_scores.append(s_t)
            total_pairs += 1
    if gf_scores:
        gf_correct = sum(1 for g, t in zip(gf_scores, t_scores) if g > t)
        print(f"  Goldfish test count:       {total_pairs}")
        print(f"  Goldfish > tench:          {gf_correct}/{total_pairs} ({gf_correct/total_pairs*100:.1f}%)")
        print(f"  Goldfish score:            {np.mean(gf_scores):.4f} ± {np.std(gf_scores):.4f}")
        print(f"  Tench score:               {np.mean(t_scores):.4f} ± {np.std(t_scores):.4f}")
        print(f"  Score gap:                 {np.mean(gf_scores) - np.mean(t_scores):.4f}")

    print(f"\n结果日志: /tmp/clip_validation_log.txt")


if __name__ == "__main__":
    from torchvision import transforms
    main()
