"""
文本到图像生成流水线 (LTGC Step 3)
使用 Stable Diffusion (HuggingFace) 替代 DALL-E 2
使用 DeepSeek Chat 替代 GPT-4-turbo

流程:
  1. 读取扩展描述文件
  2. 对每个类别的每条描述，调用 hf_gen 生成图像
  3. 使用 CLIP filter 筛选高质量图像
  4. 不满足阈值的图像，调用 description_refine 润色后重新生成
"""
import pandas as pd
import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_config import DESCRIPTIONS_DIR, DATA_DIR
from model.clip_score import score as clip_filter
from model.hf_gen import hf_gen, description_refine, get_cls_template


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text-to-Image Generation')
    parser.add_argument('-ext', '--extended_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv'),
                        type=str,
                        help='Extended description CSV path')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, type=str,
                        help='Output root directory')
    parser.add_argument('-t', '--thresh', default=0.6, type=float,
                        help='CLIP filter threshold')
    parser.add_argument('-r', '--max_rounds', default=3, type=int,
                        help='Max retry rounds for CLIP filter')
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped_list = df.groupby('label')['text'].apply(list).to_dict()

    total_labels = len(grouped_list)
    for label_idx, (label, texts) in enumerate(grouped_list.items()):
        cls_name = get_cls_template(label, label)
        dir_path = os.path.join(args.data_dir, 'gen_train', str(label))
        os.makedirs(dir_path, exist_ok=True)

        print(f"[draw_t2i] Processing class {label} ({label_idx + 1}/{total_labels}), {len(texts)} descriptions")

        for text_i, text in enumerate(texts):
            saved_path = os.path.join(dir_path, f"{label}_{text_i}.JPEG")

            if os.path.exists(saved_path):
                print(f"[draw_t2i] Skip existing: {saved_path}")
                continue

            img_path = hf_gen(saved_path, text, saved=True)
            if img_path is None:
                continue

            score = 0.0
            for round_num in range(args.max_rounds):
                if round_num == 0:
                    score = clip_filter(img_path, cls_name)
                else:
                    refine_text = description_refine(text, cls_name)
                    saved_path_refine = os.path.join(
                        dir_path, f"{label}_{text_i}_refine{round_num}.JPEG"
                    )
                    img_path = hf_gen(saved_path_refine, refine_text, saved=True)
                    if img_path is None:
                        break
                    score = clip_filter(img_path, cls_name)

                if score >= args.thresh:
                    print(f"[draw_t2i] Quality check passed: score={score:.4f} >= {args.thresh}")
                    break
                else:
                    print(f"[draw_t2i] Quality check failed: score={score:.4f} < {args.thresh}, retry {round_num + 1}/{args.max_rounds}")

    print(f"[draw_t2i] Done. Processed {total_labels} classes")


if __name__ == "__main__":
    main()
