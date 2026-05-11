"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 → SD 生成图像 → CLIP 质量筛选 → 保存
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR, DATA_DIR
from model.clip_score import score
from model.image_gen import generate
from data_txt.imagenet_label_mapping import get_readable_name

# distill_distentictive_features_prompt = (
#     f"Please use Template 2 to summarize the most distinctive features of class [y]}. Template 2: A photo of the class [y] with {feature 1}{feature 2}{...}."
# )


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text → Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv'),
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, help='Output root')
    parser.add_argument('-t', '--thresh', default=0.25, type=float, help='CLIP score threshold')
    parser.add_argument('-r', '--max_rounds', default=3, type=int, help='Max retry rounds')
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = df.groupby('label')['text'].apply(list).to_dict()

    total = len(grouped)
    for label_idx, (label, texts) in enumerate(grouped.items()):
        class_name = get_readable_name(int(label)).split(", ")[0]
        dir_path = os.path.join(args.data_dir, 'gen_train', str(label))
        os.makedirs(dir_path, exist_ok=True)

        print(f"[generate] Class {label} ({label_idx + 1}/{total}), {len(texts)} descriptions")

        for text_i, text in enumerate(texts):
            saved_path = os.path.join(dir_path, f"{label}_{text_i}.JPEG")
            if os.path.exists(saved_path):
                print(f"[generate] Skip: {saved_path}")
                continue
            
            # use a differer approach to regenerae 
            int max_retry = 10
            while max_retry > 0:
                img_path = generate(text, saved_path)
                if img_path is None:
                    continue

                #todo simplify prompt and easier clip checkability
                clip_score = score(img_path, f"A photo of a {class_name.lower()}")

                if clip_score >= args.thresh:
                    print(f"[generate] Score {clip_score:.4f} >= {args.thresh}, accepted")
                    break
                else:
                    print(f"[generate] Score {clip_score:.4f} < {args.thresh}")

                max_retry -= 1

    print(f"[generate] Done. {total} classes processed.")


if __name__ == "__main__":
    main()
