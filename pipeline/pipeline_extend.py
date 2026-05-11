"""
LTGC 流水线 - Step 2: 描述扩展
读取已有描述 → 本地 LLM 生成多样化变体 → 保存 CSV
"""
import os
import sys
import csv
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR
from model.text_llm import extend_descriptions
from data_txt.imagenet_label_mapping import get_readable_name


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 2: Description → Extended Descriptions')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Input descriptions CSV')
    parser.add_argument('-m', '--max_generate_num', default=200, type=int,
                        help='Max descriptions per class')
    parser.add_argument('-ext', '--extended_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv'),
                        help='Output extended CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)

    df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
    grouped = df.groupby('label')['text'].apply(list).to_dict()

    total = len(grouped)
    for cls_idx, (label, texts) in enumerate(grouped.items()):
        print(f"[extend] Class {label} ({cls_idx + 1}/{total}): {len(texts)} existing")

        while len(texts) < args.max_generate_num:
            class_name = get_readable_name(int(label)).split(", ")[0]
            new_texts = extend_descriptions(texts, class_name)

            if not new_texts:
                print(f"[extend] No new descriptions, stopping")
                break

            texts.extend(new_texts)

            with open(args.extended_description_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for t in new_texts:
                    writer.writerow([label, t])

            print(f"[extend] Class {label}: generated {len(new_texts)} (total: {len(texts)})")

        print(f"[extend] Class {label} done: {len(texts)} descriptions")


if __name__ == "__main__":
    main()
