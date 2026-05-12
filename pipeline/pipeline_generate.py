"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 → SD 生成图像 → CLIP 分数展示 + 交互确认 → 保存
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


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text → Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'extended_description.csv'),
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, help='Output root')
    parser.add_argument('-t', '--thresh', default=0.25, type=float, help='CLIP score threshold')
    parser.add_argument('-r', '--max_rounds', default=3, type=int, help='Max retry rounds')
    parser.add_argument('--interactive', action='store_true',
                        help='交互模式：展示 CLIP 分数并让用户确认图像是否合格')
    return parser.parse_args()


def ask_user(img_path, clip_score, class_name):
    """交互式确认图像是否合格"""
    print(f"\n{'='*50}")
    print(f"  类别: {class_name}")
    print(f"  CLIP 分数: {clip_score:.4f}")
    print(f"  图像路径: {img_path}")
    print(f"{'='*50}")
    while True:
        answer = input("  该图像是否合格？(y/n): ").strip().lower()
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print("  请输入 y 或 n")


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

            accepted = False
            for attempt in range(args.max_rounds):
                img_path = generate(text, saved_path)
                if img_path is None:
                    continue

                clip_score = score(img_path, f"A photo of a {class_name.lower()}")
                print(f"[generate] Attempt {attempt + 1}/{args.max_rounds}, Score: {clip_score:.4f}")

                clip_pass = clip_score >= args.thresh

                if args.interactive:
                    user_pass = ask_user(img_path, clip_score, class_name)
                    if user_pass:
                        print(f"[generate] 用户确认合格 ✓")
                        accepted = True
                        break
                    else:
                        print(f"[generate] 用户认为不合格，重试...")
                else:
                    if clip_pass:
                        print(f"[generate] Score {clip_score:.4f} >= {args.thresh}, accepted")
                        accepted = True
                        break
                    else:
                        print(f"[generate] Score {clip_score:.4f} < {args.thresh}")

            if not accepted:
                print(f"[generate] 所有尝试均未通过，跳过该描述")

    print(f"[generate] Done. {total} classes processed.")


if __name__ == "__main__":
    main()
