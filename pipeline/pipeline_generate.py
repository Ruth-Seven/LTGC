"""
LTGC 流水线 - Step 3: 图像生成
读取扩展描述 → SD 生成图像 → CLIP 分数展示 + 交互确认 → 保存
"""
import os
import sys
import shutil
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR, DATA_DIR, DESCRIPTION_EXAMPLE_DIR
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
    parser.add_argument('-i','--interactive', action='store_true',
                        help='交互模式：展示 CLIP 分数并让用户确认图像是否合格')
    parser.add_argument('-m', '--md', default=None, nargs='?', const=DESCRIPTION_EXAMPLE_DIR,
                        help='Markdown 示例记录模式：记录 class, description, image, clip score')
    return parser.parse_args()

def save_generation_markdown(records, output_dir):
    """将生成的图像示例保存为 Markdown 文件

    Args:
        records: list of (class_name, description, img_path, clip_score)
        output_dir: 输出目录
    """
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


def main():
    args = parse_args()

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = df.groupby('label')['text'].apply(list).to_dict()

    md_records = []

    total = len(grouped)
    for label_idx, (label, texts) in enumerate(grouped.items()):
        class_name = get_readable_name(int(label)).split(", ")[0]
        dir_path = os.path.join(args.data_dir, 'gen_train', str(label))
        os.makedirs(dir_path, exist_ok=True)

        print(f"[generate] Class {label} ({label_idx + 1}/{total}), {len(texts)} descriptions")

        for text_i, text in enumerate(texts):
            saved_path = os.path.join(dir_path, f"{label}_{text_i}.JPEG")

            accepted = False
            for attempt in range(args.max_rounds):
                img_path = generate(text, saved_path)
                if img_path is None:
                    continue

                clip_score = score(img_path, f"A photo of a {class_name.lower()}")
                print(f"[generate] Class {label} ({label_idx + 1}/{total}): Attempt {attempt + 1}/{args.max_rounds}, Score: {clip_score:.4f}")

                clip_pass = clip_score >= args.thresh

             
                if args.md is not None:
                    if clip_pass:
                        print(f"[generate] Score {clip_score:.4f} >= {args.thresh}, accepted (md mode)")
                        md_records.append((class_name, text, img_path, clip_score))
                        accepted = True
                        break
                    else:
                        print(f"[generate] Score {clip_score:.4f} < {args.thresh}")
                else:
                    if clip_pass:
                        print(f"[generate] Score {clip_score:.4f} >= {args.thresh}, accepted")
                        accepted = True
                        break
                    else:
                        print(f"[generate] Score {clip_score:.4f} < {args.thresh}")

            if not accepted:
                print(f"[generate] 所有尝试均未通过，跳过该描述")

    if md_records:
        save_generation_markdown(md_records, args.md)
    print(f"[generate] Done. {total} classes processed.")


if __name__ == "__main__":
    main()
