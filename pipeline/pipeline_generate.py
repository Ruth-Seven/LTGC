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

from config import DESCRIPTIONS_DIR, DATA_DIR, GENERATION_EXAMPLE_DIR, EXTENDED_DESCRIPTION_PATH,CLIP_MAX_TOKENS
from model.clip_score import score, score_batch
from model.image_gen import generate, generate_batch, unload_sd
from data_txt.imagenet_label_mapping import get_readable_name


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 3: Text → Image')
    parser.add_argument('-ext', '--extended_description_path',
                        default=EXTENDED_DESCRIPTION_PATH,
                        help='Extended descriptions CSV')
    parser.add_argument('-d', '--data_dir', default=DATA_DIR, help='Output root')
    parser.add_argument('-t', '--thresh', default=0.28, type=float, help='CLIP score threshold')
    parser.add_argument('-r', '--max_rounds', default=5, type=int, help='Max retry rounds')
    parser.add_argument('-m', '--md', default=None, nargs='?', const=GENERATION_EXAMPLE_DIR,
                        help='Markdown 示例记录模式：记录 class, description, image, clip score')
    parser.add_argument('-o', '--onepath', action='store_true', help='让所有图片保存的同一个地址方便查看')
    parser.add_argument('-b', '--batch', default=10, type=int, help='批量生成图片数量')

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
        texts = [t[:CLIP_MAX_TOKENS] for t in texts]
        print(f"[generate] Class {label} ({label_idx + 1}/{total}), {len(texts)} descriptions")

        # ── onepath mode: 所有图存同个路径，逐张生成 ──
        if args.onepath:
            for text_i, text in enumerate(texts):
                saved_path = os.path.join(args.data_dir, 'gen_train-onepath.JPEG')
                accepted = False
                for attempt in range(args.max_rounds):
                    img_path = generate(text, saved_path)
                    if img_path is None:
                        continue
                    clip_score = score(img_path, text)
                    print(f"[generate] Class {label} ({label_idx + 1}/{total}): Attempt {attempt + 1}/{args.max_rounds}, Score: {clip_score:.4f} Class: {class_name}")
                    if clip_score >= args.thresh:
                        print(f"[generate] accepted")
                        if args.md is not None:
                            md_records.append((class_name, text, img_path, clip_score))
                        accepted = True
                        break
                    else:
                        print(f"[generate] Score {clip_score:.4f} < {args.thresh}")
                if not accepted:
                    print(f"[generate] 所有尝试均未通过，跳过该描述")
            continue

        # ── batch 模式（按 args.batch 分块）──
        n = len(texts)
        save_paths = [os.path.join(dir_path, f"{label}", f"{label}_{i}.JPEG") for i in range(n)]
        accepted = [False] * n
        bs = args.batch

        for chunk_start in range(0, n, bs):
            chunk_end = min(chunk_start + bs, n)
            chunk_ids = list(range(chunk_start, chunk_end))
            print(f"[generate]  Batch chunk [{chunk_start}:{chunk_end}]")

            for attempt in range(args.max_rounds):
                pending = [i for i in chunk_ids if not accepted[i]]
                if not pending:
                    break

                batch_prompts = [texts[i] for i in pending]
                batch_paths = [save_paths[i] for i in pending]
                img_paths = generate_batch(batch_prompts, batch_paths)

                valid = [(i, p) for i, p in zip(pending, img_paths) if p is not None]
                if not valid:
                    break

                v_idx, v_paths = zip(*valid)
                v_texts = [texts[i] for i in v_idx]

                clip_scores = score_batch(list(v_paths), list(v_texts))

                for idx, s in zip(v_idx, clip_scores):
                    print(f"[generate] Class {label} ({label_idx + 1}/{total}): Attempt {attempt + 1}/{args.max_rounds}, Score: {s:.4f} Class: {class_name}")
                    if s >= args.thresh:
                        print(f"[generate] accepted")
                        if args.md is not None:
                            md_records.append((class_name, texts[idx], save_paths[idx], s))
                        accepted[idx] = True
                    else:
                        print(f"[generate] Score {s:.4f} < {args.thresh}")

                if all(accepted[i] for i in chunk_ids):
                    break

        failed = sum(1 for a in accepted if not a)
        if failed:
            print(f"[generate] {failed}/{n} 张失败")

        unload_sd()

    if md_records:
        save_generation_markdown(md_records, args.md)
    print(f"[generate] Done. {total} classes processed.")


if __name__ == "__main__":
    main()
