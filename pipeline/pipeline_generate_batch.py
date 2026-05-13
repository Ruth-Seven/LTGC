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
from model.clip_score import score, score_batch
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

texts = []
image_paths = []
class_names = []
batch_size = 100 # 每批处理 100 个描述

fail_texts = []
fail_image_paths = []
fail_class_names = []

def batch_process(batch_texts, batch_image_paths):
    print(f"[batch_process] Processing batch of size {len(batch_texts)}...")
    scores = score_batch(batch_image_paths, batch_texts)
    for p, t, s in zip(batch_image_paths, batch_texts, scores):
        print(f" {t} \n  CLIP Score: {s:.4f} : {p} ")
    return zip(batch_image_paths, batch_texts, scores)

def clear_batch_resource():
    global texts, image_paths
    texts = []
    image_paths = []

def add_fail_batch(img_path, text):
    global fail_texts, fail_image_paths
    fail_texts.append(text) 
    fail_image_paths.append(img_path)


def add_cal_batch_resource(img_path, text):
    global texts, image_paths
    texts.append(text)
    image_paths.append(img_path)
    if len(texts) >= batch_size:
        zip = batch_process(texts, image_paths)
        clear_batch_resource()
        return zip
    return None

def cal_last_batch_resource():
    global texts, image_paths
    if texts and image_paths:
        zip = batch_process(texts, image_paths)
        clear_batch_resource()
        return zip
    return None

def cal_fail_batch_resource():
    global fail_texts, fail_image_paths
    global texts, image_paths
    texts = fail_texts
    image_paths = fail_image_paths
    fail_texts = []
    fail_image_paths = []
    return cal_last_batch_resource()


def store_cal_res(zip_res, thresh, md_records, class_name):
    if zip_res is None:
        return
    for img_path, text, clip_score in zip_res:
    clip_pass = clip_score >= args.thresh
        if clip_pass:
            print(f"[generate] Score {clip_score:.4f} >= {args.thresh}, accepted (md mode)")
            if  md_records is not None:
                md_records.append((class_name, text, img_path, clip_score))
        else:
            print(f"[generate] Score {clip_score:.4f} < {args.thresh}, denied")
            add_fail_batch(img_path, text)

def main():
    args = parse_args()

    df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
    grouped = df.groupby('label')['text'].apply(list).to_dict()

    md_records = None if not args.md else []

    total = len(grouped)
    fail_retry = 3
    for label_idx, (label, texts) in enumerate(grouped.items()):
        
        class_name = get_readable_name(int(label)).split(", ")[0]
        dir_path = os.path.join(args.data_dir, 'gen_train', str(label))
        os.makedirs(dir_path, exist_ok=True)

        for text_i, text in enumerate(texts):
            saved_path = os.path.join(dir_path, f"{label}_{text_i}.JPEG")

            img_path = generate(text, saved_path)
            if img_path is None:
                raise RuntimeError(f"Failed to generate image for label {label} (attempt {text_i + 1}/{len(texts)}) with prompt: {text}")

            zip_res = add_cal_batch_resource(img_path, f"A photo of a {class_name.lower()}")
            store_cal_res(zip_res, args.thresh, md_records, class_name)
            
    for retry in range(fail_retry):

        store_cal_res(cal_fail_batch_resource(), args.thresh, md_records, class_name)


    if md_records is not None:
        save_generation_markdown(md_records, args.md)
    print(f"[generate] Done. {total} classes processed.")


if __name__ == "__main__":
    main()
