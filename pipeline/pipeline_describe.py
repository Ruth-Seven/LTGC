"""
LTGC 流水线 - Step 1: 图像描述生成
读取尾部类图像 → LLaVA 生成描述 → 保存 CSV
"""
import os
import sys
import json
import csv
import argparse
import time
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGENET_DIR, DESCRIPTIONS_DIR, DESCRIPTION_EXAMPLE_DIR, CLASS_COUNT_FILE
from data.data_loader import ImageNetLTDataLoader
from data_txt.imagenet_label_mapping import get_readable_name
from model.vision_lmm import describe_image
from utils import count_samples
from tqdm import tqdm


text_prompt = (
            "Please use the Template to briefly describe the image of the class {name} in only one sentence. Template:\n"
            "'A photo of the class {name}, with [distinctive features], [specific scenes].'\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 1: Image → Description')
    parser.add_argument('-d', '--data_dir', default=IMAGENET_DIR, help='Dataset root')
    parser.add_argument('-m', '--tail_num_threshold', default=50, type=int, help='Tail class threshold')
    parser.add_argument('-f', '--class_number_file',
                        default=CLASS_COUNT_FILE,
                        help='Class count file')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Output CSV path')
    parser.add_argument('--examples-dir',
                        default=DESCRIPTION_EXAMPLE_DIR,
                        help='Directory to save example markdown with images')
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode with limited examples')
    return parser.parse_args()

def describe_example_markdown(examples, output_dir):
    """将尾部类描述示例保存为 Markdown 文件（含图片）
    
    Args:
        examples: list of (cls_id, img_tensor, description, class_name) tuples
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "tail_class_description_examples.md")
    to_pil = transforms.ToPILImage()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tail Class Description Examples\n\n")
        f.write(f"Total examples: {len(examples)}\n\n")

        for i, (cls_id, img_tensor, description, class_name) in enumerate(tqdm(examples, desc="[describe] Saving examples")):
            img_filename = f"example_{cls_id}_{i}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            img_tensor_cpu = img_tensor.squeeze(0).cpu().clamp(0, 1)
            to_pil(img_tensor_cpu).save(img_path)

            f.write(f"## Example {i+1}: Class {cls_id} - {class_name}\n\n")
            f.write(f"![Image]({img_filename})\n\n")
            f.write(f"**Description:** {description}\n\n")
            f.write("---\n\n")

    print(f"[describe_example_markdown] Examples saved to {md_path}")



def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.existing_description_path), exist_ok=True)
    print("[start] Dataloading....")
    loader = ImageNetLTDataLoader(
        data_dir=args.data_dir,
        split='train',
        batch_size=1,
        shuffle=False,
        num_workers=16,
    )

    if not os.path.exists(args.class_number_file):
        count_samples(loader, output_path=args.class_number_file)
        with open(args.class_number_file, 'r') as f:
            class_counts = json.load(f)
    else:
        with open(args.class_number_file, 'r') as f:
            class_counts = json.load(f)

    data_to_write = []
    total = len(loader.dataset)
    tail_count = 0
    processed = 0
    examples = []
    example_classes = set()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    print("[Describe] Starting image description generation for tail classes...")
    description_file = args.existing_description_path
    if os.path.exists(description_file):
        print(f"[describe] Backuping existing description file: {description_file}")
        os.rename(description_file, description_file + "_" + time.strftime("%Y%m%d-%H%M%S"))
    pbar = tqdm(total=total, desc="[describe] Processing", unit="img")
    for pack in loader:
        data, target, index = pack
        cls_id = int(target)

        if class_counts.get(str(cls_id), 0) < args.tail_num_threshold:
            tail_count += 1
            real_name = get_readable_name(cls_id).split(", ")[0]
            prompt = text_prompt.format(name=real_name)

            data = data * std + mean
            description = describe_image(data, prompt)

            if description:
                data_to_write.append((cls_id, description))

                if cls_id not in example_classes:
                    example_classes.add(cls_id)
                    examples.append((cls_id, data.clone(), description, real_name))

                if len(data_to_write) >= 10:
                    with open(description_file, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    data_to_write = []

        processed += 1
        pbar.set_postfix(tail=tail_count, batch=processed, original_cls_num=class_counts.get(str(cls_id), 0))
        pbar.update(1)
        
        if args.test and len(example_classes) > 29:
            break
    pbar.close()

    if data_to_write:
        with open(args.existing_description_path, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    if examples:
        describe_example_markdown(examples, args.examples_dir)

    print(f"[describe] Done. Total: {processed}, Tail: {tail_count}, Output: {args.existing_description_path}")


if __name__ == "__main__":
    main()
