"""
LTGC 流水线 - Step 1: 图像描述生成
读取尾部类图像 → LLaVA 生成描述 → 保存 CSV
"""
import os
import json
import csv
import argparse
import torch
from torchvision import transforms

from config import IMAGENET_DIR, DESCRIPTIONS_DIR
from data_loader import ImageNetLTDataLoader
from data_txt.imagenet_label_mapping import get_readable_name
from vision_llm import describe_simple as describe_image
from utils import count_samples


DESCRIPTION_PROMPT = (
    "Describe this {name} in one sentence. Format:\n"
    "'A photo of the class {name}, with [physical features], in [background setting].'\n"
    "Do NOT mention people, hands, or specific objects. Describe ONLY the animal/subject and its surroundings."
)


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 1: Image → Description')
    parser.add_argument('-d', '--data_dir', default=IMAGENET_DIR, help='Dataset root')
    parser.add_argument('-m', '--max_num', default=100, type=int, help='Tail class threshold')
    parser.add_argument('-f', '--class_number_file',
                        default='data_txt/ImageNet_LT/imagenetlt_class_count.txt',
                        help='Class count file')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.existing_description_path), exist_ok=True)

    loader = ImageNetLTDataLoader(
        data_dir=args.data_dir,
        split='train',
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    if not os.path.exists(args.class_number_file):
        count_samples(loader)
        with open(args.class_number_file, 'r') as f:
            class_counts = json.load(f)
    else:
        with open(args.class_number_file, 'r') as f:
            class_counts = json.load(f)

    data_to_write = []
    total = len(loader.dataset)
    tail_count = 0
    processed = 0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for pack in loader:
        data, target, index = pack
        cls_id = int(target)

        if class_counts.get(str(cls_id), 0) < args.max_num:
            tail_count += 1
            real_name = get_readable_name(cls_id).split(", ")[0]
            prompt = DESCRIPTION_PROMPT.format(name=real_name)

            data = data * std + mean
            description = describe_image(data, prompt)

            if description:
                data_to_write.append((cls_id, description))

                if len(data_to_write) >= 10:
                    with open(args.existing_description_path, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    print(f"[describe] Wrote {len(data_to_write)} (total: {processed + 1}/{total}, tail: {tail_count})")
                    data_to_write = []

        processed += 1
        if processed % 100 == 0:
            print(f"[describe] Progress: {processed}/{total}")

    if data_to_write:
        with open(args.existing_description_path, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    print(f"[describe] Done. Total: {processed}, Tail: {tail_count}, Output: {args.existing_description_path}")


if __name__ == "__main__":
    main()
