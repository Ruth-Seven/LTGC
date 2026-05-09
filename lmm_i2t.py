import torch
from lt_dataloaders import ImageNetLTDataLoader
from data_txt.imagenet_label_mapping import get_readable_name
from gpt4v import gpt4v_observe
from ultis import sample_counter
import os
import json
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 1: Image Description Generation')
    parser.add_argument('-d', '--data_dir', default='/data/imagenet-lt/torch_image_folder/mnt/volume_sfo3_01/imagenet-lt/ImageDataset', type=str, help='Dataset root directory')
    parser.add_argument('-m', '--max_num', default=100, type=int, help='Tail class threshold (max samples per class)')
    parser.add_argument('-f', '--class_number_file', default='data_txt/ImageNet_LT/imagenetlt_class_count.txt', type=str, help='Class count file')
    parser.add_argument('-exi', '--existing_description_path', default='/data/descriptions_data/existing_description_list.csv', type=str, help='Output description CSV path')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.existing_description_path), exist_ok=True)

    imagenet_loader = ImageNetLTDataLoader(
        data_dir=args.data_dir,
        split='train',
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    if not os.path.exists(args.class_number_file):
        sample_counter(imagenet_loader)
        with open(args.class_number_file, 'r') as file:
            dict_class_number = json.load(file)
    else:
        with open(args.class_number_file, 'r') as file:
            dict_class_number = json.load(file)

    data_to_write = []
    total_samples = len(imagenet_loader.dataset)
    tail_count = 0
    processed = 0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for epoch, pack in enumerate(imagenet_loader):
        data, target, index = pack
        cls_id = int(target)

        if dict_class_number.get(str(cls_id), 0) < args.max_num:
            tail_count += 1
            real_name = get_readable_name(cls_id).split(", ")[0]
            text_prompt = (
                f"Describe this {real_name} in one sentence. Format:\n"
                f"'A photo of the class {real_name}, with [physical features], in [background setting].'\n"
                f"Do NOT mention people, hands, or specific objects. Describe ONLY the animal/subject and its surroundings."
            )

            data = data * std + mean
            img_description = gpt4v_observe(data, text_prompt)

            if img_description:
                data_to_write.append((cls_id, img_description))

                if len(data_to_write) >= 10:
                    with open(args.existing_description_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(data_to_write)
                    print(f"[lmm_i2t] Wrote {len(data_to_write)} descriptions (total processed: {processed + 1}/{total_samples}, tail: {tail_count})")
                    data_to_write = []

        processed += 1
        if processed % 100 == 0:
            print(f"[lmm_i2t] Progress: {processed}/{total_samples}")

    if data_to_write:
        with open(args.existing_description_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_to_write)
        print(f"[lmm_i2t] Final write: {len(data_to_write)} descriptions")

    print(f"[lmm_i2t] Done. Total: {processed}, Tail classes processed: {tail_count}, Output: {args.existing_description_path}")


if __name__ == "__main__":
    main()
