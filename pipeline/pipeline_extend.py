"""
LTGC 流水线 - Step 2: 描述扩展
读取已有描述 → 本地 LLM 生成多样化变体 → 保存 CSV
"""
import os
import sys
import csv
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR
from model.text_llm import extend_descriptions, refection_descriptions, _unload_model
from data_txt.imagenet_label_mapping import get_readable_name

extension_prompt = f"""Besides these descriptions mentioned above, please use the Template 1 to list at last {{number}} other possible [distinctive features] and [specific scenes].
Template1: A photo of the class {{class_name}}, [with distinctive features] [in specific scenes]. 
List the selected sentences using a dash (-) as the prefix. Each sentence must be on a new line. Do not number them. """


reflection_prompt = f"""From the provided description list, please select {{number}} unique sentences for the class '{{class_name}}'. 
Each sentence must describe a different [distinctive feature] (e.g., texture, shape) or a [specific scene] (e.g., lighting, environment) to ensure diversity. Avoid near-duplicates.
List the selected sentences using a dash (-) as the prefix. Each sentence must be on a new line. Do not number them."""

def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 2: Description → Extended Descriptions')
    parser.add_argument('-exi', '--existing_description_path',
                        default=os.path.join(DESCRIPTIONS_DIR, 'existing_description_list.csv'),
                        help='Input descriptions CSV')
    parser.add_argument('-m', '--max_generate_num', default=50, type=int,
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
    extension_descriptions = {}
    total = len(grouped)
    pbar = tqdm(total=total, desc="[extend] Processing classes", unit="class")
    for idx, (label, texts) in enumerate(grouped.items()):
        class_name = get_readable_name(int(label)).split(", ")[0]
        print(f"[extend] Class {label} {class_name} ({idx + 1}/{total}): {len(texts)} existing")
        extension_descriptions[label] = []
        while len(texts) + len(extension_descriptions[label]) < args.max_generate_num*2:
            new_texts = extend_descriptions(texts, prompt=extension_prompt.format(number=args.max_generate_num*2 - len(texts),class_name=class_name))
            print(f"[extend] Class {label} {class_name}: generated {len(new_texts)} new descriptions")
            for new_text in new_texts:
                print(f"  - {new_text}")
            if not new_texts:
                print(f"[extend] No new descriptions, stopping")
                break
            print(f"[extend] Class {label} {class_name}: before reflection {len(extension_descriptions[label]) + len(new_texts)} descriptions")
            extension_descriptions[label].extend(new_texts)
            extension_descriptions[label] = list(set(extension_descriptions[label])) #去重
        print(f"[extend] Class {label} {class_name}: after deduplication {len(extension_descriptions[label])} new descriptions")
        extension_descriptions[label] = refection_descriptions(extension_descriptions[label], prompt=reflection_prompt.format(number=args.max_generate_num,class_name=class_name))
        print(f"[extend] Class {label} {class_name}: after reflection {len(extension_descriptions[label])} descriptions")
        for des in extension_descriptions[label]:
            print(f"  - {des}")
        print(f"[extend] Class {label} {class_name}: generated {len(extension_descriptions[label])} (total: {len(texts)})")
        pbar.update(1)

    with open(args.extended_description_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for label, deses in extension_descriptions.items():
            for des in deses:
                writer.writerow([label, des])


    _unload_model()

if __name__ == "__main__":
    main()
