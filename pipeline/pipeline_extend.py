"""
LTGC 流水线 - Step 2: 描述扩展
读取已有描述 → 本地 LLM 生成多样化变体 → 保存 CSV
"""
import os
import sys
import csv
import argparse
import logging
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DESCRIPTIONS_DIR
from model.text_llm import extend_descriptions, reflection_descriptions, _unload_model
from data_txt.imagenet_label_mapping import get_readable_name as _imagenet_class_name

_CLASS_MAP = None


def _get_class_name(label):
    global _CLASS_MAP
    if _CLASS_MAP is not None:
        return str(_CLASS_MAP.get(str(label), label))
    return _imagenet_class_name(int(label)).split(", ")[0]

extension_prompt = f"""Besides these descriptions mentioned above, please use the Template 1 to list exactly {{number}} other possible [distinctive features] and [specific scenes].
Template1: A photo of the class {{class_name}}, [with distinctive features] [in specific scenes]. 
List the selected sentences numbered from 1 to {{number}}, one per line. Do not output more than {{number}} descriptions."""

reflection_prompt = f"""From the provided description list, please select exactly {{number}} unique sentences for the class '{{class_name}}'. 
Each sentence must describe a different [distinctive feature] (e.g., texture, shape) or a [specific scene] (e.g., lighting, environment) to ensure diversity. Avoid near-duplicates.
List the selected sentences numbered from 1 to {{number}}, one per line. Do not output more than {{number}} descriptions."""


def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger


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
    parser.add_argument('--log_dir', type=str, default="/tmp",
                        help='Log file directory')
    parser.add_argument('--class-mapping', type=str, default=None,
                        help='JSON class name mapping file (e.g. {"0":"crazing"})')
    return parser.parse_args()


def main():
    args = parse_args()
    global _CLASS_MAP
    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping) as f:
            _CLASS_MAP = json.load(f)
        print(f"[extend] Loaded class mapping: {len(_CLASS_MAP)} entries")

    os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("extend", os.path.join(args.log_dir, "pipeline_extend.log"))

    df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
    grouped = df.groupby('label')['text'].apply(list).to_dict()
    extension_descriptions = {}
    total = len(grouped)
    for idx, (label, texts) in enumerate(grouped.items()):
        class_name = _get_class_name(label)
        logger.info("Class %s %s (%d/%d): %d existing", label, class_name, idx + 1, total, len(texts))
        extension_descriptions[label] = []
        while len(texts) + len(extension_descriptions[label]) < args.max_generate_num*2:
            new_texts = extend_descriptions(texts, prompt=extension_prompt.format(number=args.max_generate_num*2 - len(texts),class_name=class_name), number=args.max_generate_num*2 - len(texts))
            logger.info("Class %s %s: generated %d new descriptions", label, class_name, len(new_texts))
            for new_text in new_texts:
                logger.info("  - %s", new_text)
            if not new_texts:
                logger.info("No new descriptions, stopping")
                break
            logger.info("Class %s %s: before reflection %d descriptions", label, class_name, len(extension_descriptions[label]) + len(new_texts))
            extension_descriptions[label].extend(new_texts)
            extension_descriptions[label] = list(set(extension_descriptions[label]))
        logger.info("Class %s %s: after deduplication %d new descriptions", label, class_name, len(extension_descriptions[label]))
        extension_descriptions[label] = reflection_descriptions(extension_descriptions[label], prompt=reflection_prompt.format(number=args.max_generate_num,class_name=class_name), number=args.max_generate_num)
        logger.info("Class %s %s: after reflection %d descriptions", label, class_name, len(extension_descriptions[label]))
        for des in extension_descriptions[label]:
            logger.info("  - %s", des)
        logger.info("Class %s %s: generated %d (total: %d)", label, class_name, len(extension_descriptions[label]), len(texts))

    with open(args.extended_description_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for label, deses in extension_descriptions.items():
            for des in deses:
                writer.writerow([label, des])

    _unload_model()


if __name__ == "__main__":
    main()
