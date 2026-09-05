"""LTGC Step 0: infer a canonical ``name (category)`` label for every class."""

import argparse
import os
import random
import sys
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import ImageNetLTDataset
from data_txt.imagenet_label_mapping import get_readable_name

from model.vision_lmm import describe_image_group, set_backend
from utils import atomic_json_dump, load_prompts, parse_semantic_label


def parse_args():
    parser = argparse.ArgumentParser(description="LTGC Step 0: class semantic disambiguation")
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--reference-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--vlm-backend", choices=["llava", "qwen2vl", "qwen3vl"], default="qwen3vl")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--log-dir", default="/tmp")
    return parser.parse_args()


def _select_references(dataset, count, seed):
    grouped = defaultdict(list)
    for path, label in zip(dataset.img_paths, dataset.labels):
        grouped[int(label)].append(path)
    selected = {}
    for label, paths in sorted(grouped.items()):
        if len(paths) < count:
            raise RuntimeError(f"class {label} has {len(paths)} images, needs {count}")
        selected[label] = random.Random(seed + label).sample(paths, count)
    return selected


def _build_semantic_label(source_name, category):
    """Combine a category-only VLM response with the canonical source name."""
    category = category.strip().strip("'\"")
    if not category:
        raise ValueError("category must not be empty")
    if "\n" in category or "->" in category:
        raise ValueError("response must contain one category only")
    if category.casefold() == source_name.casefold():
        raise ValueError("category must not same as class name")
    _, _, display = parse_semantic_label(f"{source_name} ({category})")
    return display


def _worker(rank, chunks, args_dict, prompt):
    torch.cuda.set_device(rank)
    set_backend(args_dict["vlm_backend"])
    transform = transforms.ToTensor()
    result = {}
    failures = {}
    for label, paths in chunks[rank]:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        source_name = get_readable_name(label).split(", ")[0]
        answer = ""
        reason = "empty response"
        for _ in range(args_dict["max_retries"]):
            answer = (
                describe_image_group(images, prompt.format(class_name=source_name), max_retries=1)
                .strip()
                .strip("'\"")
            )
            try:
                result[str(label)] = _build_semantic_label(source_name, answer)
                break
            except ValueError as exc:
                reason = str(exc)
        else:
            failures[str(label)] = {"response": answer, "reason": reason}
    atomic_json_dump(result, f"{args_dict['output']}.part{rank}")
    atomic_json_dump(failures, f"{args_dict['output']}.fail{rank}")


def main():
    args = parse_args()
    if args.reference_images <= 0 or args.max_retries <= 0:
        raise ValueError("reference-images and max-retries must be positive")
    prompts = load_prompts(args.prompt_file)
    prompt = prompts.get("disambiguate", {}).get("prompt")
    if not prompt:
        raise ValueError("prompt file is missing disambiguate.prompt")
    dataset = ImageNetLTDataset(args.data_dir, split="train")
    selected = _select_references(dataset, args.reference_images, args.seed)
    world_size = max(1, args.num_gpus)
    chunks = [[] for _ in range(world_size)]
    for index, item in enumerate(sorted(selected.items())):
        chunks[index % world_size].append(item)
    args_dict = vars(args)
    if world_size == 1:
        _worker(0, chunks, args_dict, prompt)
    else:
        mp.spawn(_worker, args=(chunks, args_dict, prompt), nprocs=world_size, join=True)
    merged = {}
    failures = {}
    for rank in range(world_size):
        for suffix, target in (("part", merged), ("fail", failures)):
            path = f"{args.output}.{suffix}{rank}"
            import json

            with open(path, encoding="utf-8") as f:
                target.update(json.load(f))
            os.remove(path)
    missing = sorted(set(map(str, selected)) - set(merged))
    if failures or missing:
        atomic_json_dump({"failures": failures, "missing": missing}, args.output + ".failed.json")
        raise RuntimeError(f"Step0 incomplete: {len(missing)} classes failed")
    atomic_json_dump(merged, args.output)
    failed_report = args.output + ".failed.json"
    if os.path.exists(failed_report):
        os.remove(failed_report)
    print(f"[disambiguate] wrote {len(merged)} class semantics to {args.output}")


if __name__ == "__main__":
    main()
