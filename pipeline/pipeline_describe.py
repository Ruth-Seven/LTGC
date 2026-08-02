"""LTGC Step 1: sample real images per class and generate one description each."""
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import ImageNetLTDataset
from utils.model.LTGC.model.vision_lmm import describe_image_batch, set_backend
from utils import atomic_json_dump, load_class_semantics, load_prompts, parse_semantic_label, validate_description


MAX_IMAGE_EDGE = 1024
LEADING_SEMANTIC_LABEL_RE = re.compile(
    r"^(?P<prefix>\s*A photo of\s+(?:an?\s+)?)"
    r"(?P<label>[^.\n]*?\([^()\n]+\))"
    r"(?P<suffix>.*)$",
    re.IGNORECASE,
)


def _resize_for_vlm(image, max_edge=MAX_IMAGE_EDGE):
    """Downscale oversized images while preserving aspect ratio; never upscale."""
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return image
    scale = max_edge / longest_edge
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(resized_size, Image.Resampling.LANCZOS)


def _replace_leading_semantic_label(description, display):
    """Replace a generated leading semantic label and revalidate the description."""
    if not description or not isinstance(description, str):
        return ""
    match = LEADING_SEMANTIC_LABEL_RE.match(description.strip().strip("'\""))
    if not match:
        return ""
    normalized = f"{match.group('prefix')}{display}{match.group('suffix')}"
    return normalized if validate_description(normalized, display) else ""


def parse_args():
    parser = argparse.ArgumentParser(description="LTGC Step 1: Image -> Description")
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-exi", "--existing-description-path", required=True)
    parser.add_argument("--class-mapping", required=True)
    parser.add_argument("--descriptions-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--vlm-backend", choices=["llava", "qwen2vl", "qwen3vl"], default="qwen3vl"
    )
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--examples-dir")
    parser.add_argument("--log-dir", default="/tmp")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    # Kept for compatible experiment invocations; class selection is mapping-driven.
    parser.add_argument("-m", "--tail-num-threshold", type=int, default=None)
    parser.add_argument("-f", "--class-number-file", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("-t", "--test", action="store_true")
    return parser.parse_args()


def _select_images(dataset, required_labels, limit, seed):
    grouped = defaultdict(list)
    for path, label in zip(dataset.img_paths, dataset.labels):
        if str(label) in required_labels:
            grouped[int(label)].append(path)
    selected = {}
    for label in sorted(map(int, required_labels)):
        paths = grouped.get(label, [])
        count = min(len(paths), limit)
        selected[str(label)] = random.Random(seed + label).sample(paths, count)
    return selected


def _signature(selected, mapping, args):
    payload = {
        "selected": selected,
        "mapping": mapping,
        "seed": args.seed,
        "limit": args.descriptions_per_class,
        "prompt_file": os.path.abspath(args.prompt_file),
        "backend": args.vlm_backend,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_progress(progress_dir):
    complete = {}
    if not os.path.isdir(progress_dir):
        return complete
    for name in sorted(os.listdir(progress_dir)):
        if not name.startswith("success_gpu") or not name.endswith(".csv"):
            continue
        with open(os.path.join(progress_dir, name), newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    complete[(row[0], row[1])] = row[2]
    return complete


def _worker(rank, chunks, args_dict, mapping, prompt, completed):
    torch.cuda.set_device(rank)
    set_backend(args_dict["vlm_backend"])
    transform = transforms.ToTensor()
    progress_dir = args_dict["progress_dir"]
    part_path = os.path.join(progress_dir, f"success_gpu{rank}.csv")
    failed_path = os.path.join(progress_dir, f"failed_gpu{rank}.json")
    failures = {}
    with open(part_path, "a", newline="", encoding="utf-8") as part:
        writer = csv.writer(part)
        for label, paths in chunks[rank]:
            display = mapping[str(label)]
            name, category, _ = parse_semantic_label(display)
            pending = [path for path in paths if (str(label), path) not in completed]
            for start in range(0, len(pending), args_dict["batch_size"]):
                batch_paths = pending[start:start + args_dict["batch_size"]]
                results = [""] * len(batch_paths)
                rejected = [[] for _ in batch_paths]
                for _ in range(args_dict["max_retries"]):
                    indices = [i for i, value in enumerate(results) if not value]
                    if not indices:
                        break
                    images = []
                    for index in indices:
                        with Image.open(batch_paths[index]) as image:
                            image = _resize_for_vlm(image.convert("RGB"))
                            images.append(transform(image))
                    prompts = [prompt.format(name=name, category=category)] * len(images)
                    responses = describe_image_batch(images, prompts, max_retries=1)
                    for index, response in zip(indices, responses):
                        value = response.strip().strip("'\"")
                        if validate_description(value, display):
                            results[index] = value
                        elif value:
                            rejected[index].append(value)
                for index, attempts in enumerate(rejected):
                    if results[index] or not attempts:
                        continue
                    results[index] = _replace_leading_semantic_label(
                        attempts[-1], display
                    )
                for path, value, attempts in zip(batch_paths, results, rejected):
                    if value:
                        writer.writerow((label, path, value))
                        part.flush()
                    else:
                        failures[f"{label}:{path}"] = attempts
    atomic_json_dump(failures, failed_path)


def _write_examples(path, records, mapping):
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    grouped = defaultdict(list)
    for label, image_path, description in records:
        grouped[label].append((image_path, description))
    for label, rows in grouped.items():
        safe = mapping[label].replace("/", "_").replace(" ", "_")
        with open(os.path.join(path, f"{label}_{safe}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Class {label}: {mapping[label]}\n\n")
            for index, (image_path, description) in enumerate(rows, 1):
                f.write(f"## Image {index}\n\n**Source:** `{image_path}`\n\n**Description:** {description}\n\n")


def main():
    args = parse_args()
    if args.descriptions_per_class <= 0 or args.max_retries <= 0:
        raise ValueError("descriptions-per-class and max-retries must be positive")
    prompts = load_prompts(args.prompt_file)
    prompt = prompts.get("describe", {}).get("vlm_prompt")
    if not prompt:
        raise ValueError("prompt file is missing describe.vlm_prompt")
    mapping = load_class_semantics(args.class_mapping)
    dataset = ImageNetLTDataset(args.data_dir, split="train")
    selected = _select_images(dataset, mapping, args.descriptions_per_class, args.seed)
    empty = [label for label, paths in selected.items() if not paths]
    if empty:
        raise RuntimeError(f"Step1 classes have no source images: {empty}")
    progress_dir = args.existing_description_path + ".progress"
    args.progress_dir = progress_dir
    os.makedirs(progress_dir, exist_ok=True)
    signature = _signature(selected, mapping, args)
    signature_path = os.path.join(progress_dir, "signature.json")
    old_signature = None
    if os.path.exists(signature_path):
        with open(signature_path, encoding="utf-8") as f:
            old_signature = json.load(f).get("signature")
    if args.force or not args.resume:
        import shutil
        shutil.rmtree(progress_dir)
        os.makedirs(progress_dir)
        old_signature = None
    if old_signature and old_signature != signature:
        raise RuntimeError("Step1 resume signature mismatch; use --force to restart")
    atomic_json_dump({"signature": signature, "selected": selected}, signature_path)
    completed = _load_progress(progress_dir)
    items = [(int(label), paths) for label, paths in sorted(selected.items(), key=lambda x: int(x[0]))]
    world_size = max(1, args.num_gpus)
    chunks = [[] for _ in range(world_size)]
    for index, item in enumerate(items):
        chunks[index % world_size].append(item)
    if world_size == 1:
        _worker(0, chunks, vars(args), mapping, prompt, completed)
    else:
        mp.spawn(_worker, args=(chunks, vars(args), mapping, prompt, completed), nprocs=world_size, join=True)
    complete = _load_progress(progress_dir)
    expected = {(label, path) for label, paths in selected.items() for path in paths}
    missing = sorted(expected - set(complete))
    if missing:
        atomic_json_dump({"missing": missing}, args.existing_description_path + ".failed.json")
        raise RuntimeError(f"Step1 incomplete: {len(missing)} selected images have no valid description")
    records = [(label, path, complete[(label, path)]) for label, path in sorted(expected, key=lambda x: (int(x[0]), x[1]))]
    tmp = args.existing_description_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows((label, description) for label, _, description in records)
    os.replace(tmp, args.existing_description_path)
    failed_report = args.existing_description_path + ".failed.json"
    if os.path.exists(failed_report):
        os.remove(failed_report)
    _write_examples(args.examples_dir, records, mapping)
    print(f"[describe] wrote {len(records)} descriptions to {args.existing_description_path}")


if __name__ == "__main__":
    main()
