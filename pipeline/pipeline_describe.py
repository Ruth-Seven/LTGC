"""
LTGC 流水线 - Step 1: 图像描述生成
读取尾部类图像 → LLaVA 生成描述 → 保存 CSV

--num_gpus N：多卡数据并行 (DistributedSampler 自动分片，N worker 各占 1 GPU)
"""
import os
import sys
import json
import csv
import argparse
import time
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGENET_DIR, DESCRIPTIONS_DIR, CLASS_COUNT_FILE
from data.data_loader import ImageNetLTDataset, SUPPORTED_EXTENSIONS
from data_txt.imagenet_label_mapping import get_readable_name
from model.vision_lmm import describe_image_batch, set_backend
from utils import validate_description, cleanup_stale_parts, load_prompts
from collections import defaultdict, Counter


def collate_no_stack(batch):
    """不堆叠 image tensor，保持 list 以支持不同尺寸的图片"""
    images = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch])
    return images, targets, indices


MAX_DESCRIBE_RETRIES = 3


def _describe_with_retry(group_imgs, name, logger, vlm_prompt):
    results = [""] * len(group_imgs)

    for attempt in range(MAX_DESCRIBE_RETRIES):
        pending = [(i, group_imgs[i]) for i, r in enumerate(results) if not r]
        if not pending:
            break

        p_indices, p_imgs = zip(*pending) if pending else ([], [])
        p_indices = list(p_indices)
        p_imgs = list(p_imgs)

        prompts = [vlm_prompt.format(name=name)] * len(p_imgs)
        batch_results = describe_image_batch(p_imgs, prompts)

        for idx, desc in zip(p_indices, batch_results):
            if desc and validate_description(desc, name):
                results[idx] = desc

        still_failed = sum(1 for r in results if not r)
        if still_failed == 0:
            break
        if attempt < MAX_DESCRIBE_RETRIES - 1:
            logger.info("Class %s: attempt %d/%d, %d/%d images passed",
                        name, attempt + 1, MAX_DESCRIBE_RETRIES,
                        len(group_imgs) - still_failed, len(group_imgs))

    final_failed = sum(1 for r in results if not r)
    if final_failed > 0:
        logger.warning("Class %s: %d/%d images failed all %d attempts",
                       name, final_failed, len(group_imgs), MAX_DESCRIBE_RETRIES)

    return results


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
                        default=None,
                        help='Directory to save per-class Markdown examples (skip if not set)')
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode with limited examples')
    parser.add_argument('--log_dir', type=str, default="/tmp",
                        help='Log file directory')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs (DistributedSampler 数据并行)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader workers per GPU (default: 16)')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size for VLM inference (default: 6)')
    parser.add_argument('--vlm-backend', type=str, default='qwen2vl',
                        choices=['llava', 'qwen2vl'],
                        help='VLM backend: llava | qwen2vl (default: qwen2vl)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='Prompt JSON 配置文件（默认使用内置 prompt）')
    return parser.parse_args()


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "[%(name)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger


def _ddp_worker(rank, world_size, args_dict):
    """DDP worker rank/GPU N: DistributedSampler 自动按 rank 分片数据"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args_dict['master_port'])

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

    set_backend(args_dict['vlm_backend'])

    log_dir = args_dict['log_dir']
    logger = setup_logger(f"describe_gpu{rank}", os.path.join(log_dir, f"pipeline_describe_gpu{rank}.log"))
    logger.info("DDP worker rank %d/%d initializing (batch_size=%d)...",
                rank, world_size, args_dict.get('batch_size', 6))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageNetLTDataset(args_dict['data_dir'], split='train', transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    batch_size = args_dict.get('batch_size', 6)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size,
        num_workers=args_dict.get('num_workers', 4), pin_memory=True,
        collate_fn=collate_no_stack,
    )

    with open(args_dict['class_number_file'], 'r') as f:
        class_counts = json.load(f)

    output_base = args_dict['existing_description_path']
    part_path = f"{output_base}.part{rank}"
    # 创建/截断空文件，避免上次残留数据（即使主进程已清过，防御极端情况）
    open(part_path, 'w').close()
    data_to_write = []
    examples_dir = args_dict.get('examples_dir')
    per_class = defaultdict(list) if examples_dir else None
    processed = 0
    tail_count = 0

    threshold = args_dict['tail_num_threshold']
    test_mode = args_dict.get('test', False)

    for data_list, target, index in loader:
        B = len(target)

        groups = defaultdict(list)
        for i in range(B):
            cls_id = int(target[i])
            if class_counts.get(str(cls_id), 0) < threshold:
                name = get_readable_name(cls_id).split(", ")[0]
                groups[(cls_id, name)].append(i)

        for (cls_id, name), indices in groups.items():
            group_imgs = [data_list[i] for i in indices]
            descriptions = _describe_with_retry(group_imgs, name, logger,
                                                args_dict.get('vlm_prompt'))

            tail_count += len(indices)
            for i, desc in zip(indices, descriptions):
                if desc:
                    data_to_write.append((cls_id, desc))

                    if per_class is not None:
                        orig_idx = int(index[i])
                        img_path = dataset.img_paths[orig_idx]
                        per_class[cls_id].append((img_path, desc))

                if len(data_to_write) >= 10:
                    with open(part_path, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    data_to_write = []

        processed += B
        if test_mode and processed >= 1000:
            break

    if data_to_write:
        with open(part_path, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    if per_class:
        tmp_dir = os.path.join(examples_dir, '.tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        for cls_id, records in per_class.items():
            name = get_readable_name(cls_id).split(", ")[0]
            part_md = os.path.join(tmp_dir, f"{cls_id}.part{rank}.md")
            with open(part_md, 'w') as f:
                f.write(f"## Original Descriptions ({len(records)})\n\n")
                for k, (img_path, desc) in enumerate(records, 1):
                    img_rel = f"images/{cls_id}/{os.path.basename(img_path)}"
                    f.write(f"### Image {k}\n\n")
                    f.write(f"![Image {k}]({img_rel})\n\n")
                    f.write(f"**Description:** {desc}\n\n")
        logger.info("DDP rank %d wrote .part md for %d classes", rank, len(per_class))

    logger.info("DDP rank %d done. Processed: %d, Tail: %d", rank, processed, tail_count)

    del loader
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def _merge_parts(output_path, num_gpus, logger):
    logger.info("Merging %d part files into %s", num_gpus, output_path)
    tmp_path = output_path + ".tmp"
    merged_count = 0
    with open(tmp_path, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        for rank in range(num_gpus):
            part_path = f"{output_path}.part{rank}"
            if not os.path.exists(part_path):
                logger.warning("Part file not found: %s", part_path)
                continue
            with open(part_path, 'r') as in_f:
                reader = csv.reader(in_f)
                for row in reader:
                    writer.writerow(row)
                    merged_count += 1
    os.rename(tmp_path, output_path)
    for rank in range(num_gpus):
        part_path = f"{output_path}.part{rank}"
        if os.path.exists(part_path):
            os.remove(part_path)
    logger.info("Merged %d rows into %s", merged_count, output_path)


def _main_single_gpu(args, logger):
    set_backend(args.vlm_backend)

    description_file = args.existing_description_path
    tmp_file = description_file + ".tmp"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    logger.info("Dataloading (single-GPU, batch_size=%d)...", args.batch_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = ImageNetLTDataset(args.data_dir, split='train', transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_no_stack,
    )

    if not os.path.exists(args.class_number_file):
        logger.info("Pre-computing class counts from filesystem...")
        _count_classes_from_fs(args.data_dir, args.class_number_file, logger)
    with open(args.class_number_file, 'r') as f:
        class_counts = json.load(f)

    data_to_write = []
    processed = 0
    tail_count = 0
    examples_dir = args.examples_dir
    per_class = defaultdict(list) if examples_dir else None

    logger.info("Starting image description generation for tail classes...")

    for data_list, target, index in loader:
        B = len(target)

        groups = defaultdict(list)
        for i in range(B):
            cls_id = int(target[i])
            if class_counts.get(str(cls_id), 0) < args.tail_num_threshold:
                name = get_readable_name(cls_id).split(", ")[0]
                groups[(cls_id, name)].append(i)

        for (cls_id, name), indices in groups.items():
            group_imgs = [data_list[i] for i in indices]
            descriptions = _describe_with_retry(group_imgs, name, logger,
                                                args.vlm_prompt)

            tail_count += len(indices)
            for i, desc in zip(indices, descriptions):
                if desc:
                    data_to_write.append((cls_id, desc))

                    if per_class is not None:
                        orig_idx = int(index[i])
                        img_path = dataset.img_paths[orig_idx]
                        per_class[cls_id].append((img_path, desc))

                if len(data_to_write) >= 10:
                    with open(tmp_file, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    data_to_write = []

        processed += B
        if args.test and processed >= 1000:
            break

    if data_to_write:
        with open(tmp_file, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    if per_class:
        _save_single_gpu_examples(per_class, examples_dir, logger)

    os.rename(tmp_file, description_file)
    logger.info("Single-GPU done. Processed: %d, Tail: %d, Output: %s", processed, tail_count, description_file)


def _count_classes_from_fs(data_dir, class_number_file, logger):
    """直接从目录结构统计各类别样本数（不走 DataLoader，极快）

    ImageNet-LT 图片尺寸各异，DataLoader 默认 collate_fn 会 torch.stack
    导致 crash。纯文件系统扫描不需要加载任何图片，0.2s 完成 11.6 万张统计。
    """
    class_counts = Counter()
    train_dir = os.path.join(data_dir, "train")
    for cls_name in os.listdir(train_dir):
        cls_dir = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_dir) and cls_name.isdigit():
            count = sum(1 for f in os.listdir(cls_dir)
                        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS)
            if count > 0:
                class_counts[int(cls_name)] = count
    with open(class_number_file, 'w') as f:
        json.dump(dict(sorted(class_counts.items())), f, indent=4)
    logger.info("Class counts saved from FS scan: %d classes, %d total samples",
                len(class_counts), sum(class_counts.values()))


def _concat_class_examples(examples_dir, logger):
    """将 .tmp/ 下的 per-worker part 文件 concat 成 examples_dir/{cls_id}.md"""
    tmp_dir = os.path.join(examples_dir, '.tmp')
    if not os.path.isdir(tmp_dir):
        return
    from collections import defaultdict as _dd
    class_parts = _dd(list)
    for fname in sorted(os.listdir(tmp_dir)):
        cls_id = fname.split('.')[0]
        class_parts[cls_id].append(os.path.join(tmp_dir, fname))
    for cls_id, parts in class_parts.items():
        name = get_readable_name(int(cls_id)).split(", ")[0]
        safe_name = name.replace(' ', '_').replace('/', '_')
        md_path = os.path.join(examples_dir, f"{cls_id}_{safe_name}.md")
        os.makedirs(examples_dir, exist_ok=True)
        with open(md_path, 'w') as out:
            out.write(f"# Class {cls_id}: {name}\n\n")
            for part_path in sorted(parts):
                with open(part_path) as f:
                    out.write(f.read())
    import shutil
    shutil.rmtree(tmp_dir)
    logger.info("Examples saved to %s (%d classes)", examples_dir, len(class_parts))


def _save_single_gpu_examples(per_class, examples_dir, logger):
    """单 GPU 模式：直接写 examples_dir/{cls_id}.md"""
    if not per_class or not examples_dir:
        return
    os.makedirs(examples_dir, exist_ok=True)
    for cls_id, records in per_class.items():
        name = get_readable_name(cls_id).split(", ")[0]
        safe_name = name.replace(' ', '_').replace('/', '_')
        md_path = os.path.join(examples_dir, f"{cls_id}_{safe_name}.md")
        with open(md_path, 'w') as f:
            f.write(f"# Class {cls_id}: {name}\n\n")
            f.write(f"## Original Descriptions ({len(records)})\n\n")
            for k, (img_path, desc) in enumerate(records, 1):
                img_rel = f"images/{cls_id}/{os.path.basename(img_path)}"
                f.write(f"### Image {k}\n\n")
                f.write(f"![Image {k}]({img_rel})\n\n")
                f.write(f"**Description:** {desc}\n\n")
    logger.info("Examples saved to %s (%d classes)", examples_dir, len(per_class))


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    args.vlm_prompt = prompts.get("describe", {}).get("vlm_prompt")

    os.makedirs(os.path.dirname(args.existing_description_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("describe", os.path.join(args.log_dir, "pipeline_describe.log"))

    cleanup_stale_parts(args.existing_description_path, logger)

    description_file = args.existing_description_path
    tmp_file = description_file + ".tmp"
    if os.path.exists(tmp_file):
        logger.info("Removing stale temp file: %s", tmp_file)
        os.remove(tmp_file)
    if os.path.exists(description_file):
        logger.info("Backuping existing description file: %s", description_file)
        os.rename(description_file, description_file + "_" + time.strftime("%Y%m%d-%H%M%S"))

    if args.num_gpus > 1:
        logger.info("Multi-GPU DDP mode: %d GPUs", args.num_gpus)

        if not os.path.exists(args.class_number_file):
            logger.info("Pre-computing class counts from filesystem...")
            _count_classes_from_fs(args.data_dir, args.class_number_file, logger)

        master_port = _find_free_port()
        worker_args = {
            'data_dir': args.data_dir,
            'tail_num_threshold': args.tail_num_threshold,
            'class_number_file': args.class_number_file,
            'existing_description_path': args.existing_description_path,
            'examples_dir': args.examples_dir,
            'test': args.test,
            'log_dir': args.log_dir,
            'num_workers': max(1, args.num_workers // args.num_gpus),
            'batch_size': args.batch_size,
            'master_port': master_port,
            'vlm_backend': args.vlm_backend,
            'vlm_prompt': args.vlm_prompt,
        }

        mp.spawn(
            _ddp_worker,
            args=(args.num_gpus, worker_args),
            nprocs=args.num_gpus,
            join=True,
        )

        _merge_parts(args.existing_description_path, args.num_gpus, logger)
        if args.examples_dir:
            _concat_class_examples(args.examples_dir, logger)
    else:
        _main_single_gpu(args, logger)

    logger.info("Done. Output: %s", description_file)


if __name__ == "__main__":
    main()
