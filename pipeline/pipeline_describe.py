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

from config import IMAGENET_DIR, DESCRIPTIONS_DIR, DESCRIPTION_EXAMPLE_DIR, CLASS_COUNT_FILE
from data.data_loader import ImageNetLTDataset, SUPPORTED_EXTENSIONS
from data_txt.imagenet_label_mapping import get_readable_name
from model.vision_lmm import describe_image_batch
from utils import validate_description
from collections import defaultdict, Counter


def collate_no_stack(batch):
    """不堆叠 image tensor，保持 list 以支持不同尺寸的图片"""
    images = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch])
    return images, targets, indices


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
    parser.add_argument('--examples-len',
                        default=30, type=int,
                        help='Number of example images to save')
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode with limited examples')
    parser.add_argument('--log_dir', type=str, default="/tmp",
                        help='Log file directory')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs (DistributedSampler 数据并行)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader workers per GPU (default: 16)')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size for LLaVA inference (default: 6)')
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


def describe_example_markdown(examples, output_dir, logger=None):
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "tail_class_description_examples.md")
    to_pil = transforms.ToPILImage()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tail Class Description Examples\n\n")
        f.write(f"Total examples: {len(examples)}\n\n")

        for i, (cls_id, img_tensor, description, class_name) in enumerate(examples):
            img_filename = f"example_{cls_id}_{i}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            img_tensor_cpu = img_tensor.squeeze(0).cpu().clamp(0, 1)
            to_pil(img_tensor_cpu).save(img_path)

            f.write(f"## Example {i+1}: Class {cls_id} - {class_name}\n\n")
            f.write(f"![Image]({img_filename})\n\n")
            f.write(f"**Description:** {description}\n\n")
            f.write("---\n\n")

    (logger or logging.getLogger()).info("Examples saved to %s", md_path)


def _ddp_worker(rank, world_size, args_dict):
    """DDP worker rank/GPU N: DistributedSampler 自动按 rank 分片数据"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args_dict['master_port'])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

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
    open(part_path, 'w').close()
    data_to_write = []
    examples = []
    example_classes = set()
    processed = 0
    tail_count = 0

    max_examples = args_dict.get('examples_len', 30) // world_size + 1
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
            G = len(indices)
            group_imgs = [data_list[i] for i in indices]
            prompts = [text_prompt.format(name=name)] * G
            descriptions = describe_image_batch(group_imgs, prompts)

            tail_count += G
            for i, desc in zip(indices, descriptions):
                if desc and validate_description(desc, name):
                    data_to_write.append((cls_id, desc))

                    if cls_id not in example_classes and len(example_classes) < max_examples:
                        example_classes.add(cls_id)
                        examples.append((cls_id, data_list[i].clone(), desc, name))

                if len(data_to_write) >= 10:
                    with open(part_path, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    data_to_write = []

        processed += B
        if test_mode and len(example_classes) >= 10:
            break

    if data_to_write:
        with open(part_path, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    if examples:
        part_examples_dir = os.path.join(args_dict.get('examples_dir', DESCRIPTION_EXAMPLE_DIR), f"part{rank}")
        describe_example_markdown(examples, part_examples_dir, logger)

    logger.info("DDP rank %d done. Processed: %d, Tail: %d", rank, processed, tail_count)
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
    logger.info("Merged %d rows into %s", merged_count, output_path)


def _main_single_gpu(args, logger):
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
    examples = []
    example_classes = set()

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
            G = len(indices)
            group_imgs = [data_list[i] for i in indices]
            prompts = [text_prompt.format(name=name)] * G
            descriptions = describe_image_batch(group_imgs, prompts)

            tail_count += G
            for i, desc in zip(indices, descriptions):
                if desc and validate_description(desc, name):
                    data_to_write.append((cls_id, desc))

                    if cls_id not in example_classes and len(example_classes) < args.examples_len:
                        example_classes.add(cls_id)
                        examples.append((cls_id, data_list[i].clone(), desc, name))

                if len(data_to_write) >= 10:
                    with open(tmp_file, 'a', newline='') as f:
                        csv.writer(f).writerows(data_to_write)
                    data_to_write = []

        processed += B
        if args.test and len(example_classes) >= args.examples_len:
            break

    if data_to_write:
        with open(tmp_file, 'a', newline='') as f:
            csv.writer(f).writerows(data_to_write)

    if examples:
        describe_example_markdown(examples, args.examples_dir, logger)

    os.rename(tmp_file, description_file)
    logger.info("Single-GPU done. Processed: %d, Tail: %d, Output: %s", processed, tail_count, description_file)


def _count_classes_from_fs(data_dir, class_number_file, logger):
    """直接从目录结构统计各类别样本数（不走 DataLoader，极快）"""
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.existing_description_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("describe", os.path.join(args.log_dir, "pipeline_describe.log"))

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
            'examples_len': args.examples_len,
            'test': args.test,
            'log_dir': args.log_dir,
            'num_workers': max(1, args.num_workers // args.num_gpus),
            'batch_size': args.batch_size,
            'master_port': master_port,
        }

        mp.spawn(
            _ddp_worker,
            args=(args.num_gpus, worker_args),
            nprocs=args.num_gpus,
            join=True,
        )

        _merge_parts(args.existing_description_path, args.num_gpus, logger)
    else:
        _main_single_gpu(args, logger)

    logger.info("Done. Output: %s", description_file)


if __name__ == "__main__":
    main()
