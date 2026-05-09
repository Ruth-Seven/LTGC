"""
数据加载模块
ImageNet-LT 数据集加载，支持数字目录结构
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.JPEG', '.bmp'}


class BalancedSampler(Sampler):
    """类别平衡采样器"""

    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)
        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        return sum(len(b) for b in self.buckets) if hasattr(self, 'retain_epoch_size') and self.retain_epoch_size else max(len(b) for b in self.buckets) * self.bucket_num


class ImageNetLTDataset(Dataset):
    """扫描数字目录结构 {data_dir}/{split}/{class_id}/*.png 的数据集"""

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.img_paths = []
        self.labels = []

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_dirs = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and d.isdigit()
        )

        for cls_name in class_dirs:
            cls_id = int(cls_name)
            cls_dir = os.path.join(split_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    self.img_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_id)

        self.targets = self.labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, index


class ImageNetLTDataLoader(DataLoader):
    """带可选平衡采样的 ImageNet-LT 数据加载器"""

    def __init__(self, data_dir, split='train', batch_size=1, shuffle=True,
                 num_workers=1, balanced=False, retain_epoch_size=True,
                 image_size=224):
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

        dataset = ImageNetLTDataset(data_dir, split, transform)
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.num_classes = len(np.unique(dataset.targets))

        self.cls_num_list = [0] * self.num_classes
        for label in dataset.targets:
            self.cls_num_list[label] += 1

        if balanced and split == 'train':
            buckets = [[] for _ in range(self.num_classes)]
            for idx, label in enumerate(dataset.targets):
                buckets[label].append(idx)
            sampler = BalancedSampler(buckets, retain_epoch_size)
            shuffle = False
        else:
            sampler = None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            sampler=sampler,
        )
