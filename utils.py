"""
工具函数
"""
from collections import Counter
from tqdm import tqdm
import json


def count_samples(dataloader, output_path=None):
    """统计数据加载器中各类别样本数，保存到文件"""
    class_counts = Counter()

    for _, batch_labels, _ in tqdm(dataloader, desc="[counting] Processing", unit="img"):
        class_counts.update(batch_labels.tolist())

    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")

    if output_path:
        with open(output_path, 'w') as f:
            f.write(json.dumps(dict(class_counts), indent=4))
