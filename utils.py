"""
工具函数
"""
from collections import Counter
import json


def count_samples(dataloader):
    """统计数据加载器中各类别样本数，保存到文件"""
    class_counts = Counter()

    for _, batch_labels, _ in dataloader:
        class_counts.update(batch_labels.tolist())

    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")

    with open('data_txt/ImageNet_LT/imagenetlt_class_count.txt', 'w') as f:
        f.write(json.dumps(dict(class_counts), indent=4))
