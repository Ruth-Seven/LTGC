"""
工具函数
"""
from collections import Counter
from tqdm import tqdm
import json
import os
import re


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


NON_PHOTO_PATTERNS = [
    r"\ba painting of\b",
    r"\ba drawing of\b",
    r"\ba sketch of\b",
    r"\ban illustration of\b",
    r"\ba cartoon of\b",
    r"\ba sculpture of\b",
    r"\ba render[ing]* of\b",
    r"\ba screenshot of\b",
]
_photo_check = re.compile("|".join(NON_PHOTO_PATTERNS), re.IGNORECASE)


def validate_description(description, class_name):
    """校验 LLaVA 生成的描述是否符合 LTGC 模板格式。

    规则:
    1. 必须以 "A photo of" 开头 (不允许 painting/drawing 等)
    2. 必须包含 "class {class_name}" (保留模板类名标识)

    Returns:
        bool: True 表示通过校验
    """
    if not description or not isinstance(description, str):
        return False
    desc = description.strip()

    if not re.match(r"^A photo of", desc, re.IGNORECASE):
        return False

    if _photo_check.search(desc):
        return False

    if f"class {class_name}" not in desc.lower():
        return False

    return True


def cleanup_stale_parts(output_path, logger=None):
    """删除指定输出路径的所有 *.part* 后缀文件（上次运行残留）

    Args:
        output_path: 主输出文件路径，part 文件名为 output_path.part{rank}
        logger: 可选的 logging.Logger，输出删除信息
    """
    import glob as _glob
    for stale in _glob.glob(f"{output_path}.part*"):
        try:
            os.remove(stale)
            if logger:
                logger.info("Removed stale part: %s", stale)
        except OSError:
            pass
