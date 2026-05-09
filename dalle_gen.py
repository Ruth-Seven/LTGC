"""
[DEPRECATED] DALL-E 2 图像生成模块

此文件已废弃，由 hf_gen.py 替代。
保留用于参考。

替代方案:
  - dalle_gen() → hf_gen() in hf_gen.py (使用 HuggingFace Stable Diffusion)
  - description_refine() → hf_gen.description_refine() (使用 DeepSeek Chat)
  - get_cls_template() → hf_gen.get_cls_template() (使用 DeepSeek Chat)
"""
import warnings
warnings.warn(
    "dalle_gen.py is deprecated. Use hf_gen.py instead.",
    DeprecationWarning,
    stacklevel=2
)

from openai import OpenAI
import requests


client = OpenAI(api_key='Replace with your own OPENAI KEY.')

def dalle_gen(client, saved_path, input_text, saved=False):
    """[DEPRECATED] Use hf_gen.hf_gen() instead"""
    print("[WARNING] dalle_gen() is deprecated. Use hf_gen() from hf_gen.py instead.")
    return None


def get_cls_index_name(label_index):
    """[DEPRECATED] Use hf_gen.get_cls_index_name() instead"""
    with open("data_txt/ImageNet_LT/ImageNet_cls_name.txt", "r") as file:
        labels = [label.strip('",') for label in file.read().splitlines()]

    if 0 <= label_index < len(labels):
        return labels[label_index]
    else:
        return "Index out of range"



def description_refine(input_text, cls_name):
    """[DEPRECATED] Use hf_gen.description_refine() instead"""
    print("[WARNING] description_refine() is deprecated. Use hf_gen.description_refine() instead.")
    return input_text


def get_cls_template(cls_name, cls_index, filename="data_txt/ImageNet_LT/class_templates.txt"):
    """[DEPRECATED] Use hf_gen.get_cls_template() instead"""
    print("[WARNING] get_cls_template() is deprecated. Use hf_gen.get_cls_template() instead.")
    return f"Template: A photo of the class {cls_name}"