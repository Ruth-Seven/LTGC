"""
文本扩展模块 - 使用 DeepSeek Chat 生成丰富的类别描述
Step 2 of LTGC pipeline
"""
import pandas as pd
from data_txt.imagenet_label_mapping import get_readable_name
import csv
import argparse
import time
import os
import requests
from pathlib import Path

from llm_config import (
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_API_ENDPOINT,
    get_deepseek_headers,
    DESCRIPTIONS_DIR,
)
import requests
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='LTGC Step 2: Description Extension')
    parser.add_argument('-exi', '--existing_description_path',
                        default=str(Path(DESCRIPTIONS_DIR) / 'existing_description_list.csv'),
                        type=str,
                        help='Input existing description CSV path')
    parser.add_argument('-m', '--max_generate_num', default=200, type=int,
                        help='Max descriptions per class')
    parser.add_argument('-ext', '--extended_description_path',
                        default=str(Path(DESCRIPTIONS_DIR) / 'extended_description.csv'),
                        type=str,
                        help='Output extended description CSV path')
    return parser.parse_args()


def parse_response(output):
    """统一解析 LLM 响应文本为描述列表

    支持格式:
    - "- description"
    - "\\n\\n- description"  
    - "\\n\\ndescription"
    - 纯文本（单条回退）
    """
    if '\n\n- ' in output:
        return [s.strip('- ').strip() for s in output.split("\n\n") if s.strip('- ').strip()]
    if '\n- ' in output:
        return [s.strip('- ').strip() for s in output.split("\n- ") if s.strip('- ').strip()]
    if '\n\n' in output:
        return [s.strip() for s in output.split("\n\n") if s.strip()]
    if '\n' in output:
        return [s.strip() for s in output.split("\n") if s.strip()]
    return [output.strip()]


def main():
    args = parse_args()
    headers = get_deepseek_headers()

    os.makedirs(os.path.dirname(args.extended_description_path), exist_ok=True)

    df = pd.read_csv(args.existing_description_path, header=None, names=['label', 'text'])
    grouped_texts = df.groupby('label')['text'].apply(lambda x: '\n'.join(x)).to_dict()
    grouped_list = df.groupby('label')['text'].apply(list).to_dict()

    total_classes = len(grouped_texts)
    for cls_idx, (label, text) in enumerate(grouped_texts.items()):
        current_all_description = grouped_list[label]

        print(f"[llm_extension] Class {label} ({cls_idx + 1}/{total_classes}): {len(current_all_description)} existing descriptions")

        while len(current_all_description) < args.max_generate_num:
            real_name = get_readable_name(int(label)).split(", ")[0]

            system_content = "You will follow the Template to describe the object. Template: A photo of the class " + real_name + " {with distinctive features}{in specific scenes}. "
            current_description = text

            user_content = "Besides these descriptions mentioned above, please use the same Template to list other possible {distinctive features} and {specific scenes} for the class " + real_name

            payload = {
                "model": DEEPSEEK_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": current_description},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": DEEPSEEK_MAX_TOKENS
            }

            try:
                response = requests.post(
                    DEEPSEEK_API_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                output = response.json()['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                print(f"[llm_extension] API request failed for class {label}: {e}")
                time.sleep(5)
                continue
            except (KeyError, ValueError) as e:
                print(f"[llm_extension] Response parse error for class {label}: {e}")
                break

            sentences = parse_response(output)
            sentences = [s for s in sentences if s.startswith('A')]

            if not sentences:
                print(f"[llm_extension] No valid descriptions generated for class {label}, retrying...")
                time.sleep(2)
                continue

            current_all_description.extend(sentences)

            with open(args.extended_description_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for s in sentences:
                    writer.writerow([label, s])

            print(f"[llm_extension] Class {label}: generated {len(sentences)} descriptions (total: {len(current_all_description)})")

        print(f"[llm_extension] Class {label} done: {len(current_all_description)} descriptions total")


if __name__ == "__main__":
    main()
