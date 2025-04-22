import json
import os
import re

API_EXAMPLES_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/api_examples.json"
REFERENCE_SOLUTIONS_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/reference_solutions.json"


def extract_code_blocks(example_list):
    code_blocks = []
    for item in example_list:
        # 提取 >>> 或 ... 开头的代码行
        code_lines = []
        for line in item.split('\n'):
            if line.strip().startswith(('>>>', '...')):
                code_lines.append(re.sub(r'^(>>>|\.\.\.)\s*', '', line.strip()))
        if code_lines:
            code_blocks.append('\n'.join(code_lines))
    return code_blocks


def build_reference_solutions(api_example_path, save_path):
    with open(api_example_path, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    reference_data = {}

    for api_name, content in api_data.items():
        from_doc_examples = content.get("from_doc", [])
        code_blocks = extract_code_blocks(from_doc_examples)

        if not code_blocks:
            print(f"[Warning] No valid code blocks found for API: {api_name}")
            continue

        # 只选择前 N 个代码块作为参考解，避免过多重复
        reference_data[api_name] = {
            "canonical_solutions": code_blocks[:3]
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(reference_data, f, indent=2, ensure_ascii=False)

    print(f"✅ reference_solutions.json has been saved to: {save_path}")


if __name__ == "__main__":
    build_reference_solutions(API_EXAMPLES_PATH, REFERENCE_SOLUTIONS_PATH)
