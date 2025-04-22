import json
import os
import requests
from bs4 import BeautifulSoup
import time
from typing import List

# 定义之前使用的25个API
previous_api_list = [
    "merge", "groupby", "pivot", "apply", "loc", "dropna", "fillna", "sort_values",
    "query", "duplicated", "value_counts", "astype", "replace", "set_index", "reset_index",
    "rename", "concat", "isnull", "sample", "nunique", "to_csv", "from_dict", "columns",
    "index", "memory_usage"
]

# 定义文件路径
SEMANTIC_VARIANTS_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/semantic_variants.json"
NOISE_VARIANTS_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/semantic_variants_noisy.json"
REFERENCE_SOLUTIONS_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/reference_solutions.json"
TRAINING_SET_PATH = "data/train/train_code_completion.jsonl"


# 加载 JSON 文件
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 保存 JSONL 文件
def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 从 Pandas 官方文档抓取所有 API 列表
def fetch_all_pandas_apis() -> List[str]:
    url = "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求文档时出错: {e}")
        return []

    apis = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        methods = soup.select('h3[id="methods"] ~ table td a')
        for method in methods:
            api = method.text.strip()
            if api and not api.startswith('_'):
                apis.append(api)
        return apis
    else:
        return []


# 过滤掉已有的25个API，获取剩余的API列表
def get_remaining_apis():
    all_apis = fetch_all_pandas_apis()
    remaining_apis = [api for api in all_apis if api not in previous_api_list]
    return remaining_apis


# 从 Pandas 官方文档抓取代码示例
def fetch_example_code_from_doc(api_name: str) -> List[str]:
    url = f"https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.{api_name}.html"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求文档时出错: {e}")
        return []

    examples = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        code_blocks = soup.select('div.highlight-default.notranslate pre')
        for block in code_blocks:
            code = block.text.strip()
            if len(code) > 20:
                examples.append(code)
        return list(set(examples))
    else:
        return []


# 从 GitHub 上抓取代码示例
def fetch_github_examples(api_name: str) -> List[str]:
    search_url = f"https://api.github.com/search/code?q=pandas+{api_name}+in:file+language:python+stars:>10"
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get(search_url, headers=headers, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求 GitHub 时出错: {e}")
        return []

    examples = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            file_url = item['html_url']
            file_content = fetch_github_file(file_url)
            if file_content:
                examples.append(file_content)
        return examples
    else:
        return []


# 获取 GitHub 文件内容
def fetch_github_file(file_url: str) -> str:
    try:
        response = requests.get(file_url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求 GitHub 文件时出错: {e}")
        return None

    if response.status_code == 200:
        return response.text
    else:
        return None


# 从 Stack Overflow 抓取代码示例
def fetch_stack_overflow_examples(api_name: str) -> List[str]:
    search_url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=activity&tagged=pandas&intitle={api_name}&site=stackoverflow"
    try:
        response = requests.get(search_url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求 Stack Overflow 时出错: {e}")
        return []

    examples = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            question_id = item['question_id']
            question_url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers?order=desc&sort=votes&site=stackoverflow"
            headers = {"Accept": "application/vnd.stackexchange.v2.3+json"}
            try:
                answers_response = requests.get(question_url, headers=headers, verify=False)
                answers_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"请求 Stack Overflow 答案时出错: {e}")
                continue

            if answers_response.status_code == 200:
                answers_data = answers_response.json()
                for answer in answers_data.get('items', []):
                    if 'body' in answer:
                        soup = BeautifulSoup(answer['body'], 'html.parser')
                        code_blocks = soup.find_all('code')
                        for block in code_blocks:
                            code = block.text.strip()
                            if len(code) > 20:
                                examples.append(code)
        return examples
    else:
        return []


# 构建提示和解决方案
def build_prompt_and_solution(api_name: str, example: str) -> dict:
    instruction = f"Provide a code example for pandas.DataFrame.{api_name}"
    return {
        "api_name": api_name,
        "instruction": instruction,
        "prompt": example,
        "solution": example
    }


# 生成训练集
def generate_training_set(num_samples: int = 3000):
    training_set = []
    remaining_apis = get_remaining_apis()

    for api_name in remaining_apis:
        # 从官方文档获取示例
        doc_examples = fetch_example_code_from_doc(api_name)
        for example in doc_examples:
            training_set.append(build_prompt_and_solution(api_name, example))
            if len(training_set) >= num_samples:
                break

        # 从 GitHub 获取示例
        github_examples = fetch_github_examples(api_name)
        for example in github_examples:
            training_set.append(build_prompt_and_solution(api_name, example))
            if len(training_set) >= num_samples:
                break

        # 从 Stack Overflow 获取示例
        so_examples = fetch_stack_overflow_examples(api_name)
        for example in so_examples:
            training_set.append(build_prompt_and_solution(api_name, example))
            if len(training_set) >= num_samples:
                break

        if len(training_set) >= num_samples:
            break

    # 确保数据集大小
    if len(training_set) < num_samples:
        print(f"警告：生成的数据集大小为 {len(training_set)}，小于目标 {num_samples}。")

    # 去重
    training_set = list({item["prompt"]: item for item in training_set}.values())

    # 保存数据集
    save_jsonl(TRAINING_SET_PATH, training_set)
    print(f"生成的训练集包含 {len(training_set)} 条数据，已保存到 {TRAINING_SET_PATH}")


if __name__ == "__main__":
    generate_training_set(num_samples=3000)