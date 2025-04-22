import requests
from bs4 import BeautifulSoup
import json
from typing import List

# 定义需要抓取的 API 列表
api_list = [
    "merge", "groupby", "pivot", "apply", "loc", "dropna", "fillna", "sort_values",
    "query", "duplicated", "value_counts", "astype", "replace", "set_index", "reset_index",
    "rename", "concat", "isnull", "sample", "nunique", "to_csv", "from_dict", "columns",
    "index", "memory_usage"
]


# 从 Pandas 官方文档抓取代码示例
def fetch_example_code_from_doc(api_name: str) -> List[str]:
    url = f"https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.{api_name}.html"
    response = requests.get(url)

    examples = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # 更精准地定位代码示例
        code_blocks = soup.select('div.highlight-default.notranslate')
        for block in code_blocks:
            code = block.text.strip()
            # 过滤掉过短的内容
            if len(code) > 20:
                examples.append(code)
        return list(set(examples))  # 去重
    else:
        return []


# 从 GitHub 上抓取代码示例
def fetch_github_examples(api_name: str) -> List[str]:
    search_url = f"https://api.github.com/search/code?q=pandas+{api_name}+in:file+language:python+stars:>10"
    response = requests.get(search_url)

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
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.text
    else:
        return None


# 从 Stack Overflow 抓取代码示例
def fetch_stack_overflow_examples(api_name: str) -> List[str]:
    search_url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=activity&tagged=pandas&intitle={api_name}&site=stackoverflow"
    response = requests.get(search_url)

    examples = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            question_id = item['question_id']
            question_url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers?order=desc&sort=votes&site=stackoverflow"
            answers_response = requests.get(question_url)
            if answers_response.status_code == 200:
                answers_data = answers_response.json()
                for answer in answers_data.get('items', []):
                    soup = BeautifulSoup(answer['body'], 'html.parser')
                    code_blocks = soup.find_all('code')
                    for block in code_blocks:
                        code = block.text.strip()
                        if len(code) > 20:
                            examples.append(code)
        return examples
    else:
        return []


# 获取 API 示例
def fetch_all_examples():
    all_examples = {}

    for api_name in api_list:
        print(f"Fetching examples for {api_name} from official documentation...")
        doc_examples = fetch_example_code_from_doc(api_name)
        if doc_examples:
            all_examples[api_name] = {
                "from_doc": doc_examples
            }
        else:
            all_examples[api_name] = {}

    for api_name in api_list:
        print(f"Searching GitHub for examples of {api_name}...")
        github_examples = fetch_github_examples(api_name)
        if github_examples:
            if api_name in all_examples:
                all_examples[api_name]["from_github"] = github_examples
            else:
                all_examples[api_name] = {
                    "from_github": github_examples
                }

    for api_name in api_list:
        print(f"Searching Stack Overflow for examples of {api_name}...")
        so_examples = fetch_stack_overflow_examples(api_name)
        if so_examples:
            if api_name in all_examples:
                all_examples[api_name]["from_stack_overflow"] = so_examples
            else:
                all_examples[api_name] = {
                    "from_stack_overflow": so_examples
                }

    return all_examples


# 保存为 JSON 文件
def save_examples_to_file(examples, filename="api_examples.json"):
    with open(filename, 'w') as f:
        json.dump(examples, f, indent=4)


# 主函数
if __name__ == "__main__":
    all_examples = fetch_all_examples()
    save_examples_to_file(all_examples)
    print("All examples fetched and saved successfully!")