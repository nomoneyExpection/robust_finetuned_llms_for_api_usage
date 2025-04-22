import os
import json
import time
import requests
from tqdm import tqdm

# === 路径配置 ===
api_config_path = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/configs/deepseek_config.json'
seed_task_path = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/train_seed_tasks.json'
output_path = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/train_semantic_variants.json'
log_path = '../log/semantic_gen.log'

# === 创建日志文件夹 ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# === 读取 API 配置 ===
with open(api_config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
API_URL = config['api_url']
API_KEY = config['api_key']
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# === 读取种子任务 ===
with open(seed_task_path, 'r', encoding='utf-8') as f:
    seed_tasks = json.load(f)

# === 如果存在已完成结果，进行恢复 ===
completed = {}
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        existing = json.load(f)
        for item in existing:
            completed[(item["api_name"], item["instruction"])] = item
else:
    existing = []

# === Chat Prompt 模板 ===
def build_prompt(instruction: str) -> str:
    return f"""You are an AI expert in natural language generation. Given an instruction related to pandas API usage, generate 15 semantically equivalent variations of the instruction. Only return a numbered list of the variations.

Instruction: {instruction}
"""

# === 日志记录 ===
def log(message: str):
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message.strip() + '\n')

# === API 调用 ===
def call_deepseek_api(prompt: str, max_retries: int = 3) -> list:
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 1.0
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            log(f"\n=== Prompt ===\n{prompt}\n=== Response ===\n{content}\n")
            return parse_variants(content)
        except Exception as e:
            log(f"[重试 {attempt}/{max_retries}] 请求失败: {e}")
            time.sleep(2 * attempt)

    log(f"[失败] 最终失败：{prompt}")
    return []

# === 返回内容解析 ===
def parse_variants(content: str) -> list:
    lines = content.strip().split('\n')
    variants = [line.split('.', 1)[-1].strip() for line in lines if '.' in line]
    return variants[:15]

# === 主流程 ===
results = existing.copy()
for task in tqdm(seed_tasks, desc="生成语义变体"):
    api_name = task["api_name"]
    inst = task["instruction"]

    if (api_name, inst) in completed:
        continue

    prompt = build_prompt(inst)
    variants = call_deepseek_api(prompt)

    if not variants:
        print(f"❌ 生成失败：{inst}")
        log(f"[ERROR] 无法生成变体 - instruction: {inst}")
        continue

    result = {
        "api_name": api_name,
        "instruction": inst,
        "variants": variants
    }
    results.append(result)

    # 增量保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 全部处理完毕，共生成 {len(results)} 项，结果保存在 {output_path}，日志保存在 {log_path}")
