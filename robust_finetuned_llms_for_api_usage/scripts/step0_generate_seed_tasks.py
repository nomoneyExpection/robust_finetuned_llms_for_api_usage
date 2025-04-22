import pandas as pd
import json
import os

CSV_PATH = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/seed_prompts.csv'
OUTPUT_PATH = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/train_seed_tasks.json'

def generate_seed_tasks():
    df = pd.read_csv(CSV_PATH)
    tasks = []

    for _, row in df.iterrows():
        task = {
            "api_name": row["api_name"],
            "instruction": row["base_prompt"],
            "input": "",         # 可留空，后续阶段有需要可添加 context/input
            "output": ""         # 初始为 ""，等待模型生成 output（代码等）
        }
        tasks.append(task)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ 共生成 {len(tasks)} 条 seed_tasks，保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_seed_tasks()
