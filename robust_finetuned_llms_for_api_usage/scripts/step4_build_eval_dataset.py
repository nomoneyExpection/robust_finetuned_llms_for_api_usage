import json
import os

# 文件路径（可根据需要自行修改）
SEMANTIC_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/semantic_variants.json"
NOISE_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/semantic_variants_noisy.json"
REFERENCE_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/reference_solutions.json"
OUTPUT_PATH = "D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/eval/eval_code_completion.jsonl"

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

# 构建统一评估数据
def build_eval_dataset():
    # 加载数据并检查
    semantic_data = load_json(SEMANTIC_PATH)
    noise_data = load_json(NOISE_PATH)
    reference_data = load_json(REFERENCE_PATH)

    # 构建 api_name -> canonical_solution 映射
    api2solution = {}
    for api_name, solutions in reference_data.items():
        if "canonical_solutions" in solutions and solutions["canonical_solutions"]:
            # 使用第一个标准解决方案作为代表
            api2solution[api_name] = solutions["canonical_solutions"][0]
        else:
            print(f"[Warning] No canonical solutions for {api_name}")

    dataset = []
    counter = {}

    def process_variants(variants_list, variant_type):
        for item in variants_list:
            api_name = item["api_name"]
            base_instruction = item["instruction"]
            prompts = item["variants"]

            if api_name not in api2solution:
                print(f"[Warning] No canonical solution for {api_name}")
                continue

            for i, prompt in enumerate(prompts):
                # 构造任务 ID
                count = counter.get(api_name, 0)
                task_id = f"{api_name}/{variant_type}_{str(count).zfill(3)}"
                counter[api_name] = count + 1

                dataset.append({
                    "task_id": task_id,
                    "prompt": prompt,
                    "canonical_solution": api2solution[api_name],
                    "api_name": api_name
                })

    print("🔧 Processing semantic variants...")
    process_variants(semantic_data, "semantic")

    print("🔧 Processing noise variants...")
    process_variants(noise_data, "noise")

    print(f"✅ Total tasks generated: {len(dataset)}")
    save_jsonl(OUTPUT_PATH, dataset)
    print(f"📁 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_eval_dataset()