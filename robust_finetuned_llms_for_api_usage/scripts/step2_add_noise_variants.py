import json
import os
import random
import logging
from tqdm import tqdm

# ========== 路径配置 ==========
INPUT_PATH = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/train_semantic_variants.json'
OUTPUT_PATH = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/data/generated_prompts/train_semantic_variants_noisy.json'
LOG_PATH = 'D:/PycharmProjects/robust_finetuned_llms_for_api_usage/log/add_noise.log'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ========== 日志配置 ==========
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========== 冗余/噪声模版 ==========
NOISE_PREFIXES = [
    "Please", "Could you", "I want to", "Try to", "Just go ahead and", "Let's",
    "Make sure to", "In pandas,", "Hey, can you", "Kindly", "You should"
]

NOISE_SUFFIXES = [
    "as soon as possible.", "if you don't mind.", "thanks!", "in pandas.",
    "before continuing.", "without using loops.", "when the data is large.",
    "for better performance.", "in the latest version."
]

def add_noise(text):
    """在原始 prompt 前后加噪声冗余语言"""
    prefix = random.choice(NOISE_PREFIXES)
    suffix = random.choice(NOISE_SUFFIXES)
    return f"{prefix} {text} {suffix}"

# ========== 断点恢复 ==========
def load_existing_output():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_output(data):
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ========== 主执行函数 ==========
def main():
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logging.error(f"读取输入失败: {e}")
        return

    existing_output = load_existing_output()
    done_api_names = {item['api_name'] for item in existing_output}
    logging.info(f"已完成API数量: {len(done_api_names)}")

    noisy_data = existing_output[:]
    for item in tqdm(input_data, desc="注入噪声"):
        api_name = item['api_name']
        if api_name in done_api_names:
            continue

        try:
            noisy_variants = []
            for var in item['variants']:
                for _ in range(3):  # 每个原始语义变体生成3个含噪版本
                    noisy_variants.append(add_noise(var))

            noisy_data.append({
                "api_name": api_name,
                "instruction": item['instruction'],
                "variants": noisy_variants
            })

            # 每个API完成后保存一次
            save_output(noisy_data)
            logging.info(f"完成: {api_name}")

        except Exception as e:
            logging.error(f"处理 {api_name} 失败: {e}")
            continue

    logging.info("🎉 所有指令已处理完成。")

if __name__ == '__main__':
    main()
