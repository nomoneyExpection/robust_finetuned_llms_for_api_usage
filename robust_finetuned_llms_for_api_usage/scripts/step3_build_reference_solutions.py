import json
import os

# 假设你已经有一个类似于 input 文件夹或其他目录来存储每个 API 的示例代码
API_EXAMPLES_DIR = "data/api_examples/"
REFERENCE_SOLUTION_PATH = "data/processed_data/reference_solutions.json"

# 读取 API 示例文件并生成参考解决方案
def generate_reference_solutions():
    # 获取所有 API 示例文件
    api_files = [f for f in os.listdir(API_EXAMPLES_DIR) if f.endswith(".py")]
    reference_solutions = []

    for api_file in api_files:
        api_name = api_file[:-3]  # 假设文件名是 API 名称，去掉 ".py" 后缀

        # 读取 API 示例文件内容
        with open(os.path.join(API_EXAMPLES_DIR, api_file), "r", encoding="utf-8") as f:
            solution = f.read().strip()

        # 构建参考解决方案数据
        reference_solutions.append({
            "api_name": api_name,
            "canonical_solution": solution
        })

    # 保存为 JSON 格式
    os.makedirs(os.path.dirname(REFERENCE_SOLUTION_PATH), exist_ok=True)
    with open(REFERENCE_SOLUTION_PATH, "w", encoding="utf-8") as f:
        json.dump(reference_solutions, f, ensure_ascii=False, indent=4)

    print(f"✅ Reference solutions saved to {REFERENCE_SOLUTION_PATH}")

if __name__ == "__main__":
    generate_reference_solutions()
