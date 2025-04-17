import csv
import json
import os

# 输入输出路径
csv_path = "/root/autodl-tmp/ft_code_2/dataset1/ChnSentiCorp_train.csv"
output_jsonl_path = "/root/autodl-tmp/ft_code_2/dataset1/datasets.jsonl"

os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

# 转换逻辑 
with open(csv_path, "r", encoding="utf-8") as f_csv, open(output_jsonl_path, "w", encoding="utf-8") as f_jsonl:
    reader = csv.DictReader(f_csv)
    for row in reader:
        text = row["text"].strip()
        label = row["label"].strip()

        prompt = f"请对下面文本进行情感分析：{text}"
        if label == "1":
            completion = "这是正面回答"
        else:
            completion = "这是负面回答"

        sample = {
            "prompt": prompt,
            "completion": completion
        }
        f_jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f" 转换完成：{output_jsonl_path}")
