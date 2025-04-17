import os
import torch
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, classification_report
from peft import PeftModel
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

# 路径配置
base_model_path = "/root/autodl-tmp/llm-ft/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
adapter_path = " "
test_path = "/root/autodl-tmp/llm-ft/data/ChnSentiCorp_test.json"
result_path = "/root/autodl-tmp/llm-ft/ia3/eval"
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024

# 输出文件夹路径
model_short_name = "deepseek-distill-1.5B"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
eval_dir = os.path.join(
    result_path,
    f"{model_short_name}-{timestamp}"
)
os.makedirs(eval_dir, exist_ok=True)

# 初始化 GPU 监控
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval().to(device)

# 加载 JSON 测试数据
with open(test_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

label_map = {"积极": 1, "消极": 0}
id2label = {v: k for k, v in label_map.items()}

examples = []
for entry in raw_data:
    prompt = entry["instruction"] + "\n" + entry["input"]
    label = label_map.get(entry["output"], 0)
    examples.append({"text": prompt, "label": label})

test_dataset = Dataset.from_list(examples)

# 推理评估
all_preds = []
all_labels = [ex["label"] for ex in examples]
inference_times = []
wrong_preds = []
gpu_memory_used = []
gpu_utilization = []

for example in tqdm(examples):
    prompt = example["text"]
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    elapsed = time.time() - start_time
    inference_times.append(elapsed)

    # GPU状态记录
    mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    util_info = nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_memory_used.append(mem_info.used / 1024**2)  # MB
    gpu_utilization.append(util_info.gpu)          # %

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.replace(prompt, "").strip().lower()

    if "积极" in response:
        pred = 1
    elif "消极" in response:
        pred = 0
    else:
        pred = 0

    all_preds.append(pred)
    if pred != example["label"]:
        wrong_preds.append({
            "text": prompt,
            "true_label": id2label[example["label"]],
            "pred_label": id2label[pred],
            "response": response
        })

# 输出评估结果
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
report = classification_report(
    all_labels,
    all_preds,
    target_names=["消极", "积极"]
)

# 推理耗时统计
avg_latency_ms = np.mean(inference_times) * 1000
max_latency_ms = np.max(inference_times) * 1000
min_latency_ms = np.min(inference_times) * 1000

# GPU资源统计
avg_mem = np.mean(gpu_memory_used)
max_mem = np.max(gpu_memory_used)
min_mem = np.min(gpu_memory_used)

avg_util = np.mean(gpu_utilization)
max_util = np.max(gpu_utilization)
min_util = np.min(gpu_utilization)

# 保存评估摘要
summary = (
    f"模型: {model_short_name}\n"
    f"时间: {timestamp}\n"
    f"样本数: {len(all_preds)}\n"
    f"Accuracy: {acc:.4f}\n"
    f"F1 Score: {f1:.4f}\n"
    f"平均推理耗时: {avg_latency_ms:.2f} ms\n"
    f"最大耗时: {max_latency_ms:.2f} ms\n"
    f"最小耗时: {min_latency_ms:.2f} ms\n"
    f"\nClassification Report:\n{report}\n"
    f"\nGPU 资源消耗统计：\n"
    f"显存使用(MB):平均 {avg_mem:.2f}，最大 {max_mem:.2f}，最小 {min_mem:.2f}\n"
    f"GPU 利用率（%）：平均 {avg_util:.2f}，最大 {max_util:.2f}，最小 {min_util:.2f}\n"
)

print(summary)

with open(
    os.path.join(eval_dir, "eval_summary.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

pd.DataFrame(wrong_preds).to_csv(
    os.path.join(eval_dir, "wrong_predictions.csv"),
    index=False
)
print(f"评估完成 结果与错误样本已保存至：{eval_dir}")