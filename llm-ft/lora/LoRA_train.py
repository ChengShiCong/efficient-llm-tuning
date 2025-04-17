import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import get_peft_model, LoraConfig, TaskType
from datetime import datetime
import time
import shutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

# 参数设置
print("正在初始化训练参数...")

model_name_or_path = "/root/autodl-tmp/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
model_short_name = "deepseek-distill-1.5B"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

base_output_root = "/root/autodl-tmp/llm-ft/lora/output"
base_log_root = "/root/autodl-tmp/llm-ft/lora/log"
gpu_log_root = "/root/autodl-tmp/llm-ft/lora/gpu_usage"

output_dir = os.path.join(base_output_root, f"{model_short_name}-{timestamp}")
logging_dir = os.path.join(base_log_root, f"{model_short_name}-{timestamp}")
gpu_log_subdir = os.path.join(gpu_log_root, f"{model_short_name}-{timestamp}")
gpu_log_path = os.path.join(gpu_log_subdir, "gpu_usage.csv")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(gpu_log_subdir, exist_ok=True)
success_training = False

train_path = "/root/autodl-tmp/llm-ft/data/ChnSentiCorp_train.json"

# 加载数据
print("正在加载训练集...")
train_df = pd.read_json(train_path)
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.select(range(1000))
print(f"训练样本数：{len(train_dataset)}")

# 加载分词器与模型
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    use_fast=False
)
config = AutoConfig.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
config.pad_token_id = tokenizer.pad_token_id

print("正在加载预训练模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
    config=config
)

# 应用 LoRA 配置
print("正在注入 LoRA 模块...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 数据预处理
oversize = 0
def process_func(example):
    MAX_LENGTH = 512
    global oversize
    input_ids, attention_mask, labels = [], [], []

    prompt = f"{example['instruction']}\n{example['input']}"
    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        padding=False
    )
    target_enc = tokenizer(
        example["output"],
        add_special_tokens=False,
        truncation=True,
        padding=False
    )

    input_ids += prompt_enc["input_ids"] + target_enc["input_ids"] + [tokenizer.eos_token_id]
    attention_mask += prompt_enc["attention_mask"] + target_enc["attention_mask"] + [1]
    labels += [-100] * len(prompt_enc["input_ids"]) + target_enc["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        oversize += 1
        print("数据超出最大长度")
        print(prompt)
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_dataset = train_dataset.map(
    process_func,
    remove_columns=["instruction", "input", "output"]
)
print("超出长度数据量：", oversize)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# 设置训练参数
print("正在配置训练参数...")
training_args = TrainingArguments(
    warmup_steps=100,
    output_dir=output_dir,
    logging_dir=logging_dir,
    report_to="tensorboard",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_grad_norm=5.0,
    learning_rate=3e-5,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=3,
    remove_unused_columns=False
)

# 自定义回调
class TrainingSpeedCallback(TrainerCallback):
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_start = time.time()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        train_end = time.time()
        duration = train_end - self.train_start
        step_count = state.global_step
        avg_step_time = duration / step_count if step_count > 0 else 0
        summary = (
            f"训练总时长: {duration:.2f} 秒\n"
            f"训练步数: {step_count}\n"
            f"平均每步耗时: {avg_step_time:.4f} 秒\n"
        )
        print(summary)
        with open(os.path.join(gpu_log_subdir, "train_time_summary.txt"), "w") as f:
            f.write(summary)

class GPUStatsCallback(TrainerCallback):
    def __init__(self, log_file=gpu_log_path, gpu_index=0, log_every_n_steps=5):
        self.log_file = log_file
        self.gpu_index = gpu_index
        self.log_every_n_steps = log_every_n_steps
        nvmlInit()
        self.device = nvmlDeviceGetHandleByIndex(self.gpu_index)
        with open(self.log_file, "w") as f:
            f.write("timestamp,step,used_memory_MB,total_memory_MB,utilization_percent\n")

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step + 1) % self.log_every_n_steps != 0:
            return
        mem_info = nvmlDeviceGetMemoryInfo(self.device)
        util_info = nvmlDeviceGetUtilizationRates(self.device)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(
                f"{now},{state.global_step},{mem_info.used/1024**2:.2f},"
                f"{mem_info.total/1024**2:.2f},{util_info.gpu}\n"
            )

# 启动训练
print("开始训练模型...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        TrainingSpeedCallback(),
        GPUStatsCallback()
    ]
)

try:
    trainer.train()
    success_training = True
except Exception as e:
    print(f"训练中发生错误：{str(e)}")

# 保存模型 or 清理
if success_training:
    print("训练完成，正在保存模型和分词器...")
    model.save_pretrained(os.path.join(output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print("模型保存成功，训练结束。")
else:
    print("训练未完成，正在删除输出目录...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)
    if os.path.exists(gpu_log_subdir):
        shutil.rmtree(gpu_log_subdir)
    print("相关目录已清理。")