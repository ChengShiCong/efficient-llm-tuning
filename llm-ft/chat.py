import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 路径配置
base_model_path = "/root/autodl-tmp/llm-ft/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
adapter_path = ""  # 替换为adapter路径，如果不需要adapter则留空
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

if adapter_path and adapter_path.strip():
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"Loaded adapter from {adapter_path}")
else:
    model = base_model
    print("No adapter loaded, using the base model.")

model.eval().to(device)

print("模型已加载完毕，可以开始对话 (输入 'exit' 退出).")

while True:
    question = input("用户：")
    if question.lower() == 'exit':
        break

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  
            do_sample=True,      
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"模型：{response}")

print("对话结束。")
