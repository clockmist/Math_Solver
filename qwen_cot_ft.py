import json
import re
import torch
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab

# ==================== 配置参数 ====================
MAX_LENGTH = 512              # 设为 512（8的倍数），更适合 GPU 算力对齐
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output/Qwen_CoT"
TRAIN_JSON_PATH = "train_cot.json"

# ==================== 数据处理函数 ====================
def process_func(example):
    """
    将带思维链的数据预处理为模型输入格式。
    """
    # 1. 构造 Prompt（System + User + Assistant 头部）
    prompt = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    
    # 2. 构造 Response：【重要】直接在文本末尾加上 Qwen 的结束符 <|im_end|>\n
    response = tokenizer(
        f"{example['answer']}<|im_end|>\n", 
        add_special_tokens=False
    )

    # 3. 拼接（干净拼接，不手动加 pad_token）
    input_ids = prompt["input_ids"] + response["input_ids"]
    attention_mask = prompt["attention_mask"] + response["attention_mask"]
    
    # 标签：Prompt 部分设为 -100 忽略，Response 部分为真实 token
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"]

    # 截断超长序列
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ==================== 加载模型和分词器 ====================
print("Downloading model from ModelScope...")
snapshot_download(MODEL_NAME, cache_dir="./", revision="master")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(f"./{MODEL_NAME}", use_fast=False, trust_remote_code=True)

# Qwen2.5 的 pad_token 设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    f"./{MODEL_NAME}",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()

# ==================== 加载并处理训练数据 ====================
with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

train_dataset = [process_func(item) for item in raw_data]

# ==================== LoRA 配置 ====================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# ==================== 训练参数 ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,          # 0.5B 模型较小，显存够的话可以设为 4
    gradient_accumulation_steps=2,          # 保持总 batch_size = 8
    logging_steps=10,
    num_train_epochs=5,
    save_steps=200,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    report_to="none",
    bf16=True,
)

# ==================== SwanLab 回调 ====================
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-CoT",
    experiment_name="cot_finetune",
    config={
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "lora_r": 8,
        "learning_rate": 1e-4,
        "epochs": 5,
    }
)

# ==================== 训练器 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # 【优化】添加 pad_to_multiple_of=8 提升 GPU 效率
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
    callbacks=[swanlab_callback],
)

# ==================== 开始训练 ====================
trainer.train(resume_from_checkpoint=True)
swanlab.finish()