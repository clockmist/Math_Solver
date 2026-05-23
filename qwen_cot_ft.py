"""LoRA fine-tune Qwen2.5-0.5B-Instruct on CoT-formatted training data."""
import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab


def process_func(example):
    """
    将 CoT 格式数据集进行预处理。
    answer 字段已经包含完整的推理链（结尾为「答案：数字」）。
    """
    MAX_LENGTH = 768  # increased from 384 for CoT chains
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['answer']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master")

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2___5-0___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2___5-0___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

with open("train_cot.json", "r", encoding="utf-8") as file:
    train_data = json.load(file)
train_dataset = []
for d in train_data:
    train_dataset.append(process_func(d))
print(f"Training on {len(train_dataset)} CoT samples, MAX_LENGTH=768")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen_cot",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-CoT-fintune",
    experiment_name="Qwen/Qwen2.5-0.5B-Instruct-cot",
    config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "train_cot",
    }
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

swanlab.finish()
