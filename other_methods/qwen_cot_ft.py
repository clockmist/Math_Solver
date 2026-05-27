import json
import re
import torch
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab

# ==================== 配置参数 ====================
MAX_LENGTH = 512
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output/Qwen_CoT_v2"
TRAIN_JSON_PATH = "train_cot.json"

# ==================== 数据处理函数 ====================
def process_func(example):
    """
    将带思维链的数据预处理为模型输入格式。
    【关键修改】确保 eos_token 是 response 的绝对最后一个 token，
    让模型学会"答案结束 → 立即停止"的条件反射。
    """
    instruction = example.get('instruction', '').strip()
    question = example['question'].strip()
    answer = example['answer'].strip()

    # 1. 构造 System + User + Assistant 头部（Prompt）
    if instruction:
        prompt_text = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        # 无 instruction 时省略 system 字段，避免空 system message
        prompt_text = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    prompt = tokenizer(prompt_text, add_special_tokens=False)

    # 2. 构造 Response：先对 answer 单独 tokenize，再手动追加 eos_token_id
    # 【关键】不用字符串拼接 <|im_end|>，避免引入多余字符（如 \n）
    answer_tokens = tokenizer(answer, add_special_tokens=False)

    # 手动追加 eos_token_id，确保它是 response 的最后一个 token
    response_input_ids = answer_tokens["input_ids"] + [tokenizer.eos_token_id]
    response_attention_mask = answer_tokens["attention_mask"] + [1]

    # 3. 拼接 input_ids 和 attention_mask
    input_ids = prompt["input_ids"] + response_input_ids
    attention_mask = prompt["attention_mask"] + response_attention_mask

    # 4. 构造 labels：Prompt 部分忽略（-100），Response 部分为真实 token（含 eos）
    labels = [-100] * len(prompt["input_ids"]) + response_input_ids

    # 5. 截断超长序列（从尾部截断，优先保留前面的 prompt 和后面的 answer）
    if len(input_ids) > MAX_LENGTH:
        # 如果超长，优先截断 answer 部分，但至少要保留 eos_token
        excess = len(input_ids) - MAX_LENGTH
        if excess < len(response_input_ids) - 1:  # 至少留1个token给eos
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        else:
            # 如果超长太多，从前面截断（但 prompt 不能截断太多）
            input_ids = input_ids[-MAX_LENGTH:]
            attention_mask = attention_mask[-MAX_LENGTH:]
            labels = labels[-MAX_LENGTH:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ==================== 加载模型和分词器 ====================
print("Downloading model from ModelScope...")
snapshot_download(MODEL_NAME, cache_dir="./", revision="master")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    f"./{MODEL_NAME}", 
    use_fast=False, 
    trust_remote_code=True
)

# Qwen2.5 的 pad_token 设置（与 eos_token 相同是正常的）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    f"./{MODEL_NAME}",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.enable_input_require_grads()

# ==================== 加载并处理训练数据 ====================
with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 数据格式校验
for i, item in enumerate(raw_data):
    if "question" not in item or "answer" not in item:
        raise ValueError(f"第 {i} 条数据缺少 question 或 answer 字段")
    # 确保 answer 末尾没有多余的 <|im_end|> 或换行
    item["answer"] = item["answer"].strip()

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=200,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    report_to="none",
    bf16=True,
    # 【新增】确保保存 tokenizer，方便后续推理时复现相同格式
    save_safetensors=True,
)

# ==================== SwanLab 回调 ====================
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-0.5B-CoT",
    experiment_name="cot_finetune_v2",  # 改名区分版本
    config={
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "lora_r": 8,
        "learning_rate": 1e-4,
        "epochs": 5,
        "fix": "eos_token_as_last_token",  # 记录修改点
    }
)

# ==================== 训练器 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True, 
        pad_to_multiple_of=8
    ),
    callbacks=[swanlab_callback],
)

# ==================== 开始训练 ====================
trainer.train()
swanlab.finish()

# ==================== 保存最终模型 ====================
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"模型已保存到: {OUTPUT_DIR}")