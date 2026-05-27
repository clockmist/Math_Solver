"""
DPO (Direct Preference Optimization) 训练脚本
用于小学数学应用题自动解题任务

前置条件：
- 已完成SFT训练，有SFT的LoRA权重或合并后的模型
- 已构建DPO偏好数据（dpo_data.json），格式为 [{prompt, chosen, rejected}, ...]
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# =============================================================================
# 第一部分：配置与超参数
# =============================================================================

# 固定随机种子，保证实验可复现
set_seed(42)

# 路径配置
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct/"      # 原始预训练模型（HuggingFace仓库）
SFT_LORA_PATH = "./output/Qwen_CoT_v2/checkpoint-7325"        # SFT阶段训练好的LoRA权重目录
DPO_DATA_PATH = "./dpo_data.json"           # DPO偏好数据集路径
OUTPUT_DIR = "./qwen_dpo_output_2"            # DPO训练输出目录

# DPO关键超参数
BETA = 0.3                            # DPO温度系数，控制与参考模型的偏离程度
LEARNING_RATE = 5e-6                        # DPO学习率，比SFT低
NUM_EPOCHS = 2                              # 训练轮次
BATCH_SIZE = 1                              # 单卡batch size
GRAD_ACCUMULATION = 4                       # 梯度累积步数，等效batch=16
LORA_R = 16                                 # LoRA秩
LORA_ALPHA = 32                             # LoRA缩放系数
LORA_DROPOUT = 0.05                         # LoRA dropout

MAX_LENGTH = 768                            # 最大序列长度（prompt + response），增加到768
MAX_PROMPT_LENGTH = 512                     # prompt最大长度，增加到512

# 是否使用系统指令（需与SFT阶段完全一致）
USE_SYSTEM_INSTRUCTION = True               # 根据SFT实际情况设置
SYSTEM_MESSAGE = "你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"
# 如果SFT阶段没有使用系统指令，请将 USE_SYSTEM_INSTRUCTION 设为 False

# =============================================================================
# 第二部分：数据加载与预处理（关键修复：prompt格式与SFT一致）
# =============================================================================

def load_dpo_data(data_path):
    """加载DPO偏好数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"加载了 {len(raw_data)} 条DPO偏好对")
    return raw_data


def format_dpo_dataset(raw_data):
    formatted_data = []
    for item in raw_data:
        # 构建消息列表（不含 assistant 回复，因为 DPOTrainer 会自动拼接）
        if USE_SYSTEM_INSTRUCTION:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": item["prompt"]}
            ]
        else:
            messages = [{"role": "user", "content": item["prompt"]}]
        
        # chosen 和 rejected 分别作为 assistant 的回复（也是消息列表形式）
        chosen = [{"role": "assistant", "content": item["chosen"]}]
        rejected = [{"role": "assistant", "content": item["rejected"]}]
        
        formatted_data.append({
            "prompt": messages,          # 重要：这里传消息列表，不是字符串！
            "chosen": chosen,
            "rejected": rejected,
        })
    return Dataset.from_list(formatted_data)


# =============================================================================
# 第三部分：模型加载与准备
# =============================================================================

def load_sft_model_for_dpo(base_path, sft_lora_path):
    """
    加载SFT模型并准备DPO训练（策略模型）
    步骤：基座 → 加载SFT LoRA → 合并 → 添加DPO LoRA
    """
    print(f"正在加载基座模型: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"正在加载SFT LoRA权重: {sft_lora_path}")
    model = PeftModel.from_pretrained(base_model, sft_lora_path)

    # 合并SFT LoRA到基座，固化SFT知识
    print("正在合并SFT LoRA权重到基座...")
    model = model.merge_and_unload()

    # 为DPO添加新的LoRA（学习偏好差异）
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("DPO LoRA已添加，可训练参数统计：")
    model.print_trainable_parameters()
    return model


def load_reference_model(base_path, sft_lora_path):
    """
    加载参考模型（冻结），必须与策略模型有相同的SFT起点
    """
    print("正在加载参考模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(base_model, sft_lora_path)
    ref_model = ref_model.merge_and_unload()
    # 冻结所有参数
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("参考模型已加载并冻结")
    return ref_model


# =============================================================================
# 第四部分：训练配置与执行
# =============================================================================

def create_dpo_trainer(model, ref_model, tokenizer, dataset):
    # 将所有 DPO 和训练相关的参数都放入 DPOConfig
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=BETA,                         # DPO 超参数
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to = "none" 
    )

    # 关键修改：用 processing_class 替代 tokenizer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    return trainer


def main():
    """主训练流程"""
    print("=" * 60)
    print("开始DPO训练流程")
    print("=" * 60)

    # 1. 加载tokenizer并设置关键属性
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )
    # 设置pad_token（Qwen2.5通常使用eos_token作为pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # 【关键】left padding 对于decoder-only模型的生成至关重要，DPOTrainer内部会处理
    tokenizer.padding_side = "left"

    # 2. 加载并格式化数据
    print("\n[1/4] 加载DPO数据...")
    raw_data = load_dpo_data(DPO_DATA_PATH)
    dataset = format_dpo_dataset(raw_data)
    print(f"数据集大小: {len(dataset)} 条")

    # 3. 加载策略模型
    print("\n[2/4] 加载策略模型...")
    model = load_sft_model_for_dpo(BASE_MODEL_PATH, SFT_LORA_PATH)

    # 4. 加载参考模型
    print("\n[3/4] 加载参考模型...")
    ref_model = load_reference_model(BASE_MODEL_PATH, SFT_LORA_PATH)

    # 5. 创建Trainer并开始训练
    print("\n[4/4] 初始化DPOTrainer...")
    trainer = create_dpo_trainer(model, ref_model, tokenizer, dataset)

    print("\n开始训练！")
    print("=" * 60)
    trainer.train()

    # 6. 保存最终模型
    print("\n训练完成，保存模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n模型已保存到: {OUTPUT_DIR}")
    print("训练完成！")


if __name__ == "__main__":
    main()