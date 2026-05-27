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
# 在LLM训练中非常重要，因为不同的随机初始化/采样顺序会导致结果差异很大
set_seed(42)

# 路径配置
BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B"      # 原始预训练模型（HuggingFace仓库）
SFT_LORA_PATH = "./sft_lora_weights"        # SFT阶段训练好的LoRA权重目录
DPO_DATA_PATH = "./dpo_data.json"           # DPO偏好数据集路径
OUTPUT_DIR = "./qwen_dpo_output"            # DPO训练输出目录

# DPO关键超参数
BETA = 0.1                                  # DPO温度系数，控制与参考模型的偏离程度
                                            # 越小越激进（偏离参考模型远），越大越保守
                                            # 数学应用题建议0.1-0.3，从0.1开始调

LEARNING_RATE = 5e-6                        # DPO学习率，通常比SFT低（SFT约1e-4到5e-5）
                                            # 因为DPO是在SFT基础上微调，步长太大会破坏已有知识

NUM_EPOCHS = 3                              # DPO通常不需要太多轮次，2-5轮即可
                                            # 偏好优化比SFT更容易过拟合，太多轮次会导致模型输出退化

BATCH_SIZE = 4                              # 单卡batch size，根据显存调整
GRAD_ACCUMULATION = 4                       # 梯度累积步数，等效batch = 4 * 4 = 16
                                            # 等效batch size越大，梯度估计越稳定，DPO训练越稳定

LORA_R = 16                                 # LoRA秩，DPO可以比SFT稍大（SFT可能用8）
                                            # 因为DPO需要建模"偏好差异"，比SFT的"模仿学习"需要更多表达能力
LORA_ALPHA = 32                             # LoRA缩放系数，通常 = 2 * r
LORA_DROPOUT = 0.05                         # 防止LoRA过拟合

MAX_LENGTH = 512                            # 最大序列长度（prompt + response）
MAX_PROMPT_LENGTH = 256                     # prompt最大长度，超过会被截断

# =============================================================================
# 第二部分：数据加载与预处理
# =============================================================================

def load_dpo_data(data_path):
    """
    加载DPO偏好数据

    数据格式要求：
    [
        {
            "prompt": "题目文本",
            "chosen": "正确的CoT推理过程",
            "rejected": "错误的CoT推理过程"
        },
        ...
    ]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"加载了 {len(raw_data)} 条DPO偏好对")
    return raw_data


def format_dpo_dataset(raw_data):
    """
    将原始数据格式化为DPOTrainer期望的Dataset格式

    DPOTrainer要求每个样本包含三个字段：
    - prompt: 问题描述（模型输入）
    - chosen: 优选回答（正确CoT）
    - rejected: 不优选回答（错误CoT）

    注意：不需要手动添加模板（如"解答："），DPOTrainer会通过tokenizer的
    chat_template自动处理。但如果你的SFT训练用了特定模板，这里要保持一致。
    """
    formatted_data = []

    for item in raw_data:
        # 保持与SFT训练时一致的prompt格式
        # 如果SFT时用了特定system message，这里必须完全一致
        prompt = f"这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。\n题目：{item['prompt']}\n解答："

        formatted_data.append({
            "prompt": prompt,
            "chosen": item["chosen"],      # 正确推理：步骤完整、计算正确、答案正确
            "rejected": item["rejected"]   # 错误推理：步骤跳步/计算错误/理解错误
        })

    # 转换为HuggingFace Dataset对象，DPOTrainer只接受这种格式
    dataset = Dataset.from_list(formatted_data)
    return dataset


# =============================================================================
# 第三部分：模型加载与准备（核心部分）
# =============================================================================

def load_sft_model_for_dpo(base_path, sft_lora_path):
    """
    加载SFT模型并准备DPO训练

    关键逻辑：
    1. 加载原始基座模型 Qwen-0.5B
    2. 加载SFT的LoRA权重
    3. merge_and_unload()：将SFT学到的知识"固化"到基座参数中
    4. 在合并后的模型上添加新的LoRA用于DPO训练

    为什么必须merge？
    - DPO需要两个模型：策略模型（训练）和参考模型（冻结）
    - 两者必须从同一个"SFT后"的起点出发
    - merge后SFT知识变成基座的一部分，DPO的LoRA只负责学习"偏好差异"
    """

    # 步骤1：加载原始预训练模型（FP16节省显存，trust_remote_code用于Qwen）
    print(f"正在加载基座模型: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,          # 半精度训练，减少显存占用
        device_map="auto",                  # 自动分配层到GPU/CPU，单卡时全放GPU
        trust_remote_code=True,             # Qwen系列需要此参数加载自定义架构
    )

    # 步骤2：加载SFT阶段训练的LoRA权重
    # 此时模型结构：Qwen-0.5B + LoRA_A（SFT的LoRA）
    print(f"正在加载SFT LoRA权重: {sft_lora_path}")
    model = PeftModel.from_pretrained(base_model, sft_lora_path)

    # 步骤3：合并LoRA权重到基座
    # merge_and_unload() 做两件事：
    # 1. 计算 W_merged = W_base + A×B（LoRA的低秩分解还原）
    # 2. 返回一个普通的AutoModelForCausalLM（不再是PeftModel）
    # 结果：模型参数包含了SFT学到的知识，但结构上就是普通模型
    print("正在合并SFT LoRA权重到基座...")
    model = model.merge_and_unload()

    # 步骤4：为DPO训练添加新的LoRA（LoRA_B）
    # 这个LoRA只会在DPO训练中被更新，学习"好回答vs坏回答"的偏好差异
    lora_config = LoraConfig(
        r=LORA_R,                           # LoRA秩：控制新增参数的数量
        lora_alpha=LORA_ALPHA,              # 缩放系数：控制LoRA对输出的影响强度
        target_modules=[                    # 目标模块：Qwen2.5的注意力层和MLP层
            "q_proj", "k_proj", "v_proj", "o_proj",   # 自注意力四个投影矩阵
            "gate_proj", "up_proj", "down_proj"       # MLP层的三个投影矩阵
        ],
        lora_dropout=LORA_DROPOUT,          # Dropout：防止DPO过拟合到特定偏好对
        bias="none",                        # 不训练偏置项，减少参数量
        task_type="CAUSAL_LM",              # 任务类型：因果语言模型（生成式）
    )

    # get_peft_model() 将普通模型包装为PeftModel，添加可训练的LoRA_B
    model = get_peft_model(model, lora_config)
    print("DPO LoRA已添加，可训练参数统计：")
    model.print_trainable_parameters()      # 打印LoRA参数量，通常只占原模型的0.1%-1%

    return model


def load_reference_model(base_path, sft_lora_path):
    """
    加载参考模型（Reference Model）

    参考模型的作用：
    - 在DPO损失中作为"基准"，计算策略模型相对于基准的偏好概率变化
    - 必须冻结（不更新参数），保证训练稳定性
    - 必须与策略模型有相同的SFT起点，即同样的merge后模型

    为什么需要参考模型？
    DPO的损失函数：
    loss = -log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))

    其中π_ref就是参考模型的输出概率，提供"相对偏好"的基准。
    没有参考模型，DPO退化为普通的对比学习，失去"在SFT基础上优化"的意义。
    """

    print("正在加载参考模型...")
    # 同样的加载+合并流程
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    ref_model = PeftModel.from_pretrained(base_model, sft_lora_path)
    ref_model = ref_model.merge_and_unload()

    # 冻结所有参数：参考模型不参与梯度更新
    ref_model.eval()                        # 设置为评估模式（关闭Dropout等）
    for param in ref_model.parameters():
        param.requires_grad = False         # 禁止计算梯度，节省显存和计算

    print("参考模型已加载并冻结")
    return ref_model


# =============================================================================
# 第四部分：训练配置与执行
# =============================================================================

def create_dpo_trainer(model, ref_model, tokenizer, dataset):
    """
    创建DPOTrainer

    DPOTrainer是TRL库封装好的训练器，自动处理：
    - 对chosen和rejected分别计算log概率
    - 应用DPO损失函数
    - 管理参考模型的前向传播（不计算梯度）
    """

    # DPOConfig：TRL库专用的配置类，继承自TrainingArguments
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,              # 模型和日志保存目录

        # 训练基本参数
        num_train_epochs=NUM_EPOCHS,        # 总训练轮次
        per_device_train_batch_size=BATCH_SIZE,     # 单卡batch size
        gradient_accumulation_steps=GRAD_ACCUMULATION,  # 梯度累积步数

        # 优化器参数
        learning_rate=LEARNING_RATE,        # 学习率
        lr_scheduler_type="cosine",         # 余弦退火调度器，训练后期自动降低LR
        warmup_ratio=0.1,                   # 前10%步数线性warmup，防止初期训练不稳定

        # DPO特有参数
        beta=BETA,                          # DPO温度系数（最关键的超参数！）
                                            # 控制策略模型可以偏离参考模型的程度
                                            # β→0：可以任意偏离；β→∞：必须紧贴参考模型

        # 长度限制
        max_length=MAX_LENGTH,              # 整个序列（prompt+response）的最大长度
        max_prompt_length=MAX_PROMPT_LENGTH, # prompt部分的最大长度

        # 保存与日志
        logging_steps=10,                   # 每10步打印一次日志
        save_steps=500,                     # 每500步保存一个checkpoint
        save_total_limit=2,                 # 最多保留2个checkpoint，防止磁盘占满

        # 混合精度训练
        bf16=True,                          # 使用bfloat16（比fp16更稳定，需Ampere架构GPU）
                                            # 如果GPU不支持bf16，改为fp16=True

        # 其他优化
        remove_unused_columns=False,        # DPO需要保留prompt/chosen/rejected三列
                                            # 设为False防止HuggingFace自动删除"多余"列
        gradient_checkpointing=True,        # 梯度检查点：用时间换显存，训练大batch时必需
    )

    # 初始化DPOTrainer
    # 参数说明：
    # - model: 策略模型（带LoRA_B，可训练）
    # - ref_model: 参考模型（冻结的SFT模型）
    # - args: DPOConfig配置
    # - train_dataset: 偏好数据集
    # - tokenizer: 用于编码文本
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # 可选：数据整理函数，DPOTrainer内部已封装好
        # 它会自动将chosen和rejected拼接成合适的tensor格式
    )

    return trainer


# =============================================================================
# 第五部分：主函数
# =============================================================================

def main():
    """主训练流程"""

    # 1. 加载tokenizer
    # Qwen的tokenizer需要trust_remote_code，且需要设置pad_token
    print("=" * 60)
    print("开始DPO训练流程")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        pad_token="<|endoftext|>"           # Qwen的pad_token默认就是eos_token
    )
    tokenizer.pad_token = tokenizer.eos_token  # 确保pad_token已设置

    # 2. 加载并格式化数据
    print("\n[1/4] 加载DPO数据...")
    raw_data = load_dpo_data(DPO_DATA_PATH)
    dataset = format_dpo_dataset(raw_data)
    print(f"数据集大小: {len(dataset)} 条")

    # 3. 加载策略模型（带DPO LoRA）
    print("\n[2/4] 加载策略模型...")
    model = load_sft_model_for_dpo(BASE_MODEL_PATH, SFT_LORA_PATH)

    # 4. 加载参考模型（冻结）
    print("\n[3/4] 加载参考模型...")
    ref_model = load_reference_model(BASE_MODEL_PATH, SFT_LORA_PATH)

    # 5. 创建Trainer并开始训练
    print("\n[4/4] 初始化DPOTrainer...")
    trainer = create_dpo_trainer(model, ref_model, tokenizer, dataset)

    print("\n开始训练！")
    print("=" * 60)
    trainer.train()                         # 开始训练循环

    # 6. 保存最终模型
    print("\n训练完成，保存模型...")
    trainer.save_model(OUTPUT_DIR)          # 保存DPO的LoRA权重
    tokenizer.save_pretrained(OUTPUT_DIR)   # 保存tokenizer配置

    print(f"\n模型已保存到: {OUTPUT_DIR}")
    print("训练完成！")


if __name__ == "__main__":
    main()