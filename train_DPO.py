"""
增强版 DPO (Direct Preference Optimization) 训练脚本
用于小学数学应用题自动解题任务

前置条件：
- SFT LoRA已合并到基座模型，BASE_MODEL_PATH指向合并后的完整模型
- 已构建DPO偏好数据（dpo_data.json），格式为 [{prompt, chosen, rejected}, ...]
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# =============================================================================
# 第一部分：配置与超参数
# =============================================================================

set_seed(42)

# 路径配置
BASE_MODEL_PATH = "./qwen_sft_full/"  # 已合并SFT LoRA的完整模型
DPO_DATA_PATH = "./dpo_data.json"
OUTPUT_DIR = "./qwen_dpo_output_enhanced"
MERGED_MODEL_DIR = "./qwen_dpo_merged_final"

# DPO关键超参数
BETA = 0.3
LEARNING_RATE = 5e-6
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 512

USE_SYSTEM_INSTRUCTION = True
SYSTEM_MESSAGE = "你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"

# 监控配置
LOG_EVERY_N_STEPS = 10
EVAL_EVERY_N_STEPS = 100
SAVE_EVERY_N_STEPS = 200

# =============================================================================
# 第二部分：数据加载与预处理
# =============================================================================

def load_dpo_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"[数据加载] 原始数据条数: {len(raw_data)}")

    valid_count = 0
    issues = []
    for i, item in enumerate(raw_data):
        problems = []
        if "prompt" not in item:
            problems.append("缺少prompt字段")
        if "chosen" not in item:
            problems.append("缺少chosen字段")
        if "rejected" not in item:
            problems.append("缺少rejected字段")
        if "chosen" in item and "rejected" in item:
            if item["chosen"] == item["rejected"]:
                problems.append("chosen和rejected完全相同（致命错误！）")
        if "chosen" in item and len(item["chosen"].strip()) < 5:
            problems.append("chosen回复过短")
        if "rejected" in item and len(item["rejected"].strip()) < 5:
            problems.append("rejected回复过短")

        if problems:
            issues.append(f"  样本{i}: {', '.join(problems)}")
        else:
            valid_count += 1

    if issues:
        print(f"[警告] 发现 {len(issues)} 个有问题的样本（展示前10个）:")
        for issue in issues[:10]:
            print(issue)

    print(f"[数据质量] 有效样本: {valid_count}/{len(raw_data)}")
    print(f"{'='*60}\n")

    return raw_data


def format_dpo_dataset(raw_data):
    formatted_data = []
    for item in raw_data:
        if USE_SYSTEM_INSTRUCTION:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": item["prompt"]}
            ]
        else:
            messages = [{"role": "user", "content": item["prompt"]}]

        chosen = [{"role": "assistant", "content": item["chosen"]}]
        rejected = [{"role": "assistant", "content": item["rejected"]}]

        formatted_data.append({
            "prompt": messages,
            "chosen": chosen,
            "rejected": rejected,
        })

    dataset = Dataset.from_list(formatted_data)

    print(f"\n{'='*60}")
    print("[数据格式验证] 第一个样本:")
    sample = dataset[0]
    print(f"  Prompt: {sample['prompt'][-1]['content'][:80]}...")
    print(f"  Chosen: {sample['chosen'][0]['content'][:80]}...")
    print(f"  Rejected: {sample['rejected'][0]['content'][:80]}...")
    print(f"{'='*60}\n")

    return dataset


# =============================================================================
# 第三部分：自定义DPOTrainer（增强监控 + 修复中间评估bug）
# =============================================================================

class EnhancedDPOTrainer(DPOTrainer):
    def __init__(self, *args, raw_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_stats = defaultdict(list)
        self.raw_dataset = raw_dataset  # 保存原始数据集用于中间评估

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.state.global_step % LOG_EVERY_N_STEPS == 0 and self.state.global_step > 0:
            self._print_step_stats(inputs)

        if self.state.global_step % EVAL_EVERY_N_STEPS == 0 and self.state.global_step > 0:
            self._mid_evaluation(model)

        return (loss, outputs) if return_outputs else loss

    def _print_step_stats(self, inputs):
        print(f"\n{'='*60}")
        print(f"[Step {self.state.global_step}] DPO训练统计:")
        print(f"{'='*60}")

        # 从DPOTrainer的metrics中获取关键指标
        if hasattr(self, 'metrics') and self.metrics:
            for k, v in self.metrics.items():
                if isinstance(v, (int, float)) and not k.startswith('_'):
                    print(f"  {k}: {v:.6f}")

        # 序列长度统计
        if "chosen_input_ids" in inputs:
            chosen_lens = [len(ids[ids != self.processing_class.pad_token_id]) 
                          for ids in inputs["chosen_input_ids"]]
            rejected_lens = [len(ids[ids != self.processing_class.pad_token_id]) 
                            for ids in inputs["rejected_input_ids"]]
            print(f"  Chosen长度: 均值={np.mean(chosen_lens):.0f}, 范围=[{min(chosen_lens)}, {max(chosen_lens)}]")
            print(f"  Rejected长度: 均值={np.mean(rejected_lens):.0f}, 范围=[{min(rejected_lens)}, {max(rejected_lens)}]")

        # 学习率
        if hasattr(self, "optimizer") and self.optimizer:
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"  学习率: {current_lr:.2e}")

        print(f"{'='*60}\n")

    def _mid_evaluation(self, model):
        """中间评估：从原始数据集中抽样，检查模型偏好判断"""
        if self.raw_dataset is None or len(self.raw_dataset) == 0:
            print("[中间评估] 原始数据集不可用，跳过")
            return

        print(f"\n{'='*60}")
        print(f"[Step {self.state.global_step}] 中间评估（抽样3条）:")
        print(f"{'='*60}")

        eval_indices = np.random.choice(len(self.raw_dataset), min(3, len(self.raw_dataset)), replace=False)

        correct_prefs = 0
        for idx in eval_indices:
            sample = self.raw_dataset[int(idx)]

            try:
                with torch.no_grad():
                    # 构建完整对话文本
                    chosen_msgs = sample["prompt"] + sample["chosen"]
                    rejected_msgs = sample["prompt"] + sample["rejected"]

                    chosen_text = self.processing_class.apply_chat_template(
                        chosen_msgs, tokenize=False, add_generation_prompt=False
                    )
                    rejected_text = self.processing_class.apply_chat_template(
                        rejected_msgs, tokenize=False, add_generation_prompt=False
                    )

                    chosen_ids = self.processing_class(
                        chosen_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
                    ).to(model.device)
                    rejected_ids = self.processing_class(
                        rejected_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
                    ).to(model.device)

                    chosen_outputs = model(**chosen_ids, use_cache=False)
                    rejected_outputs = model(**rejected_ids, use_cache=False)

                    # 计算log prob（用loss的负数近似）
                    chosen_logprob = -chosen_outputs.loss.item() if hasattr(chosen_outputs, 'loss') and chosen_outputs.loss is not None else 0
                    rejected_logprob = -rejected_outputs.loss.item() if hasattr(rejected_outputs, 'loss') and rejected_outputs.loss is not None else 0

                    is_correct = chosen_logprob > rejected_logprob
                    if is_correct:
                        correct_prefs += 1

                    prompt_preview = sample["prompt"][-1]["content"][:30] if isinstance(sample["prompt"], list) else str(sample["prompt"])[:30]
                    print(f"  样本{idx} [{prompt_preview}...]: chosen={chosen_logprob:.2f}, rejected={rejected_logprob:.2f} {'✓' if is_correct else '✗'}")
            except Exception as e:
                print(f"  样本{idx}: 评估失败 ({str(e)[:50]})")

        acc = correct_prefs / len(eval_indices) if len(eval_indices) > 0 else 0
        print(f"  偏好判断准确率: {acc*100:.1f}% ({correct_prefs}/{len(eval_indices)})")
        if acc < 0.6:
            print(f"  ⚠️ 警告: 准确率偏低，可能数据标签方向有问题或模型未学好")
        print(f"{'='*60}\n")

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if hasattr(loss, 'item'):
            self.step_stats["loss"].append(loss.item())
        return loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self.step_stats["loss"]:
            recent = self.step_stats["loss"][-LOG_EVERY_N_STEPS:]
            if len(recent) >= 5:
                if all(l > recent[0] for l in recent[-5:]):
                    logs["WARNING"] = "Loss持续上升！"
                if any(np.isnan(l) or np.isinf(l) for l in recent):
                    logs["WARNING"] = "NaN/Inf detected!"
        super().log(logs, start_time)


# =============================================================================
# 第四部分：模型加载
# =============================================================================

def load_policy_model(base_path):
    print(f"\n{'='*60}")
    print(f"[模型加载] 策略模型（Policy Model）")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  设备: {next(model.parameters()).device}")

    if isinstance(model, PeftModel):
        print(f"  检测到PeftModel，自动合并...")
        model = model.merge_and_unload()
    else:
        print(f"  ✓ 已是合并状态")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"{'='*60}\n")
    return model


def load_reference_model(base_path):
    print(f"\n{'='*60}")
    print(f"[模型加载] 参考模型（Reference Model - 冻结）")
    print(f"{'='*60}")

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if isinstance(ref_model, PeftModel):
        ref_model = ref_model.merge_and_unload()

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in ref_model.parameters())
    frozen = sum(p.numel() for p in ref_model.parameters() if not p.requires_grad)
    print(f"  参数量: {total/1e6:.1f}M, 冻结: {frozen/1e6:.1f}M ({100*frozen/total:.1f}%)")
    print(f"{'='*60}\n")

    return ref_model


# =============================================================================
# 第五部分：训练配置与执行
# =============================================================================

def create_dpo_trainer(model, ref_model, tokenizer, dataset, raw_dataset):
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=LOG_EVERY_N_STEPS,
        save_steps=SAVE_EVERY_N_STEPS,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        logging_first_step=True,
    )

    trainer = EnhancedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        raw_dataset=raw_dataset,  # 传入原始数据集用于中间评估
    )
    return trainer


def save_merged_model(model, tokenizer, output_path):
    print(f"\n{'='*60}")
    print(f"[模型保存] 合并后的完整模型")
    print(f"{'='*60}")

    os.makedirs(output_path, exist_ok=True)

    if hasattr(model, 'merge_and_unload'):
        merged = model.merge_and_unload()
    else:
        merged = model

    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    total_size = sum(os.path.getsize(os.path.join(dp, f)) 
                     for dp, dn, filenames in os.walk(output_path) for f in filenames)
    print(f"  保存到: {output_path}")
    print(f"  总大小: {total_size/1024/1024:.1f} MB")
    print(f"{'='*60}\n")

    return merged


def print_training_summary(trainer):
    print(f"\n{'='*60}")
    print(f"[训练摘要]")
    print(f"{'='*60}")

    if trainer.step_stats.get("loss"):
        losses = trainer.step_stats["loss"]
        print(f"  总步数: {len(losses)}")
        print(f"  初始loss: {losses[0]:.6f}")
        print(f"  最终loss: {losses[-1]:.6f}")
        print(f"  最小loss: {min(losses):.6f}")
        print(f"  趋势: {'↓下降' if losses[-1] < losses[0] else '↑上升/持平'}")
        if any(np.isnan(l) for l in losses):
            print(f"  ⚠️ 出现NaN")

    print(f"{'='*60}\n")


def main():
    print(f"\n{'='*60}")
    print(f"DPO训练（SFT已合并到基座）")
    print(f"{'='*60}")
    print(f"模型: {BASE_MODEL_PATH}")
    print(f"数据: {DPO_DATA_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"合并: {MERGED_MODEL_DIR}")
    print(f"{'='*60}\n")

    # 1. Tokenizer
    print("[1/5] 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"  pad={tokenizer.pad_token}, vocab={len(tokenizer)}")

    # 2. 数据
    print("\n[2/5] 加载DPO数据...")
    raw_data = load_dpo_data(DPO_DATA_PATH)
    dataset = format_dpo_dataset(raw_data)
    print(f"  数据集: {len(dataset)} 条")

    # 3. 策略模型
    print("\n[3/5] 加载策略模型...")
    model = load_policy_model(BASE_MODEL_PATH)

    # 4. 参考模型
    print("\n[4/5] 加载参考模型...")
    ref_model = load_reference_model(BASE_MODEL_PATH)

    # 5. 训练
    print("\n[5/5] 初始化DPOTrainer...")
    trainer = create_dpo_trainer(model, ref_model, tokenizer, dataset, raw_dataset=dataset)

    print(f"\n{'='*60}")
    print(f"开始训练！")
    print(f"{'='*60}")
    trainer.train()

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}")

    print_training_summary(trainer)

    print("\n保存DPO LoRA...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n保存合并模型...")
    save_merged_model(trainer.model, tokenizer, MERGED_MODEL_DIR)

    print(f"\n{'='*60}")
    print(f"完成！DPO LoRA: {OUTPUT_DIR}, 合并: {MERGED_MODEL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()