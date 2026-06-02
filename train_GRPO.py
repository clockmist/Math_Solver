"""
train_GRPO.py — GRPO 训练脚本（加速版）
基于原 train_GRPO_fast.py 的思路，仅保留两个最有效的简单优化：

1. 批量生成：一次 model.generate() 生成全部 G 个 completion（vs 逐条 G 次）—— 最大加速
2. 去除 output_scores：生成后通过一次 forward pass 计算 old_log_probs（省 300MB+ 显存）
3. 预 tokenize 所有 prompt：加载数据时一次性完成，避免训练中重复 tokenize
"""

import json
import os
import re
import shutil
import glob
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# =============================================================================
# 第一部分：配置与超参数
# =============================================================================

set_seed(42)

# 路径配置
BASE_MODEL_PATH = "./qwen_sft_full/"
TRAIN_DATA_PATH = "./train.json"
OUTPUT_DIR = "./qwen_grpo_output"
MERGED_MODEL_DIR = "./qwen_grpo_merged_final"
CHECKPOINT_DIR = "./qwen_grpo_checkpoints"
BEST_CHECKPOINT_DIR = "./qwen_grpo_best"   # 验证集准确率最高的检查点

# GRPO关键超参数
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3
GRAD_ACCUMULATION = 2          # mini-batch大小（用grad accumulation模拟）

# GRPO特有参数
NUM_GENERATIONS = 4            # 每个问题生成G个completion
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.2              # 适当提高温度，增加生成多样性
TOP_P = 0.95
TOP_K = 50

# PPO参数
PPO_EPOCHS = 2                 # 每个mini-batch上做几轮PPO更新（核心！）
EPSILON = 0.2                  # PPO clipping范围
BETA_KL = 0.05                 # KL惩罚系数（增大防止重复退化）

# LoRA参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 系统指令（与SFT训练保持一致）
SYSTEM_MESSAGE = "你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"

# 监控配置
LOG_EVERY_N_STEPS = 20
SAVE_EVERY_N_STEPS = 200
EVAL_EVERY_N_STEPS = 200        # 每隔多少步在验证集上评估
VAL_DATA_PATH = "small_val.json"  # 验证集路径
KEEP_LAST_N_CHECKPOINTS = 2
RESUME_FROM_CHECKPOINT = None   # ← 从头训

# =============================================================================
# 第二部分：答案提取与奖励函数（同原版）
# =============================================================================

def parse_number(s):
    s = s.strip().replace(" ", "")
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    if s.startswith("-") or s.startswith("+"):
        s = s[1:]
    try:
        return sign * float(s)
    except ValueError:
        pass
    mixed_match = re.match(r"^(\d+)[_又](\d+)/(\d+)$", s)
    if mixed_match:
        whole, num, den = float(mixed_match.group(1)), float(mixed_match.group(2)), float(mixed_match.group(3))
        if den != 0:
            return sign * (whole + num / den)
    frac_match = re.match(r"^(\d+)/(\d+)$", s)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        if den != 0:
            return sign * (num / den)
    return None


def extract_answer(text):
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    patterns = [
        r"答案[：:]\s*(.+?)(?:\n|$)",
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"答案[是为]\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip().rstrip("。，,.;;")
            num = parse_number(candidate)
            if num is not None:
                return num
            num_match = re.search(r"(-?\d+\.?\d*(?:/\d+)?)", candidate)
            if num_match:
                num = parse_number(num_match.group(1))
                if num is not None:
                    return num
    for line in reversed(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"答案[：:]\s*(.+)", line)
        if m:
            num = parse_number(m.group(1).strip().rstrip("。,;；"))
            if num is not None:
                return num
    frac_matches = re.findall(r"\d+/\d+", text)
    if frac_matches:
        num = parse_number(frac_matches[-1].strip())
        if num is not None:
            return num
    num_matches = re.findall(r"-?\d+\.?\d*", text)
    if num_matches:
        try:
            return float(num_matches[-1])
        except ValueError:
            pass
    return None


def compute_mixed_reward(generated_text, correct_answer, tolerance=1e-6):
    extracted = extract_answer(generated_text)
    correct_answer_val = parse_number(str(correct_answer).strip())

    if extracted is not None and correct_answer_val is not None:
        if abs(correct_answer_val) < 1e-8:
            correctness = 1.0 if abs(extracted) < tolerance else 0.0
        elif abs(extracted - correct_answer_val) < tolerance:
            correctness = 1.0
        else:
            rel_error = abs(extracted - correct_answer_val) / (abs(correct_answer_val) + 1e-8)
            correctness = 0.3 / (1.0 + rel_error)
    else:
        correctness = 0.0

    format_reward = 0.0
    if "已知条件" in generated_text: format_reward += 0.025
    if "求解目标" in generated_text: format_reward += 0.025
    if "计算过程" in generated_text: format_reward += 0.025
    if "验证" in generated_text or "检验" in generated_text: format_reward += 0.025
    if re.search(r"答案[：:]", generated_text): format_reward += 0.02

    text_len = len(generated_text)
    if text_len < 30:       len_bonus = -0.05
    elif text_len < 50:     len_bonus = -0.02
    elif 60 <= text_len <= 600: len_bonus = 0.02
    elif text_len > 1000:   len_bonus = -0.05
    else:                   len_bonus = 0.0

    # ---- 重复惩罚：检测退化重复（如 "求求求..." 或反复重复同一句话） ----
    rep_penalty = 0.0
    # 连续字符重复检测（如 "求求求求求..."）
    if len(generated_text) > 20:
        max_char_run = max(len(m.group()) for m in re.finditer(r'(.)\1{4,}', generated_text)) if re.search(r'(.)\1{4,}', generated_text) else 0
        if max_char_run > 10:
            rep_penalty = -0.5
        elif max_char_run > 6:
            rep_penalty = -0.2
    # n-gram 重复检测（同一段话反复出现）
    if rep_penalty == 0.0 and len(generated_text) > 50:
        words = generated_text.split()
        if len(words) > 10:
            mid = len(words) // 2
            front = ' '.join(words[:mid])
            back = ' '.join(words[mid:])
            # 如果后半段几乎等于前半段 → 严重重复
            if front[:30] == back[:30] or (len(front) > 60 and front[:60] in back):
                rep_penalty = -0.3

    total = correctness + format_reward + len_bonus + rep_penalty
    return max(0.0, min(1.0, total)), extracted


# =============================================================================
# 第三部分：数据加载（加速：预 tokenize）
# =============================================================================

def load_and_prepare_data(data_path, tokenizer):
    """加载数据并预 tokenize 所有 prompt（仅一次，训练中免重复 tokenize）"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\n[数据] 加载 {len(raw_data)} 条样本，预 tokenize 中...")

    prepared = []
    for item in raw_data:
        if "question" not in item or "answer" not in item:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": item["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids  # [1, L]

        prepared.append({
            "prompt_ids": prompt_ids,
            "prompt_len": prompt_ids.shape[1],
            "correct_answer": item["answer"],
            "question": item["question"],
        })

    print(f"[数据] 有效样本: {len(prepared)}")
    return prepared


# =============================================================================
# 第四部分：模型加载
# =============================================================================

def load_policy_model(base_path):
    print(f"\n[模型] 加载策略模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_reference_model(base_path):
    print(f"\n[模型] 加载参考模型（冻结）...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if isinstance(ref_model, PeftModel):
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


# =============================================================================
# 第五部分：生成与Log Probability计算（加速核心）
# =============================================================================

@torch.no_grad()
def generate_batch(model, tokenizer, prompt_ids, num_generations,
                   max_new_tokens, temperature, top_p, top_k):
    """
    【加速点1】一次 model.generate() 生成 G 个 completion（vs 原版逐条 G 次）。
    """
    model.eval()
    prompt_len = prompt_ids.shape[1]
    device = prompt_ids.device

    # 批量生成
    batch_prompt = prompt_ids.repeat(num_generations, 1)
    gen_outputs = model.generate(
        batch_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        repetition_penalty=1.1,           # 防止训练中产生重复退化
        no_repeat_ngram_size=3,           # 禁止 3-gram 重复
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    completions = []
    completion_ids_list = []
    for g in range(num_generations):
        gen_ids = gen_outputs.sequences[g][prompt_len:]
        completion_ids_list.append(gen_ids)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions.append(text)

    # 批量 forward pass 获取 old_log_probs（代替 output_scores，省显存）
    max_comp_len = max(len(ids) for ids in completion_ids_list)
    if max_comp_len == 0:
        model.train()
        return completions, completion_ids_list, [torch.tensor([], dtype=torch.float32)] * num_generations

    total_len = prompt_len + max_comp_len
    batched_inputs = []
    for g in range(num_generations):
        comp_ids = completion_ids_list[g]
        full = torch.cat([prompt_ids[0], comp_ids.to(device)])
        if len(full) < total_len:
            pad = torch.full((total_len - len(full),), tokenizer.pad_token_id, device=device)
            full = torch.cat([full, pad])
        batched_inputs.append(full)

    batched_input = torch.stack(batched_inputs)
    attention_mask = (batched_input != tokenizer.pad_token_id).long()

    fwd_outputs = model(batched_input, attention_mask=attention_mask)
    logits = fwd_outputs.logits  # [G, total_len, vocab]

    old_log_probs_list = []
    for g in range(num_generations):
        comp_len = len(completion_ids_list[g])
        if comp_len == 0:
            old_log_probs_list.append(torch.tensor([], dtype=torch.float32))
            continue
        comp_logits = logits[g, prompt_len-1:prompt_len-1+comp_len, :]
        log_probs = F.log_softmax(comp_logits, dim=-1)
        token_lp = log_probs[torch.arange(comp_len, device=device),
                             completion_ids_list[g].to(device)]
        old_log_probs_list.append(token_lp.cpu())

    model.train()
    return completions, completion_ids_list, old_log_probs_list


def compute_token_log_probs(model, prompt_ids, completion_ids):
    """前向传播计算 per-token log probability。返回 shape [T] 的 tensor。"""
    full_ids = torch.cat([prompt_ids[0], completion_ids.to(model.device)]).unsqueeze(0)
    outputs = model(full_ids, use_cache=False)
    logits = outputs.logits[:, prompt_ids.shape[1] - 1 : -1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, torch.arange(log_probs.shape[1]), completion_ids.to(model.device)]
    return token_log_probs


# =============================================================================
# 第六部分：GRPO损失计算（同原版）
# =============================================================================

def compute_grpo_loss_per_token(model, ref_model, prompt_ids, completion_ids,
                                 old_log_probs, ref_log_probs, advantage, epsilon, beta_kl):
    if len(completion_ids) == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True), {}

    new_log_probs = compute_token_log_probs(model, prompt_ids, completion_ids)

    if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
        return torch.tensor(0.0, device=model.device, requires_grad=True), {"skipped": True}

    ratio = torch.exp(new_log_probs - old_log_probs.to(model.device))
    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
        return torch.tensor(0.0, device=model.device, requires_grad=True), {"skipped": True}

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
    ppo_loss_per_token = -torch.min(surr1, surr2)
    kl_per_token = new_log_probs - ref_log_probs.to(model.device)
    loss_per_token = ppo_loss_per_token + beta_kl * kl_per_token
    loss = loss_per_token.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=model.device, requires_grad=True), {"skipped": True}

    metrics = {
        "ppo_loss": ppo_loss_per_token.mean().item(),
        "kl": kl_per_token.mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_max": ratio.max().item(),
    }
    return loss, metrics


# =============================================================================
# 第七部分：断点保存与加载（仅保留最近 2 个）
# =============================================================================

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=KEEP_LAST_N_CHECKPOINTS):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*"))
    if len(checkpoints) <= keep_last_n:
        return
    def get_step(p):
        try:
            return int(os.path.basename(p).split("-")[-1])
        except ValueError:
            return 0
    for ckpt_path in sorted(checkpoints, key=get_step)[:-keep_last_n]:
        try:
            shutil.rmtree(ckpt_path)
            print(f"  [清理] 删除旧检查点: {os.path.basename(ckpt_path)}")
        except Exception as e:
            print(f"  [警告] 删除失败 {ckpt_path}: {e}")


def save_checkpoint(model, optimizer, epoch, global_step, step_stats, output_dir):
    checkpoint_path = os.path.join(output_dir, f"checkpoint-step-{global_step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    torch.save({
        "epoch": epoch, "global_step": global_step,
        "step_stats": dict(step_stats),
    }, os.path.join(checkpoint_path, "training_state.pt"))
    print(f"\n[断点保存] {checkpoint_path} (Step {global_step})")
    cleanup_old_checkpoints(output_dir)


def load_checkpoint(model, optimizer, checkpoint_path):
    print(f"\n[断点续训] 从 {checkpoint_path} 恢复")
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model.load_adapter(checkpoint_path, adapter_name="default")

    opt_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        return state["epoch"], state["global_step"], defaultdict(list, state["step_stats"])
    return 0, 0, defaultdict(list)


# =============================================================================
# 第八部分：周期评估
# =============================================================================

def evaluate_on_val(base_model_path, checkpoint_path, val_path, system_message):
    """加载检查点并在验证集上评估准确率"""
    print(f"\n[评估] 正在评估 {os.path.basename(checkpoint_path)} ...")
    try:
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    except Exception:
        print("[评估] 验证集加载失败，跳过")
        return None

    # 加载模型
    val_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, fix_mistral_regex=True)
    if val_tokenizer.pad_token is None:
        val_tokenizer.pad_token = val_tokenizer.eos_token
    val_tokenizer.padding_side = "left"

    val_base = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    val_model = PeftModel.from_pretrained(val_base, checkpoint_path)
    val_model.eval()

    correct = 0
    for item in tqdm(val_data, desc="  评估中", leave=False):
        q, a = item["question"], str(item["answer"]).strip()
        msgs = [{"role": "system", "content": system_message},
                {"role": "user", "content": q}]
        prompt = val_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = val_tokenizer(prompt, return_tensors="pt").to(val_model.device)
        with torch.no_grad():
            out = val_model.generate(**inp, max_new_tokens=256, do_sample=False,
                                     temperature=1.0, repetition_penalty=1.1,
                                     pad_token_id=val_tokenizer.pad_token_id,
                                     eos_token_id=val_tokenizer.eos_token_id)
        gen = out[0][inp.input_ids.shape[1]:]
        text = val_tokenizer.decode(gen, skip_special_tokens=True).strip()
        m = re.search(r"答案[：:]\s*(-?[\d\./]+)", text)
        pred = m.group(1) if m else (re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text) or [""])[-1]
        if pred == a:
            correct += 1

    acc = correct / len(val_data) * 100
    print(f"[评估] 准确率: {correct}/{len(val_data)} = {acc:.2f}%")

    # 释放评估模型显存
    del val_model, val_base, val_tokenizer
    torch.cuda.empty_cache()
    return acc


# =============================================================================
# 第九部分：训练循环
# =============================================================================

def train_grpo(model, ref_model, tokenizer, prepared_data, output_dir, resume_path=None):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, betas=(0.9, 0.99), eps=1e-5, weight_decay=0.01,
    )

    start_epoch, global_step, step_stats = 0, 0, defaultdict(list)
    if resume_path is not None and os.path.exists(resume_path):
        start_epoch, global_step, step_stats = load_checkpoint(model, optimizer, resume_path)

    model.train()
    ref_model.eval()

    print(f"\n{'='*60}")
    print(f"GRPO 训练（加速版）")
    print(f"{'='*60}")
    print(f"  样本数: {len(prepared_data)}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Mini-batch: {GRAD_ACCUMULATION}")
    print(f"  G (每问生成数): {NUM_GENERATIONS}")
    print(f"  PPO epochs: {PPO_EPOCHS}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*60}\n")

    best_val_acc = -1.0  # 追踪最佳验证准确率
    skip_count = 0
    running_reward_sum = 0.0
    running_reward_count = 0
    running_correct_sum = 0
    running_correct_total = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        indices = np.random.permutation(len(prepared_data))

        pbar = tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                    unit="q", dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

        batch_start = 0
        while batch_start < len(indices):
            # ---- Phase 1: 收集 mini-batch ----
            batch_data = []

            for _ in range(GRAD_ACCUMULATION):
                if batch_start >= len(indices):
                    break
                sample = prepared_data[indices[batch_start]]
                batch_start += 1

                prompt_ids = sample["prompt_ids"].to(model.device)
                correct_answer = sample["correct_answer"]

                # 【加速点1】批量生成 G 个 completion + 批量计算 old_log_probs
                completions, comp_ids, old_log_probs = generate_batch(
                    model, tokenizer, prompt_ids, NUM_GENERATIONS,
                    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K
                )

                # 计算奖励
                rewards = []
                extracted_answers = []
                for text in completions:
                    r, e = compute_mixed_reward(text, correct_answer)
                    rewards.append(r)
                    extracted_answers.append(e)

                running_reward_sum += sum(rewards)
                running_reward_count += len(rewards)
                running_correct_sum += sum(1 for r in rewards if r > 0.5)
                running_correct_total += len(rewards)

                # Group advantage
                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                mean_r = rewards_t.mean().item()
                std_r = rewards_t.std().item()
                if std_r < 1e-6:
                    advantages = torch.zeros_like(rewards_t)
                    skip_count += 1
                else:
                    advantages = (rewards_t - mean_r) / std_r

                # 计算 ref model log probs（逐条，与原来逻辑一致）
                ref_log_probs_list = []
                with torch.no_grad():
                    for g in range(NUM_GENERATIONS):
                        if len(comp_ids[g]) == 0:
                            ref_log_probs_list.append(torch.tensor([], dtype=torch.float32))
                            continue
                        ref_lp = compute_token_log_probs(ref_model, prompt_ids, comp_ids[g])
                        ref_log_probs_list.append(ref_lp.detach().cpu())

                batch_data.append({
                    "prompt_ids": prompt_ids,
                    "completion_ids": comp_ids,
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs_list,
                    "advantages": advantages,
                    "rewards": rewards,
                    "extracted_answers": extracted_answers,
                    "correct_answer": correct_answer,
                    "completions": completions,
                    "std_r": std_r,
                })

            if len(batch_data) == 0:
                continue

            # ---- Phase 2: 多轮 PPO 更新 ----
            batch_loss_vals = []
            for _ in range(PPO_EPOCHS):
                epoch_total_loss = 0.0
                epoch_kl = 0.0
                epoch_ratio = 0.0
                epoch_valid = 0
                epoch_ppo_loss = 0.0

                for data in batch_data:
                    for g in range(NUM_GENERATIONS):
                        if len(data["completion_ids"][g]) == 0:
                            continue

                        loss, metrics = compute_grpo_loss_per_token(
                            model, ref_model,
                            data["prompt_ids"],
                            data["completion_ids"][g],
                            data["old_log_probs"][g],
                            data["ref_log_probs"][g],
                            data["advantages"][g].item(),
                            EPSILON, BETA_KL,
                        )

                        if metrics.get("skipped", False):
                            continue

                        loss = loss / (len(batch_data) * NUM_GENERATIONS)
                        loss.backward()

                        epoch_total_loss += loss.item() * len(batch_data) * NUM_GENERATIONS
                        epoch_ppo_loss += metrics.get("ppo_loss", 0)
                        epoch_kl += metrics.get("kl", 0)
                        epoch_ratio += metrics.get("ratio_mean", 0)
                        epoch_valid += 1

                if epoch_valid > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss = epoch_total_loss / epoch_valid
                    avg_kl = epoch_kl / epoch_valid
                    step_stats["loss"].append(avg_loss)
                    step_stats["kl"].append(avg_kl)
                    step_stats["ratio"].append(epoch_ratio / epoch_valid)
                    batch_loss_vals.append(avg_loss)

                    # 定期详细日志
                    if global_step % LOG_EVERY_N_STEPS == 0:
                        last_data = batch_data[-1]
                        pbar.write(f"\n{'='*60}")
                        pbar.write(f"[Step {global_step}] Loss={avg_loss:.4f} | "
                                   f"KL={avg_kl:.4f} | Grad={grad_norm:.2f}")
                        pbar.write(f"  最近 Rewards: {[f'{r:.2f}' for r in last_data['rewards']]} "
                                   f"→ mean={np.mean(last_data['rewards']):.2f}, std={last_data['std_r']:.3f}")
                        pbar.write(f"  正确答案: {last_data['correct_answer']}, "
                                   f"提取: {last_data['extracted_answers']}")
                        pbar.write(f"  生成示例:")
                        for i, text in enumerate(last_data['completions'][:2]):
                            pbar.write(f"    [{i}] {text[:120].replace(chr(10), ' ')}...")
                        pbar.write(f"{'='*60}")

                    if global_step % SAVE_EVERY_N_STEPS == 0:
                        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-step-{global_step}")
                        save_checkpoint(model, optimizer, epoch, global_step, step_stats, CHECKPOINT_DIR)
                        # 周期评估
                        if os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
                            val_acc = evaluate_on_val(BASE_MODEL_PATH, ckpt_path,
                                                      VAL_DATA_PATH, SYSTEM_MESSAGE)
                            if val_acc is not None:
                                step_stats["val_acc"].append((global_step, val_acc))
                                # 保留最佳检查点
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    if os.path.exists(BEST_CHECKPOINT_DIR):
                                        shutil.rmtree(BEST_CHECKPOINT_DIR)
                                    shutil.copytree(ckpt_path, BEST_CHECKPOINT_DIR)
                                    print(f"[最佳] ★ 新最佳准确率: {best_val_acc:.2f}%，已保存到 {BEST_CHECKPOINT_DIR}")

            # 更新进度条
            pbar.update(len(batch_data))
            postfix = {"step": str(global_step)}
            if batch_loss_vals:
                postfix["loss"] = f"{np.mean(batch_loss_vals):.3f}"
            if running_reward_count > 0:
                postfix["reward"] = f"{running_reward_sum/running_reward_count:.2f}"
            if running_correct_total > 0:
                postfix["acc"] = f"{running_correct_sum/running_correct_total*100:.0f}%"
            if step_stats.get("val_acc"):
                postfix["val"] = f"{step_stats['val_acc'][-1][1]:.1f}%"
            if skip_count > 0:
                postfix["skip"] = str(skip_count)
            pbar.set_postfix(postfix, refresh=True)

        pbar.close()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} 完成 | 跳过: {skip_count}")

    print(f"\n[训练结束] 跳过样本: {skip_count}")
    if step_stats.get("val_acc"):
        vals = step_stats["val_acc"]
        print(f"[训练结束] 验证准确率历史: {', '.join(f'Step{s}={a:.1f}%' for s, a in vals)}")
        print(f"[训练结束] 最佳验证准确率: {best_val_acc:.2f}%，已保存到 {BEST_CHECKPOINT_DIR}")
    return step_stats


# =============================================================================
# 第九部分：保存与入口
# =============================================================================

def save_merged_model(model, tokenizer, output_path):
    os.makedirs(output_path, exist_ok=True)
    merged = model.merge_and_unload() if hasattr(model, 'merge_and_unload') else model
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    total_sz = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, dn, filenames in os.walk(output_path) for f in filenames)
    print(f"[保存] {output_path} ({total_sz/1024/1024:.1f} MB)")


def main():
    print(f"\n{'='*60}")
    print(f"GRPO 训练（加速版）")
    print(f"{'='*60}")

    # 1. Tokenizer
    print("\n[1/3] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 2. 数据（预 tokenize）
    print("\n[2/3] 加载并预 tokenize 数据...")
    prepared_data = load_and_prepare_data(TRAIN_DATA_PATH, tokenizer)

    # 3. 模型
    print("\n[3/3] 加载模型...")
    model = load_policy_model(BASE_MODEL_PATH)
    ref_model = load_reference_model(BASE_MODEL_PATH)

    # 训练
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    step_stats = train_grpo(
        model, ref_model, tokenizer, prepared_data, OUTPUT_DIR,
        resume_path=RESUME_FROM_CHECKPOINT,
    )

    # 保存
    print("\n保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    save_merged_model(model, tokenizer, MERGED_MODEL_DIR)
    print(f"完成！GRPO LoRA: {OUTPUT_DIR}, 合并: {MERGED_MODEL_DIR}")


if __name__ == "__main__":
    main()
