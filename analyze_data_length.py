#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据长度分布分析工具

功能：读取 train.json（包含 instruction, question, answer），模拟训练时的 tokenization 过程，
      统计每个样本的 input_ids 长度（含 prompt + response + pad_token_id），
      输出长度分布的统计信息，并给出建议的 MAX_LENGTH 值。

使用方法：
    python analyze_data_length.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from modelscope import snapshot_download
from transformers import AutoTokenizer

# ==================== 配置 ====================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_JSON_PATH = "train_cot.json"          # 请修改为您的实际路径
CACHE_DIR = "./"
OUTPUT_PLOT = "cot_length_distribution.png"  # 直方图保存路径（可选）

def load_tokenizer():
    """加载 tokenizer 并确保 pad_token 存在"""
    print("正在加载 tokenizer...")
    # 如果模型尚未下载，snapshot_download 会自动下载
    snapshot_download(MODEL_NAME, cache_dir=CACHE_DIR, revision="master")
    tokenizer = AutoTokenizer.from_pretrained(f"./{MODEL_NAME}", use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("已将 pad_token 设置为 eos_token")
    return tokenizer

def compute_input_ids_length(example, tokenizer):
    """
    完全模拟训练时的 input_ids 构造过程（不含 labels 和 attention_mask）
    返回 input_ids 的长度
    """
    # 1. 构造 prompt
    prompt_text = (
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # 2. 构造 response
    response_ids = tokenizer(f"{example['answer']}", add_special_tokens=False)["input_ids"]

    # 3. 拼接 + 末尾 pad_token_id
    total_ids = prompt_ids + response_ids + [tokenizer.pad_token_id]

    return len(total_ids)

def main():
    # 加载 tokenizer
    tokenizer = load_tokenizer()

    # 读取数据
    print(f"正在读取 {TRAIN_JSON_PATH} ...")
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"数据总量: {len(data)}")

    # 计算每条数据的长度
    lengths = []
    for idx, item in enumerate(data):
        # 确保必要字段存在
        if 'instruction' not in item or 'question' not in item or 'answer' not in item:
            print(f"警告: 第 {idx} 条数据缺少 instruction/question/answer 字段，跳过")
            continue
        try:
            length = compute_input_ids_length(item, tokenizer)
            lengths.append(length)
        except Exception as e:
            print(f"处理第 {idx} 条数据时出错: {e}")
            continue

    lengths = np.array(lengths)
    total = len(lengths)

    # 基本统计
    max_len = int(np.max(lengths))
    min_len = int(np.min(lengths))
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    print("\n========== 长度统计结果 ==========")
    print(f"有效样本数: {total}")
    print(f"最小长度: {min_len}")
    print(f"最大长度: {max_len}")
    print(f"平均长度: {mean_len:.2f}")
    print(f"中位数: {median_len:.1f}")
    print(f"90% 分位数: {p90:.1f}")
    print(f"95% 分位数: {p95:.1f}")
    print(f"99% 分位数: {p99:.1f}")

    # 建议 MAX_LENGTH
    print("\n========== 建议 ==========")
    # 通常选择 95% 分位数或 max_len，同时考虑显存限制
    suggest_1 = int(p95) + 16   # 留一点余量
    suggest_2 = int(p99) + 16
    print(f"若希望保留95%样本不截断，建议设置 MAX_LENGTH = {suggest_1}")
    print(f"若希望保留99%样本不截断，建议设置 MAX_LENGTH = {suggest_2}")
    print(f"若希望完全保留所有样本，设置 MAX_LENGTH = {max_len}")
    print("注意：实际训练时还需要考虑 batch_size 和显存，可根据 GPU 情况适当降低。")

    # 可选：绘制直方图
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(mean_len, color='r', linestyle='--', label=f'Mean ({mean_len:.1f})')
        plt.axvline(median_len, color='g', linestyle='--', label=f'Median ({median_len:.1f})')
        plt.axvline(p95, color='orange', linestyle='--', label=f'95% ({p95:.1f})')
        plt.xlabel('Input IDs Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Input IDs Length (prompt + response + pad)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_PLOT)
        print(f"\n直方图已保存至: {OUTPUT_PLOT}")
    except Exception as e:
        print(f"无法绘制直方图: {e}")

    # 输出超长样本示例（前5条长度超过 p95 的）
    if total > 0:
        over_p95_indices = np.where(lengths > p95)[0]
        if len(over_p95_indices) > 0:
            print("\n========== 超长样本示例（长度 > 95%分位数）==========")
            for i, idx in enumerate(over_p95_indices[:5]):
                item = data[idx]
                print(f"样本 {idx}: 长度 = {lengths[idx]}")
                print(f"  问题: {item['question'][:80]}...")
                print(f"  答案长度(字符): {len(item['answer'])}")
                print()

if __name__ == "__main__":
    main()