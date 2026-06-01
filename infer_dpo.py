"""
使用 DPO 合并模型在 test.json 上推理，生成 submit.csv

输出格式：id,答案（每行一条，无表头）
默认使用 GPU 1（INFER_GPU 可改），避免与 v3 流水线（GPU 0）冲突。
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("INFER_GPU", "1")

import csv
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./qwen_dpo_merged_final"
TEST_JSON_PATH = "test.json"
OUTPUT_CSV_PATH = "submit.csv"
SAVE_INTERVAL = 20

INSTRUCTION = (
    "你是小学数学解题助手。请按以下步骤解答问题："
    "先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。"
    "最后一行必须是「答案：数字」。"
)


def load_model(model_path):
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_final_answer(text: str) -> str:
    match = re.search(r"答案[：:]\s*(-?[\d\./]+)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else ""


def load_processed_ids(csv_path):
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    processed.add(str(row[0].strip()))
    return processed


def append_results(results, csv_path):
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for sample_id, answer in results:
            writer.writerow([sample_id, answer])


def main():
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    processed_ids = load_processed_ids(OUTPUT_CSV_PATH)
    pending = []
    for idx, item in enumerate(test_data):
        sample_id = str(item.get("id", idx))
        if sample_id not in processed_ids:
            pending.append((sample_id, item))

    print(f"测试集总数: {len(test_data)}，已完成: {len(processed_ids)}，待处理: {len(pending)}")
    if not pending:
        print("所有样本已推理完毕。")
        return

    model, tokenizer = load_model(MODEL_PATH)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    buffer = []
    for sample_id, item in tqdm(pending, desc="推理中"):
        question = item["question"]
        prompt = (
            f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        answer = extract_final_answer(response)

        buffer.append((sample_id, answer))
        if len(buffer) >= SAVE_INTERVAL:
            append_results(buffer, OUTPUT_CSV_PATH)
            buffer.clear()

    if buffer:
        append_results(buffer, OUTPUT_CSV_PATH)

    total_lines = len(load_processed_ids(OUTPUT_CSV_PATH))
    print(f"\n推理完成！结果已保存至: {OUTPUT_CSV_PATH}（共 {total_lines} 条）")


if __name__ == "__main__":
    main()
