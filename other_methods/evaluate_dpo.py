"""
测试合并后的 DPO 完整模型（不含任何 LoRA）
模型路径：./qwen_dpo_full（由 merge_dpo_full.py 生成）
"""

import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 配置参数 ====================
MODEL_PATH = "./qwen_grpo_checkpoints/checkpoints-step-5400"               # 合并后的完整模型路径
VAL_JSON_PATH = "small_val.json"             # 验证集路径
ERROR_OUTPUT_PATH = "sft_full_val_errors.json"
PRINT_INTERVAL = 10
# ==================================================

def load_full_model(model_path):
    """直接加载完整的模型（不含 LoRA）"""
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    # 设置 pad_token（Qwen2.5 通常用 eos_token 作为 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def extract_final_answer(model_output_text: str) -> str:
    """从模型生成的文本中提取最终答案（支持负数、小数、分数）"""
    pattern = r'答案[：:]\s*(-?[\d\./]+)'
    match = re.search(pattern, model_output_text)
    if match:
        return match.group(1).strip()
    else:
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', model_output_text)
        return numbers[-1] if numbers else ""

def main():
    model, tokenizer = load_full_model(MODEL_PATH)

    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    correct_count = 0
    total_count = len(val_data)
    error_samples = []

    instruction = (
        "你是小学数学解题助手。请按以下步骤解答问题："
        "先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。"
        "最后一行必须是「答案：数字」。"
    )

    print(f"开始评估 DPO 完整模型，样本总数: {total_count}")

    with tqdm(total=total_count, desc="评估中") as pbar:
        for idx, item in enumerate(val_data, 1):
            question = item["question"]
            true_answer = str(item["answer"]).strip()

            prompt = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=tokenizer.encode("<|im_end|>")[0],
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            pred_answer = extract_final_answer(full_response)

            if pred_answer == true_answer:
                correct_count += 1
            else:
                error_item = item.copy()
                error_item["full_response"] = full_response
                error_item["extracted_pred"] = pred_answer
                error_samples.append(error_item)

            current_acc = correct_count / idx * 100
            pbar.set_postfix({
                "correct": correct_count,
                "acc": f"{current_acc:.2f}%"
            })
            pbar.update(1)

            if idx % PRINT_INTERVAL == 0:
                print(f"\n已处理 {idx}/{total_count} 样本，正确数: {correct_count}，准确率: {current_acc:.2f}%")

    accuracy = (correct_count / total_count) * 100
    print("\n" + "=" * 30)
    print(f"评估完成!")
    print(f"总样本数: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("=" * 30)

    with open(ERROR_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, ensure_ascii=False, indent=2)
    print(f"错误样本已保存至: {ERROR_OUTPUT_PATH}")

if __name__ == "__main__":
    main()