"""
评估 GRPO 检查点（LoRA 适配器）在 small_val.json 上的效果
用法：python evaluate_grpo.py
"""

import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 配置 ====================
BASE_MODEL_PATH = "./qwen_sft_full/"                          # 基座模型
CHECKPOINT_PATH = "./qwen_grpo_checkpoints/checkpoint-step-400"   # 最新快速训练检查点
VAL_PATH = "small_val.json"                                   # 验证集
ERROR_OUTPUT = "grpo_step5400_errors.json"                    # 错误样本输出
MAX_NEW_TOKENS = 256
# ==============================================

SYSTEM_MESSAGE = "你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"


def load_model(base_path, checkpoint_path):
    """加载基座模型 + GRPO LoRA 适配器"""
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"加载 LoRA 适配器: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, question):
    """构造与训练一致的 chat template prompt"""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_final_answer(text):
    """从模型输出中提取最终答案"""
    if not text:
        return ""
    # 找「答案：xxx」模式
    m = re.search(r"答案[：:]\s*(-?[\d\./]+)", text)
    if m:
        return m.group(1).strip()
    # fallback: 最后一个数字
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else ""


def main():
    model, tokenizer = load_model(BASE_MODEL_PATH, CHECKPOINT_PATH)

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n评估 GRPO 检查点: {CHECKPOINT_PATH}")
    print(f"验证样本数: {len(data)}")
    print()

    correct = 0
    errors = []

    for item in tqdm(data, desc="评估中"):
        question = item["question"]
        true_ans = str(item["answer"]).strip()

        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        pred = extract_final_answer(text)

        if pred == true_ans:
            correct += 1
        else:
            errors.append({**item, "pred": pred, "output": text})

    acc = correct / len(data) * 100
    print(f"\n{'='*50}")
    print(f"准确率: {correct}/{len(data)} = {acc:.2f}%")
    print(f"{'='*50}")

    with open(ERROR_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"错误样本已保存: {ERROR_OUTPUT}")


if __name__ == "__main__":
    main()
