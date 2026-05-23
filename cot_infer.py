import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

COT_SYSTEM_PROMPT = (
    "你是小学数学解题助手。请严格按照以下步骤解答题目：\n"
    "\n"
    "第一步：提取条件\n"
    "- 列出题目中所有已知的数值和它们之间的关系。\n"
    "\n"
    "第二步：明确目标\n"
    "- 指出需要求解的是什么。\n"
    "\n"
    "第三步：分步求解\n"
    "- 按顺序写出每一步的计算过程和结果。\n"
    "\n"
    "第四步：验证\n"
    "- 将答案代入原题检查是否合理。\n"
    "\n"
    "【重要】你必须且只能在回答的最后一行输出最终答案，格式必须严格为：\n"
    "答案：数字\n"
    "不要输出「所以答案是xxx」「因此xxx」等其他结尾语句。\n"
    "答案必须是纯数字，不带任何单位。分数请使用 a/b 格式。"
)


def extract_answer(text: str) -> str:
    """从模型输出中提取 '答案：xxx' 后面的数字。"""
    m = re.search(r"答案[：:]\s*([\d]+(?:\.[\d]+)?(?:\/[\d]+)?)", text)
    if m:
        return m.group(1)
    numbers = re.findall(r"[\d]+(?:\.[\d]+)?(?:\/[\d]+)?", text)
    if numbers:
        return numbers[-1]
    return text.strip().replace("\n", " ")


def build_qwen_prompt(messages) -> str:
    """手动构造 Qwen2.5 chat 格式。"""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def predict(messages, model, tokenizer) -> str:
    device = "cuda"
    text = build_qwen_prompt(messages)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1024,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    model_path = "./Qwen/Qwen2___5-0___5B-Instruct/"
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded.")

    # --- 验证集评估 ---
    try:
        with open("val.json", "r", encoding="utf-8") as f:
            val_data = json.load(f)
        print(f"\nEvaluating on {len(val_data)} validation samples ...")
        correct = 0
        for row in tqdm(val_data):
            question = row["question"]
            label = row["answer"]
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            response = predict(messages, model, tokenizer)
            pred = extract_answer(response)
            if pred == label:
                correct += 1
        acc = correct / len(val_data) * 100
        print(f"Validation accuracy: {correct}/{len(val_data)} = {acc:.2f}%")
    except FileNotFoundError:
        print("val.json not found, skipping validation.")

    # --- 测试集推理 ---
    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"\nRunning inference on {len(test_data)} test samples ...")
    with open("submit.csv", "w", encoding="utf-8") as f:
        for row in tqdm(test_data):
            qid = row["id"]
            question = row["question"]
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            response = predict(messages, model, tokenizer)
            answer = extract_answer(response)
            f.write(f"{qid},{answer}\n")
    print("Done. Output saved to submit.csv")


if __name__ == "__main__":
    main()
