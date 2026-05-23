"""对验证集前 N 条样本评估准确率，并保存详细结果供 review。"""
import json
import re
import sys
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

COT_SYSTEM_PROMPT = (
    "你是小学数学解题助手。请参考以下示例的解题格式和风格来解答题目。\n"
    "\n"
    "---示例1---\n"
    "题目：食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？\n"
    "解答：\n"
    "已知条件：萝卜105千克，青菜是萝卜的3倍。\n"
    "求解目标：青菜多少千克。\n"
    "计算过程：青菜 = 萝卜 × 3 = 105 × 3 = 315。\n"
    "验证：315 ÷ 3 = 105，等于萝卜重量，符合题意。\n"
    "答案：315\n"
    "\n"
    "---示例2---\n"
    "题目：一张长方形纸，涂红色占3/8，涂蓝色占1/8，没涂色的占这张纸的几分之几？\n"
    "解答：\n"
    "已知条件：红色占3/8，蓝色占1/8，整张纸看作1。\n"
    "求解目标：没涂色的占几分之几。\n"
    "计算过程：没涂色 = 1 - 3/8 - 1/8 = 8/8 - 3/8 - 1/8 = 4/8 = 1/2。\n"
    "验证：3/8 + 1/8 + 1/2 = 3/8 + 1/8 + 4/8 = 8/8 = 1，符合题意。\n"
    "答案：1/2\n"
    "\n"
    "---示例3---\n"
    "题目：一列快车和一列慢车同时从甲乙两地相对开出，快车每小时行75千米，慢车每小时行65千米，相遇时快车比慢车多行了40千米，甲乙两地相距多少千米？\n"
    "解答：\n"
    "已知条件：快车速度75千米/时，慢车速度65千米/时，相遇时快车多行40千米。\n"
    "求解目标：甲乙两地距离。\n"
    "计算过程：\n"
    "快车每小时比慢车多行 75-65=10千米。\n"
    "多行40千米需要 40÷10=4小时，即相遇时间为4小时。\n"
    "两地距离 = (75+65)×4 = 140×4 = 560千米。\n"
    "验证：快车行75×4=300千米，慢车行65×4=260千米，300-260=40，符合题意。\n"
    "答案：560\n"
    "\n"
    "---示例4---\n"
    "题目：把一根长12米的长方体木料从中间横截成两个小长方体，表面积增加了6平方米，原来这根木料的体积是多少立方米？\n"
    "解答：\n"
    "已知条件：木料长12米，横截后表面积增加6平方米（增加了2个底面积）。\n"
    "求解目标：原来木料的体积。\n"
    "计算过程：\n"
    "增加的表面积 = 2 × 底面积，所以底面积 = 6 ÷ 2 = 3平方米。\n"
    "体积 = 底面积 × 长 = 3 × 12 = 36立方米。\n"
    "验证：截断后新增2个底面共6平方米，每个底面3平方米，体积3×12=36，符合题意。\n"
    "答案：36\n"
    "\n"
    "---答题规则---\n"
    "1. 请模仿以上示例的格式解答新题目，包含：已知条件、求解目标、计算过程、验证。\n"
    '2. 最后一行必须是「答案：数字」，不要写「所以答案是xxx」。\n'
    '3. 答案必须是纯数字，不带单位。分数用 a/b 格式。\n'
    '4. 题目中的中文数字（如"两""三"）要正确识别为数值。'
)


def extract_answer(text: str) -> str:
    m = re.search(r"答案[：:]\s*([\d]+(?:\.[\d]+)?(?:\/[\d]+)?)", text)
    if m:
        return m.group(1)
    numbers = re.findall(r"[\d]+(?:\.[\d]+)?(?:\/[\d]+)?", text)
    if numbers:
        return numbers[-1]
    return text.strip().replace("\n", " ")


def build_qwen_prompt(messages) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
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
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    model_path = "./Qwen/Qwen2___5-0___5B-Instruct/"
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    print("Model loaded.")

    with open("val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    val_data = val_data[:n]
    results = []
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
        ok = (pred == label)
        if ok:
            correct += 1
        results.append({
            "id": row["id"],
            "question": question,
            "label": label,
            "pred": pred,
            "correct": ok,
            "model_output": response,
        })

    acc = correct / len(val_data) * 100
    print(f"\nAccuracy: {correct}/{len(val_data)} = {acc:.2f}%")

    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"cot_val_results_{timestamp}.json"
    summary = {
        "n": n,
        "accuracy": acc,
        "correct": correct,
        "system_prompt": COT_SYSTEM_PROMPT,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
