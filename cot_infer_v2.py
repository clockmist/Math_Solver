import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# 1. 改进 Prompt：去掉验证步骤，保留 4 个 few-shot 示例，结构更简洁
# =============================================================================
COT_SYSTEM_PROMPT = (
    "你是小学数学解题助手。请参考以下示例的解题格式和风格来解答题目。\n"
    "\n"
    "---示例1---\n"
    "题目：食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？\n"
    "解答：\n"
    "已知条件：萝卜105千克，青菜是萝卜的3倍。\n"
    "求解目标：青菜多少千克。\n"
    "计算过程：青菜 = 萝卜 × 3 = 105 × 3 = 315。\n"
    "答案：315\n"
    "\n"
    "---示例2---\n"
    "题目：一张长方形纸，涂红色占3/8，涂蓝色占1/8，没涂色的占这张纸的几分之几？\n"
    "解答：\n"
    "已知条件：红色占3/8，蓝色占1/8，整张纸看作1。\n"
    "求解目标：没涂色的占几分之几。\n"
    "计算过程：没涂色 = 1 - 3/8 - 1/8 = 8/8 - 3/8 - 1/8 = 4/8 = 1/2。\n"
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
    "答案：36\n"
    "\n"
    "---答题规则---\n"
    "1. 请模仿以上示例的格式解答新题目，包含：已知条件、求解目标、计算过程。\n"
    '2. 最后一行必须是「答案：数字」，不要写「所以答案是xxx」。\n'
    '3. 答案必须是纯数字，不带单位。分数用 a/b 格式。\n'
    '4. 题目中的中文数字（如"两""三"）要正确识别为数值。'
)


# =============================================================================
# 2. 改进答案提取器：增强鲁棒性，处理中英文冒号、大小写、常见异常格式
# =============================================================================
def extract_answer(text: str) -> str:
    """
    从模型输出中提取最终答案。
    优先级：
      1. 匹配「答案」或「answer」（不区分大小写）+ 中英文冒号
      2. 取文本中最后一个独立数字（整数/小数/分数/百分数）
      3. 兜底：返回去空白后的文本
    """
    text = text.strip()
    if not text:
        return ""

    # 优先级1：匹配 答案：xxx 或 answer：xxx（中英文冒号都支持）
    # 支持整数、小数、分数（如 3/4）、带分数（如 1_3/4）、百分数（如 12.5%）
    for pattern in [
        r"(?:答案|answer)[：:]\s*([\d]+(?:_[\d]+/[\d]+)?(?:\.[\d]+)?(?:/[\d]+)?(?:%)?)",
        r"(?:答案|answer)[：:]\s*([\d]+(?:\.[\d]+)?(?:/[\d]+)?)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 优先级2：取文本中最后一个独立数字 token
    # 匹配模式：整数、小数、分数、带分数、百分数
    number_pattern = r"[\d]+(?:_[\d]+/[\d]+)?(?:\.[\d]+)?(?:/[\d]+)?(?:%)?"
    numbers = re.findall(number_pattern, text)
    if numbers:
        return numbers[-1].strip()

    # 兜底：如果上面都失败，返回清理后的文本（去掉换行）
    cleaned = text.replace("\n", " ").strip()
    return cleaned


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
    # =============================================================================
    # 3. 使用 Qwen2.5-0.5B-Instruct，不微调（直接加载基座/指令模型）
    # =============================================================================
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

    # 加载测试集
    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    infer_limit = 4000  # 只推理前 4000 条

    print(f"\nTotal test samples: {total}")
    print(f"Will run inference on first {infer_limit} samples.")
    print(f"Samples {infer_limit}-{total} will be filled with 0.\n")

    # 保存完整模型输出（JSONL 格式，便于后续分析）
    outputs = []
    csv_rows = []

    # 前 4000 条：实际推理
    for row in tqdm(test_data[:infer_limit], desc="Inferring"):
        qid = row["id"]
        question = row["question"]

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        response = predict(messages, model, tokenizer)
        answer = extract_answer(response)

        outputs.append({
            "id": qid,
            "question": question,
            "model_output": response,
            "extracted_answer": answer,
        })
        csv_rows.append((qid, answer))

    # 后 4000 条：答案填 0，不推理
    for row in tqdm(test_data[infer_limit:], desc="Filling zeros"):
        qid = row["id"]
        outputs.append({
            "id": qid,
            "question": row["question"],
            "model_output": "",
            "extracted_answer": "0",
        })
        csv_rows.append((qid, "0"))

    # 保存完整模型输出到 JSON 文件
    out_json = "cot_v2_model_outputs.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"\nModel outputs saved to {out_json}")

    # 按照 submit.csv 格式保存 CSV
    out_csv = "submit_v2.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for qid, answer in csv_rows:
            writer.writerow([qid, answer])
    print(f"Submission saved to {out_csv} ({len(csv_rows)} rows)")

    # 简单统计
    non_zero = sum(1 for _, ans in csv_rows[:infer_limit] if ans != "0")
    empty = sum(1 for _, ans in csv_rows[:infer_limit] if ans == "" or ans == "0")
    print(f"\nStats (first {infer_limit} samples):")
    print(f"  Non-zero answers: {non_zero}")
    print(f"  Empty/zero answers: {empty}")
    print("Done.")


if __name__ == "__main__":
    main()
