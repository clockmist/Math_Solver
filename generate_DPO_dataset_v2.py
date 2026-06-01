"""
DPO 数据集构建 v2：最小改动（Minimal-Edit）偏好对构造

核心思路（见 dpo_readme.md 第3节）：
  DPO 更擅长学习「在相似推理路径下，哪个步骤/结果更好」，而非完全不同的推理风格。
  因此 rejected 应从 chosen（正确 CoT）出发，仅改动一个算式（算错结果、改运算符、改操作数），
  使 chosen/rejected 结构高度一致，模型能学到「如何修正具体计算错误」。

输入：train_cot.json（DeepSeek 生成的正确 CoT）
输出：dpo_data_v2.json
"""

import json
import os
import random
import re
from fractions import Fraction
from typing import Optional

# ==================== 配置 ====================
TRAIN_DATA_PATH = "train_cot.json"
OUTPUT_PATH = "dpo_data_v2.json"
STATS_PATH = "dpo_data_v2_stats.json"
RANDOM_SEED = 42
MAX_SAMPLES = None  # None = 全量；调试时可设小值
# ==============================================

random.seed(RANDOM_SEED)

# 匹配 "expr = result" 算式，支持中文括号单位
CALC_PATTERN = re.compile(
    r"(?P<expr>[\d\.\/\(\)\s\+\-\*×÷]+?)\s*=\s*(?P<result>-?\d+(?:\.\d+)?(?:/\d+)?)"
    r"(?:\s*（[^）]*）)?"
)

ANSWER_PATTERN = re.compile(r"答案[：:]\s*(-?[\d\./]+)")
OP_CHARS = {"+", "-", "*", "/", "×", "÷"}


def parse_number(text: str) -> Optional[float]:
    text = str(text).strip()
    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) == 2 and float(parts[1]) != 0:
                return float(parts[0]) / float(parts[1])
            return None
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.4g}".rstrip("0").rstrip(".")


def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else None


def find_calc_spans(text: str) -> list[re.Match]:
    return list(CALC_PATTERN.finditer(text))


def eval_simple_expr(expr: str) -> Optional[float]:
    """安全评估仅含四则运算的表达式"""
    expr = expr.strip().replace("×", "*").replace("÷", "/")
    expr = re.sub(r"\s+", "", expr)
    if not re.fullmatch(r"[\d\.\/\(\)\+\-\*]+", expr):
        return None
    try:
        return float(Fraction(eval(expr, {"__builtins__": {}}, {})))
    except Exception:
        return None


def corrupt_wrong_result(result_str: str) -> Optional[str]:
    """算错结果：±1 或 ±10%，保证与正确答案不同"""
    val = parse_number(result_str)
    if val is None:
        return None

    candidates = []
    for delta in (1, -1, 2, -2, 5, -5, 10, -10):
        candidates.append(val + delta)
    for factor in (1.1, 0.9, 1.2, 0.8):
        candidates.append(val * factor)

    for new_val in candidates:
        if abs(new_val - val) > 1e-6:
            return format_number(new_val)
    return format_number(val + 1)


def corrupt_wrong_operator(expr: str) -> Optional[str]:
    """改运算符：* <-> +, / <-> - 等"""
    expr = expr.strip()
    for op in ("*", "×", "+", "-", "/", "÷"):
        if op in expr:
            alt_map = {"*": "+", "×": "+", "+": "*", "-": "/", "/": "-", "÷": "-"}
            alt = alt_map.get(op)
            if alt:
                new_expr = expr.replace(op, alt, 1)
                if eval_simple_expr(new_expr) is not None:
                    return new_expr
    return None


def corrupt_wrong_operand(expr: str) -> Optional[str]:
    """改操作数：将表达式中某个数字 ±1"""
    numbers = list(re.finditer(r"-?\d+(?:\.\d+)?(?:/\d+)?", expr))
    if not numbers:
        return None

    idx = random.randrange(len(numbers))
    m = numbers[idx]
    val = parse_number(m.group())
    if val is None:
        return None

    for delta in (1, -1, 2, -2):
        new_val = val + delta
        new_num = format_number(new_val)
        new_expr = expr[: m.start()] + new_num + expr[m.end() :]
        if eval_simple_expr(new_expr) is not None:
            return new_expr
    return None


def apply_corruption(text: str, match: re.Match, strategy: str) -> Optional[str]:
    expr = match.group("expr").strip()
    result = match.group("result")

    if strategy == "wrong_result":
        new_result = corrupt_wrong_result(result)
        if new_result is None or new_result == result:
            return None
        replacement = match.group(0).replace(f"= {result}", f"= {new_result}", 1)
        new_text = text[: match.start()] + replacement + text[match.end() :]
        return new_text

    if strategy == "wrong_operator":
        new_expr = corrupt_wrong_operator(expr)
        if new_expr is None:
            return None
        new_val = eval_simple_expr(new_expr)
        if new_val is None:
            return None
        new_result = format_number(new_val)
        unit_suffix = ""
        unit_match = re.search(r"（[^）]*）", match.group(0))
        if unit_match:
            unit_suffix = unit_match.group(0)
        replacement = f"{new_expr} = {new_result}{unit_suffix}"
        new_text = text[: match.start()] + replacement + text[match.end() :]
        return new_text

    if strategy == "wrong_operand":
        new_expr = corrupt_wrong_operand(expr)
        if new_expr is None:
            return None
        new_val = eval_simple_expr(new_expr)
        if new_val is None:
            return None
        new_result = format_number(new_val)
        unit_suffix = ""
        unit_match = re.search(r"（[^）]*）", match.group(0))
        if unit_match:
            unit_suffix = unit_match.group(0)
        replacement = f"{new_expr} = {new_result}{unit_suffix}"
        new_text = text[: match.start()] + replacement + text[match.end() :]
        return new_text

    return None


def update_final_answer(text: str, wrong_answer: str) -> str:
    if ANSWER_PATTERN.search(text):
        return ANSWER_PATTERN.sub(f"答案：{wrong_answer}", text, count=1)
    return text.rstrip() + f"\n\n答案：{wrong_answer}"


def build_rejected(correct_cot: str) -> Optional[dict]:
    spans = find_calc_spans(correct_cot)
    if not spans:
        return None

    correct_ans = extract_answer(correct_cot)
    if correct_ans is None:
        return None

    span = spans[-1]  # 只改动最后一个算式
    strategies = ["wrong_result", "wrong_operand", "wrong_operator"]
    random.shuffle(strategies)

    for strategy in strategies:
        rejected = apply_corruption(correct_cot, span, strategy)
        if rejected is None or rejected == correct_cot:
            continue

        new_spans = find_calc_spans(rejected)
        if not new_spans:
            continue

        wrong_final = new_spans[-1].group("result")
        rejected = update_final_answer(rejected, wrong_final)

        if wrong_final == correct_ans:
            continue

        return {
            "rejected": rejected,
            "corruption": {
                "strategy": strategy,
                "step_index": len(spans) - 1,
                "total_steps": len(spans),
                "old_result": span.group("result"),
                "new_result": wrong_final,
                "correct_answer": correct_ans,
                "wrong_answer": wrong_final,
            },
        }

    return None


def build_dpo_dataset():
    print("=" * 60)
    print("DPO 数据集构建 v2（最小改动 / Minimal-Edit）")
    print("=" * 60)

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    if MAX_SAMPLES:
        train_data = train_data[:MAX_SAMPLES]

    dpo_data = []
    stats = {
        "total_input": len(train_data),
        "generated": 0,
        "skipped_no_calc": 0,
        "skipped_failed": 0,
        "strategy_counts": {},
    }

    for item in train_data:
        correct_cot = item["answer"]
        result = build_rejected(correct_cot)
        if result is None:
            if not find_calc_spans(correct_cot):
                stats["skipped_no_calc"] += 1
            else:
                stats["skipped_failed"] += 1
            continue

        strategy = result["corruption"]["strategy"]
        stats["strategy_counts"][strategy] = stats["strategy_counts"].get(strategy, 0) + 1

        dpo_data.append(
            {
                "prompt": item["question"],
                "chosen": correct_cot,
                "rejected": result["rejected"],
                "metadata": {
                    "id": item.get("id"),
                    "corruption": result["corruption"],
                    "source": "minimal_edit_v2",
                },
            }
        )
        stats["generated"] += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n输入样本: {stats['total_input']}")
    print(f"成功生成: {stats['generated']}")
    print(f"跳过(无算式): {stats['skipped_no_calc']}")
    print(f"跳过(构造失败): {stats['skipped_failed']}")
    print(f"策略分布: {stats['strategy_counts']}")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 60)

    return dpo_data


if __name__ == "__main__":
    build_dpo_dataset()
