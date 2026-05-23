"""
使用外部强模型（DeepSeek）为训练数据生成 CoT 推理链。

用法（三选一）:
    1. 环境变量:  set LLM_API_KEY=your-key && python generate_cot_data.py
    2. 命令行:    python generate_cot_data.py --api-key your-key
    3. 文件:      创建 deepseek_key.txt，第一行写入你的 API key

也可通过环境变量切换模型:
    set LLM_BASE_URL=https://api.deepseek.com
    set LLM_MODEL=deepseek-v4-flash

输出:
    train_cot.json   - 验证通过的 CoT 格式训练数据
    cot_progress.json - 断点续传进度
    cot_failed.json   - 失败样本记录
"""
import json
import os
import re
import sys
import time
from datetime import datetime

from openai import OpenAI

# --- Config ---
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
REQUEST_DELAY = float(os.getenv("LLM_DELAY", "1.0"))


def get_api_key() -> str:
    """Try to get API key from: 1) env var, 2) CLI arg, 3) key file."""
    # 1) Environment variable (check both generic and legacy names)
    for env_name in ("LLM_API_KEY", "KIMI_API_KEY", "DEEPSEEK_API_KEY"):
        key = os.getenv(env_name, "")
        if key:
            return key

    # 2) Command-line --api-key
    for i, arg in enumerate(sys.argv):
        if arg == "--api-key" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]

    # 3) Key files
    for filename in ("deepseek_key.txt", "kimi_key.txt", "api_key.txt"):
        key_file = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.readline().strip()
                if key:
                    return key

    return ""


MAX_RETRIES = 3

# --- Prompt templates ---
SYSTEM_PROMPT = (
    "你是一位经验丰富的小学数学老师。你会收到一道小学数学题和它的正确答案。"
    "请根据题目和答案，写出完整、清晰的解题过程，确保推理能自然推导出给定的答案。\n"
    "\n"
    "输出必须严格按照以下格式：\n"
    "\n"
    "已知条件：\n"
    "[列出题目中的所有已知数值和关系]\n"
    "\n"
    "求解目标：\n"
    "[明确指出需要计算什么]\n"
    "\n"
    "计算过程：\n"
    "[按顺序写出每一步的计算算式和结果，每步一行。算式要完整，如：\n"
    "75 - 65 = 10（千米）\n"
    "40 ÷ 10 = 4（小时）\n"
    "(75 + 65) × 4 = 560（千米）]\n"
    "\n"
    "验证：\n"
    "[将答案代回原题验证是否合理，写出验证算式]\n"
    "\n"
    "答案：数字\n"
    "\n"
    "重要规则：\n"
    "1. 不要使用 LaTeX 格式（如 \\(...\\)），用纯文本写算式\n"
    "2. 最后一行必须严格是「答案：数字」，不要写「所以答案是...」或其他结尾\n"
    "3. 答案必须是纯数字，不带任何单位（如千米、千克、米）\n"
    "4. 分数使用 a/b 格式（如 3/4），小数使用标准格式（如 7.5）\n"
    "5. 题目中的中文数字（一、二、两、三...）要正确识别为阿拉伯数值"
)

USER_PROMPT_TEMPLATE = "题目：{question}\n\n正确答案：{answer}\n\n请根据以上题目和正确答案，写出详细的解题过程。"

# --- Answer extraction ---
# Matches: 315, 7.5, 4/5, 2_1/5 (mixed fraction), 70%
ANSWER_RE = re.compile(r"答案[：:]\s*([\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?)\s*%?")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return normalize_answer(m.group(1))
    # fallback: last number-like token
    numbers = re.findall(r"[\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?%?", text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def normalize_answer(answer: str) -> str:
    """Strip %, whitespace; handle None."""
    if not answer:
        return ""
    return str(answer).strip().rstrip("%")


# --- API call with retry ---
def call_llm(client: OpenAI, question: str, answer: str) -> str | None:
    # Strip % from the answer shown to the model (model outputs pure numbers)
    clean_answer = normalize_answer(answer)
    user_content = USER_PROMPT_TEMPLATE.format(question=question, answer=clean_answer)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=2048,
                timeout=120,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [attempt {attempt + 1}/{MAX_RETRIES}] API error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    return None


# --- Main ---
def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: API key not found. Provide it via one of:")
        print("  1. set LLM_API_KEY=your-key && python generate_cot_data.py")
        print("  2. python generate_cot_data.py --api-key your-key")
        print("  3. Create deepseek_key.txt with your API key on the first line")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    print(f"API: {BASE_URL}, Model: {MODEL}")

    with open("train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")

    # Resume support
    progress_path = "cot_progress.json"
    failed_path = "cot_failed.json"
    output_path = "train_cot.json"

    processed_ids = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            processed_ids = set(json.load(f))
        print(f"Resuming: {len(processed_ids)} already processed.")

    results = []  # validated CoT samples
    failed = []
    if os.path.exists(failed_path):
        with open(failed_path, "r", encoding="utf-8") as f:
            failed = json.load(f)

    total = len(train_data)
    matched = 0
    mismatched = 0
    api_errors = 0

    for i, sample in enumerate(train_data):
        sid = sample["id"]
        if sid in processed_ids:
            continue

        question = sample["question"]
        label = sample["answer"]

        # 跳过标注答案为空的样本
        if not label or not str(label).strip():
            print(f"[{i + 1}/{total}] id={sid} SKIP (empty label)")
            processed_ids.add(sid)
            continue

        print(f"[{i + 1}/{total}] id={sid} Q: {question[:60]}...")

        response = call_llm(client, question, str(label))
        if response is None:
            api_errors += 1
            failed.append({"id": sid, "question": question, "label": label, "reason": "API error"})
            processed_ids.add(sid)
            time.sleep(REQUEST_DELAY)
            continue

        pred = extract_answer(response)
        if pred is not None and normalize_answer(pred) == normalize_answer(str(label)):
            matched += 1
            results.append({
                "id": sid,
                "question": question,
                "answer": response,  # full CoT reasoning as answer
                "instruction": (
                    "你是小学数学解题助手。请按以下步骤解答问题："
                    "先提取已知条件，明确求解目标，写出详细的计算过程，"
                    "最后验证答案。最后一行必须是「答案：数字」。"
                ),
            })
        else:
            mismatched += 1
            failed.append({
                "id": sid, "question": question, "label": label,
                "pred": pred, "reason": "answer mismatch",
                "model_output": response,
            })

        processed_ids.add(sid)
        time.sleep(REQUEST_DELAY)

        # 每条立即写入，方便查看进度
        _save(results, output_path)
        _save(list(processed_ids), progress_path)
        _save(failed, failed_path)
        print(f"  [status] matched={matched} mismatched={mismatched} errors={api_errors}")

    # Final save
    _save(results, output_path)
    _save(list(processed_ids), progress_path)
    _save(failed, failed_path)

    print(f"\nDone. Total: {total}")
    print(f"  Matched: {matched} ({matched / total * 100:.1f}%)")
    print(f"  Mismatched: {mismatched} ({mismatched / total * 100:.1f}%)")
    print(f"  API errors: {api_errors} ({api_errors / total * 100:.1f}%)")
    print(f"  Output: {output_path} ({len(results)} samples)")
    print(f"  Failed log: {failed_path}")


def _save(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
