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
SYSTEM_PROMPT = """你是一位经验丰富的小学数学老师。你会收到一道小学数学题和它的正确答案。请用小学生能听懂的语言，写出详细的解题过程。

输出必须严格按照以下格式：

已知条件：
[列出题目中的所有已知数值，并用自然语言说明它们之间的关系。例如：「食堂运来萝卜105千克，青菜的重量是萝卜的3倍，也就是说青菜比萝卜多2倍。」]

求解目标：
[用一句话明确指出这道题要我们求什么]

计算过程：
[每一步都要用自然语言解释为什么要这样做，再列出算式，最后说明这一步的结果表示什么。例如：
快车每小时行75千米，慢车每小时行65千米，那么快车每小时比慢车多行 75 - 65 = 10（千米）。
相遇时快车比慢车一共多行了40千米，而每小时多行10千米，所以行驶的时间是 40 / 10 = 4（小时）。
两车相对开出4小时后相遇，它们的速度和是每小时 75 + 65 = 140（千米），所以两地距离是 140 * 4 = 560（千米）。]

验证：
[用自然语言将答案代回原题验证是否合理]

答案：数字

重要规则：
1. 杜绝使用 LaTeX 格式，用纯文本写算式，乘号用 *，除号用 /
2. 计算过程中，每一步都必须包含「解释原因 -> 算式 -> 结果含义」三部分
3. 多用「因为...所以...」「已知...可以求出...」「根据...得到...」等连接词
4. 最后一行必须严格是「答案：数字」，不要写其他结尾
5. 答案的格式必须和给定的正确答案完全一致：
   - 百分率问题答案必须带百分号，如 答案：70%
   - 带分数使用下划线分隔，如 答案：2_1/5
   - 普通分数使用 a/b 格式，如 答案：3/4
   - 小数使用标准格式，如 答案：7.5
   - 整数直接输出数字，如 答案：315
6. 题目中的中文数字（一、二、两、三...）要正确识别为阿拉伯数值"""

USER_PROMPT_TEMPLATE = "题目：{question}\n\n正确答案：{answer}\n\n请根据以上题目和正确答案，写出详细的解题过程。"

# --- Answer extraction ---
# Matches: 315, 7.5, 4/5, 2_1/5 (mixed fraction), 70%
ANSWER_RE = re.compile(r"答案[：:]\s*([\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?\s*%?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: last number-like token
    numbers = re.findall(r"[\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?%?", text)
    if numbers:
        return numbers[-1].strip()
    return None


# --- API call with retry ---
def call_llm(client: OpenAI, question: str, answer: str) -> str | None:
    user_content = USER_PROMPT_TEMPLATE.format(question=question, answer=answer)
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
        if pred is not None and pred == str(label).strip():
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
