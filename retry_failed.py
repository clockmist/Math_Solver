"""
从 cot_failed.json 中读取失败样本，重新调用模型生成 CoT。
成功则从 failed 中移除并添加到 train_cot.json,失败保留在 failed 中。

用法：
    python retry_failed.py --api-key your-key
"""
import json
import os
import re
import sys
import time

from openai import OpenAI

# --- Config ---
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
REQUEST_DELAY = float(os.getenv("LLM_DELAY", "1.0"))
MAX_RETRIES = 3


def get_api_key() -> str:
    for env_name in ("LLM_API_KEY", "KIMI_API_KEY", "DEEPSEEK_API_KEY"):
        key = os.getenv(env_name, "")
        if key:
            return key
    for i, arg in enumerate(sys.argv):
        if arg == "--api-key" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    for filename in ("deepseek_key.txt", "kimi_key.txt", "api_key.txt"):
        key_file = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.readline().strip()
                if key:
                    return key
    return ""


SYSTEM_PROMPT = """你是一位经验丰富的小学数学老师。你会收到一道小学数学题和它的正确答案。请写出简洁清晰的解题过程，每一步用一句自然语言解释原因，再列出对应的算式。

输出格式：

已知条件：
[简述题目中的已知数值和关系]

求解目标：
[一句话指出要求什么]

计算过程：
[每一步先解释为什么要算这一步，再给出算式。例如：
快车每小时比慢车多行 75 - 65 = 10（千米）。
多行40千米需要行驶 40 / 10 = 4（小时）。
两车速度和为 75 + 65 = 140（千米/时），两地距离 140 * 4 = 560（千米）。]

答案：数字

规则：
1. 纯文本，不用 LaTeX，乘除用 * 和 /
2. 每一步「解释+算式」即可，不拖沓
3. 最后一行必须是「答案：数字」
4. 答案格式与给定正确答案一致：百分数带%（70%）、带分数用下划线（2_1/5）、普通分数用 a/b（3/4）
5. 中文数字（一、两、三...）识别为阿拉伯数值"""

USER_PROMPT_TEMPLATE = "题目：{question}\n\n正确答案：{answer}\n\n请根据以上题目和正确答案，写出详细的解题过程。"

ANSWER_RE = re.compile(r"答案[：:]\s*([\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?\s*%?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    numbers = re.findall(r"[\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?%?", text)
    if numbers:
        return numbers[-1].strip()
    return None


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


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: API key not found. Provide via --api-key or deepseek_key.txt")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    print(f"API: {BASE_URL}, Model: {MODEL}")

    # Load existing data
    train_cot_path = "train_cot.json"
    failed_path = "cot_failed.json"

    if not os.path.exists(failed_path):
        print("No cot_failed.json found. Nothing to retry.")
        return

    with open(train_cot_path, "r", encoding="utf-8") as f:
        train_cot = json.load(f)
    with open(failed_path, "r", encoding="utf-8") as f:
        failed = json.load(f)

    print(f"Loaded {len(train_cot)} train samples, {len(failed)} failed samples.")

    recovered = []
    still_failed = []

    for i, item in enumerate(failed):
        sid = item["id"]
        question = item["question"]
        label = str(item["label"]).strip()

        print(f"[{i + 1}/{len(failed)}] id={sid} retrying...")

        response = call_llm(client, question, label)
        if response is None:
            print(f"  -> API error, keeping in failed.")
            still_failed.append(item)
            time.sleep(REQUEST_DELAY)
            continue

        pred = extract_answer(response)
        if pred is not None and pred == label:
            recovered.append({
                "id": sid,
                "question": question,
                "answer": response,
                "instruction": (
                    "你是小学数学解题助手。请按以下步骤解答问题："
                    "先提取已知条件，明确求解目标，写出详细的计算过程，"
                    "最后验证答案。最后一行必须是「答案：数字」。"
                ),
            })
            print(f"  -> RECOVERED! answer={pred}")
        else:
            item["retry_pred"] = pred
            item["retry_output"] = response
            still_failed.append(item)
            print(f"  -> still mismatch. pred={pred} label={label}")

        time.sleep(REQUEST_DELAY)

        # Save after every sample
        if recovered:
            train_cot.extend(recovered)
            recovered.clear()
        with open(train_cot_path, "w", encoding="utf-8") as f:
            json.dump(train_cot, f, ensure_ascii=False, indent=2)
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(still_failed, f, ensure_ascii=False, indent=2)

    with open(train_cot_path, "r", encoding="utf-8") as f:
        final_train_cot = json.load(f)
    print(f"\nDone. Recovered: {len(final_train_cot) - len(train_cot) + len(recovered)}")
    print(f"Still failed: {len(still_failed)}")


if __name__ == "__main__":
    main()
