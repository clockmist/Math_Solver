"""
对 test_gt_no_answer.json 中的题目用更简洁直接的 prompt 重新求解。

用法:
    python fill_no_answer.py --api-key your-key
"""
import json
import os
import re
import sys
import time

from openai import OpenAI

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


SYSTEM_PROMPT = (
    "你是一个计算器。请直接输出数学题的答案。\n"
    "规则：只输出答案数字，不要任何解释。分数用 a/b，带分数用 a_b/c，百分数带%。"
)


def extract_answer(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    # 去掉常见前缀
    for prefix in ("答案：", "答案是", "答案:", "结果是", "等于"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # 去掉句号结尾
    text = text.rstrip("。.")
    if not text:
        return None
    # 匹配数字/分数/百分数
    m = re.search(r"[\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?%?", text)
    if m:
        return m.group(0)
    return text


def call_llm(client: OpenAI, question: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{question}\n请直接输出答案："},
                ],
                temperature=0.1,
                max_tokens=128,
                timeout=60,
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

    no_answer_path = "test_gt_no_answer.json"
    test_gt_path = "test_gt.json"

    with open(no_answer_path, "r", encoding="utf-8") as f:
        no_answers = json.load(f)
    with open(test_gt_path, "r", encoding="utf-8") as f:
        test_gt = json.load(f)

    print(f"Loaded {len(no_answers)} no-answer questions, {len(test_gt)} answered.")

    recovered = 0
    still_no_answer = []

    for i, item in enumerate(no_answers):
        sid = item["id"]
        question = item["question"]
        print(f"[{i + 1}/{len(no_answers)}] id={sid} Q: {question[:60]}...")

        response = call_llm(client, question)
        if response is None:
            print(f"  -> API error, keeping in no-answer.")
            still_no_answer.append(item)
            time.sleep(REQUEST_DELAY)
            continue

        answer = extract_answer(response)
        if answer and answer.strip():
            test_gt.append({
                "id": sid,
                "question": question,
                "answer": answer,
            })
            recovered += 1
            print(f"  -> answer={answer}")
        else:
            item["raw_response"] = response
            still_no_answer.append(item)
            print(f"  -> empty/unparsable. raw={response[:60]}")

        time.sleep(REQUEST_DELAY)

        # Save after every sample
        with open(test_gt_path, "w", encoding="utf-8") as f:
            json.dump(test_gt, f, ensure_ascii=False, indent=2)
        with open(no_answer_path, "w", encoding="utf-8") as f:
            json.dump(still_no_answer, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Recovered: {recovered}, Still no answer: {len(still_no_answer)}")


if __name__ == "__main__":
    main()
