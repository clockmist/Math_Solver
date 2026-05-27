"""
用 DeepSeek 为 test.json 生成伪标注答案（test_gt.json），用于本地快速测试评估。

用法：
    python generate_test_gt.py --api-key your-key
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


SYSTEM_PROMPT = (
    "你是小学数学解题助手。请解答题目并输出最终答案。\n"
    "\n"
    "示例1：\n"
    "题目：食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？\n"
    "答案：315\n"
    "\n"
    "示例2：\n"
    "题目：一张长方形纸，涂红色占3/8，涂蓝色占1/8，没涂色的占这张纸的几分之几？\n"
    "答案：1/2\n"
    "\n"
    "示例3：\n"
    "题目：李平家用600kg稻谷碾出420kg大米。他家稻谷的出米率是多少?\n"
    "答案：70%\n"
    "\n"
    "规则：\n"
    "1. 必须严格以「答案：」开头输出最终答案\n"
    "2. 答案必须是纯数字，不带单位\n"
    "3. 分数用 a/b 格式，带分数用 a_b/c 格式\n"
    "4. 百分率答案带百分号如 70%\n"
    "5. 不要写任何解题过程，只输出「答案：数字」"
)


def extract_answer(text: str) -> str | None:
    """提取文本中的数字/分数/百分数。"""
    text = text.strip()
    if not text:
        return None
    # 优先匹配「答案：xxx」格式
    m = re.search(r"答案[：:]\s*([\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?\s*%?)", text)
    if m:
        return m.group(1).strip()
    # fallback: 取最后一个看起来像数字的 token
    numbers = re.findall(r"[\d]+(?:_[\d]+)?(?:\.[\d]+)?(?:\/[\d]+)?%?", text)
    if numbers:
        return numbers[-1].strip()
    return None


def call_llm(client: OpenAI, question: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"题目：{question}\n请输出答案："},
                ],
                temperature=0.5,
                max_tokens=512,
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

    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    output_path = "test_gt.json"
    no_answer_path = "test_gt_no_answer.json"

    results = []
    no_answers = []
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_ids = {r["id"] for r in results}
        print(f"Resuming: {len(results)} already processed.")

    for i, sample in enumerate(test_data):
        sid = sample["id"]
        if sid in processed_ids:
            continue

        question = sample["question"]
        print(f"[{i + 1}/{len(test_data)}] id={sid} Q: {question[:60]}...")

        response = call_llm(client, question)
        if response is None:
            print(f"  -> API error, skipping.")
            time.sleep(REQUEST_DELAY)
            continue

        answer = extract_answer(response)
        if answer is None:
            print(f"  -> NO ANSWER extracted. Raw: {response[:100]}...")
            no_answers.append({
                "id": sid,
                "question": question,
                "raw_response": response,
            })
        else:
            results.append({
                "id": sid,
                "question": question,
                "answer": answer,
            })
            print(f"  -> answer={answer}")

        processed_ids.add(sid)
        time.sleep(REQUEST_DELAY)

        # Save after every sample
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if no_answers:
            with open(no_answer_path, "w", encoding="utf-8") as f:
                json.dump(no_answers, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total answered: {len(results)} / {len(test_data)}")
    print(f"No answer: {len(no_answers)}")
    print(f"Output: {output_path}")
    if no_answers:
        print(f"No-answer log: {no_answer_path}")


if __name__ == "__main__":
    main()
