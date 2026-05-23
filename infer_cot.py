"""CoT inference on test set using fine-tuned Qwen + LoRA adapter."""
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ANSWER_RE = re.compile(r"答案[：:]\s*([\d]+(?:\.[\d]+)?(?:\/[1-9]\d*)?)")

COT_INSTRUCTION = (
    "你是小学数学解题助手。请按以下步骤解答问题："
    "先提取已知条件，明确求解目标，写出详细的计算过程，"
    "最后验证答案。最后一行必须是「答案：数字」。"
)


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1)
    numbers = re.findall(r"[\d]+(?:\.[\d]+)?(?:\/[1-9]\d*)?", text)
    if numbers:
        return numbers[-1]
    return text.strip().replace("\n", " ")


def predict(messages, model, tokenizer) -> str:
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "./output/Qwen_cot/checkpoint-2812/"

    model_path = "./Qwen/Qwen2___5-0___5B-Instruct/"
    print(f"Loading base model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id=ckpt)
    print(f"LoRA adapter loaded from {ckpt}")

    with open("test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Running inference on {len(test_data)} test samples ...")

    with open("submit.csv", "w", encoding="utf-8") as f:
        for row in tqdm(test_data):
            messages = [
                {"role": "system", "content": COT_INSTRUCTION},
                {"role": "user", "content": row["question"]},
            ]
            response = predict(messages, model, tokenizer)
            answer = extract_answer(response)
            f.write(f"{row['id']},{answer}\n")
    print("Done. Output saved to submit.csv")


if __name__ == "__main__":
    main()
