import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 配置参数 ====================
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct/"
SFT_LORA_PATH = "./output/Qwen_CoT_v2/checkpoint-7325"  # SFT LoRA 路径
DPO_LORA_PATH = "./qwen_dpo_output_2"     # DPO LoRA 路径

VAL_JSON_PATH = "small_val.json"
ERROR_OUTPUT_PATH = "dpo_full_val_errors.json"
PRINT_INTERVAL = 10
# ==================================================

def load_dpo_model(base_path, sft_lora_path, dpo_lora_path):
    """
    加载 DPO 模型：基座 + SFT LoRA（合并）+ DPO LoRA（不合并，直接叠加）
    
    由于 DPO 训练时是在 SFT 合并后的模型上添加的 LoRA，
    所以加载顺序：基座 → SFT LoRA → merge → DPO LoRA（保持为 LoRA）
    """
    print("=" * 60)
    print("加载 DPO 模型：基座 + SFT(合并) + DPO(LoRA)")
    print("=" * 60)
    
    # Step 1: 加载 Tokenizer
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Step 2: 加载基座模型
    print(f"正在加载基座模型: {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Step 3: 加载 SFT LoRA 并合并
    print(f"正在加载 SFT LoRA: {sft_lora_path}")
    model = PeftModel.from_pretrained(base_model, sft_lora_path)
    print("合并 SFT LoRA...")
    model = model.merge_and_unload()
    
    # Step 4: 加载 DPO LoRA（不合并，保持为 LoRA）
    print(f"正在加载 DPO LoRA: {dpo_lora_path}")
    model = PeftModel.from_pretrained(model, dpo_lora_path)
    
    model.eval()
    print("模型加载完成！")
    return model, tokenizer


def extract_final_answer(model_output_text: str) -> str:
    """从模型生成的文本中提取最终答案（支持负数、小数、分数）"""
    pattern = r'答案[：:]\s*(-?[\d\./]+)'
    match = re.search(pattern, model_output_text)
    if match:
        return match.group(1).strip()
    else:
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', model_output_text)
        return numbers[-1] if numbers else ""


def main():
    model, tokenizer = load_dpo_model(
        BASE_MODEL_PATH, 
        SFT_LORA_PATH, 
        DPO_LORA_PATH
    )

    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    correct_count = 0
    total_count = len(val_data)
    error_samples = []

    instruction = (
        "你是小学数学解题助手。请按以下步骤解答问题："
        "先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。"
        "最后一行必须是「答案：数字」。"
    )

    print(f"\n开始评估 DPO 模型，样本总数: {total_count}")
    print("=" * 60)

    with tqdm(total=total_count, desc="评估中") as pbar:
        for idx, item in enumerate(val_data, 1):
            question = item["question"]
            true_answer = str(item["answer"]).strip()

            prompt = (
                f"||<|im_start|>system\n{instruction}odes\n"
                f"||<|im_start|>user\n{question}odes\n"
                f"||<|im_start|>assistant\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            pred_answer = extract_final_answer(full_response)

            if pred_answer == true_answer:
                correct_count += 1
            else:
                error_item = item.copy()
                error_item["full_response"] = full_response
                error_item["extracted_pred"] = pred_answer
                error_samples.append(error_item)

            current_acc = correct_count / idx * 100
            pbar.set_postfix({
                "correct": correct_count,
                "acc": f"{current_acc:.2f}%"
            })
            pbar.update(1)

            if idx % PRINT_INTERVAL == 0:
                print(f"\n已处理 {idx}/{total_count} 样本，正确数: {correct_count}，准确率: {current_acc:.2f}%")

    accuracy = (correct_count / total_count) * 100
    print("\n" + "=" * 60)
    print("评估完成!")
    print(f"总样本数: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"错误数量: {total_count - correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("=" * 60)

    if error_samples:
        with open(ERROR_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(error_samples, f, ensure_ascii=False, indent=2)
        print(f"错误样本已保存至: {ERROR_OUTPUT_PATH} ({len(error_samples)}条)")
    else:
        print("恭喜！没有错误样本。")


if __name__ == "__main__":
    main()