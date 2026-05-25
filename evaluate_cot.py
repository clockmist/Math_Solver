import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 配置参数 ====================
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct/"
LORA_PATH = "./output/Qwen_CoT/checkpoint-7325"  # 替换为你训练好的 checkpoint 路径
VAL_JSON_PATH = "small_val.json"                      # 验证集路径
ERROR_OUTPUT_PATH = "cot_val_errors_2.json"
PRINT_INTERVAL = 10                          # 每隔多少样本打印一次当前准确率
# ==================================================

def load_model_and_tokenizer(base_path, lora_path):
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, trust_remote_code=True)
    
    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    print("正在加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model, tokenizer

def extract_final_answer(model_output_text: str) -> str:
    """
    从模型生成的文本中提取最终答案（支持负数、小数、分数）
    """
    pattern = r'答案[：:]\s*(-?[\d\./]+)'
    match = re.search(pattern, model_output_text)
    if match:
        return match.group(1).strip()
    else:
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', model_output_text)
        return numbers[-1] if numbers else ""

def main():
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_PATH)
    
    with open(VAL_JSON_PATH, 'r', encoding='utf-8') as file:
        val_data = json.load(file)
    
    correct_count = 0
    total_count = len(val_data)
    error_samples = []
    
    print(f"开始评估 CoT 模型，样本总数: {total_count}")
    
    # 使用 tqdm，并设置 postfix 初始值
    with tqdm(total=total_count, desc="评估中") as pbar:
        for idx, item in enumerate(val_data, 1):
            #instruction = item["instruction"]
            instruction = f"你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"
            question = item["question"]
            true_answer = str(item["answer"]).strip()
            
            prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.encode("<|im_end|>")[0]
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
            
            # 更新进度条的后缀信息（实时显示当前正确数和准确率）
            current_acc = correct_count / idx * 100
            pbar.set_postfix({
                "correct": correct_count,
                "acc": f"{current_acc:.2f}%"
            })
            pbar.update(1)
            
            # 可选：每隔 PRINT_INTERVAL 样本打印一次详细的中间统计
            if idx % PRINT_INTERVAL == 0:
                print(f"\n已处理 {idx}/{total_count} 样本，当前正确数: {correct_count}，准确率: {current_acc:.2f}%")
    
    # 最终统计
    accuracy = (correct_count / total_count) * 100
    print("\n" + "="*30)
    print(f"评估完成!")
    print(f"总样本数: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print("="*30)
    
    with open(ERROR_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, ensure_ascii=False, indent=2)
    print(f"错误样本已保存至: {ERROR_OUTPUT_PATH}")

if __name__ == "__main__":
    main()