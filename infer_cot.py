import json
import torch
import re
import csv
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 配置参数 ====================
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct/"
LORA_PATH = "./output/Qwen_CoT_v2/checkpoint-7325"   # 训练好的 checkpoint 路径
TEST_JSON_PATH = "test.json"                      # 测试集路径（需包含 id 和 question）
OUTPUT_CSV_PATH = "test_predictions_v2.csv"           # 输出 CSV 文件路径
SAVE_INTERVAL = 10                                 # 每推理 N 条就写入一次文件
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

def load_processed_ids(csv_path):
    """
    读取已经生成的 CSV 文件，返回已经处理过的样本 id 集合。
    若文件不存在或为空，返回空集合。
    """
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 非空行
                    # 第一列是 id，转换为字符串存储（便于比较）
                    processed.add(str(row[0].strip()))
    return processed

def append_results_to_csv(results, csv_path):
    """
    将一批结果（列表，每个元素为 (id, answer)）追加到 CSV 文件末尾。
    """
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for sample_id, answer in results:
            writer.writerow([sample_id, answer])

def main():
    # 检查输出文件是否已存在，获取已处理的 id 集合
    processed_ids = load_processed_ids(OUTPUT_CSV_PATH)
    if processed_ids:
        print(f"发现已有结果文件，已处理样本数: {len(processed_ids)}，将跳过这些样本继续推理。")
    
    # 加载测试数据
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    
    # 构建未处理的样本列表（保留原始顺序）
    pending_items = []
    for idx, item in enumerate(test_data):
        sample_id = str(item.get("id", item.get("idx", idx)))  # 统一转为字符串
        if sample_id not in processed_ids:
            pending_items.append((sample_id, item))
    
    total_pending = len(pending_items)
    if total_pending == 0:
        print("所有样本均已处理完毕，无需继续推理。")
        return
    
    print(f"测试集总样本数: {len(test_data)}，待处理样本数: {total_pending}")
    
    # 加载模型（放在识别待处理样本之后，避免白加载）
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_PATH)
    
    instruction = "你是小学数学解题助手。请按以下步骤解答问题：先提取已知条件，明确求解目标，写出详细的计算过程，最后验证答案。最后一行必须是「答案：数字」。"
    
    # 缓冲结果，每 SAVE_INTERVAL 条写入一次
    result_buffer = []
    processed_in_this_run = 0
    
    # 使用 tqdm 显示进度
    with tqdm(total=total_pending, desc="推理中") as pbar:
        for sample_id, item in pending_items:
            question = item["question"]
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
            
            result_buffer.append((sample_id, pred_answer))
            processed_in_this_run += 1
            pbar.update(1)
            
            # 达到保存间隔，写入文件并清空缓冲
            if len(result_buffer) >= SAVE_INTERVAL:
                append_results_to_csv(result_buffer, OUTPUT_CSV_PATH)
                result_buffer.clear()
    
    # 写入最后剩余的结果（不足 SAVE_INTERVAL 条）
    if result_buffer:
        append_results_to_csv(result_buffer, OUTPUT_CSV_PATH)
    
    print(f"\n推理完成！结果已保存至: {OUTPUT_CSV_PATH}")
    print(f"本次共处理 {processed_in_this_run} 个新样本，总结果行数: {len(processed_ids) + processed_in_this_run}")
    
    # 可选：打印前5条预测结果示例（从输出文件读取）
    print("\n前 5 条预测结果示例（来自输出文件）：")
    with open(OUTPUT_CSV_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.strip())

if __name__ == "__main__":
    main()