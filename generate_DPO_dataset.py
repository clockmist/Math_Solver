"""
流式DPO数据集构建脚本（修复版）

修正：
1. build_prompt 中的特殊 token 错误（<|<|im_end|> -> <|im_end|>）
2. 删除 padding_side="left"，使用默认 right padding，避免生成截取错误
"""

import json
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== 配置参数 ====================
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct/"
SFT_LORA_PATH = "./output/Qwen_CoT_v2/checkpoint-7325"
TRAIN_DATA_PATH = "train_cot.json"
OUTPUT_PATH = "dpo_data.json"
CHECKPOINT_PATH = "dpo_build_checkpoint.json"

SFT_BATCH_SIZE = 4        # 每批处理4条
MAX_NEW_TOKENS = 512
# ==================================================


def load_sft_model():
    """加载SFT模型（不使用left padding）"""
    print("正在加载SFT模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
        # 不设置 padding_side，默认 "right"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, SFT_LORA_PATH)
    model.eval()
    print("SFT模型加载完成")
    return model, tokenizer


def extract_answer(cot_text):
    """
    从CoT中提取最终数字答案
    支持：整数、小数、分数（如 3/4）、带单位前的数字
    """
    if not cot_text:
        return None

    # 优先匹配 "答案：xxx" 或 "答案: xxx"
    pattern = r'答案[：:]\s*(-?\d+(?:\.\d+)?(?:/\d+)?)'
    match = re.search(pattern, cot_text)
    if match:
        ans_str = match.group(1).strip()
        return parse_number(ans_str)

    # fallback：找最后一个看起来像数字的片段
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', cot_text)
    if numbers:
        return parse_number(numbers[-1])

    return None


def parse_number(num_str):
    """将字符串解析为float，支持分数"""
    num_str = str(num_str).strip()
    try:
        if '/' in num_str:
            parts = num_str.split('/')
            if len(parts) == 2 and float(parts[1]) != 0:
                return float(parts[0]) / float(parts[1])
            return None
        return float(num_str)
    except (ValueError, ZeroDivisionError):
        return None


def build_prompt(question, instruction):
    """
    构建与SFT训练时一致的prompt
    【修复】特殊 token 正确写法：<|im_end|> (不是 <|<|im_end|>)
    如果instruction为空，则省略system字段
    """
    if instruction and instruction.strip():
        return (
            f"<|im_start|>system\n{instruction.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return (
            f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def clean_sft_output(raw_output, question):
    """
    清理SFT输出，去除可能的prompt残留
    """
    if not raw_output:
        return raw_output

    output = raw_output.strip()

    # 如果输出以问题文本开头，截断掉
    question_prefix = question.strip()[:20]
    if question_prefix and output.startswith(question_prefix):
        idx = output.find(question.strip())
        if idx != -1:
            after_question = output[idx + len(question.strip()):]
            cleaned = re.sub(r'^[\s\n]*(?:assistant\s*\n?)?', '', after_question)
            if cleaned.strip():
                return cleaned.strip()

    # 如果输出以"assistant"开头，去掉
    if output.lower().startswith("assistant"):
        output = re.sub(r'^assistant\s*\n?', '', output, flags=re.IGNORECASE).strip()

    return output


def sft_generate_batch(model, tokenizer, batch_items):
    prompts = [build_prompt(item["question"], item.get("instruction", "")) for item in batch_items]
    
    # left padding 关键：保证每个序列的 prompt 末尾对齐
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    # 获取每个样本的真实 prompt 长度（不含左边 pad）
    # left padding 下，attention_mask 最左边是 0，最右边是 1；我们需要找到第一个 1 的位置
    prompt_lengths = []
    for j in range(len(batch_items)):
        mask = inputs.attention_mask[j]
        # 找到第一个 1 的位置，即有效 prompt 的起始索引
        start_idx = (mask == 1).nonzero(as_tuple=True)[0][0].item()
        # 有效 prompt token 数 = mask 中 1 的总数
        prompt_len = mask.sum().item()
        # 但生成的 token 是跟在有效 prompt 之后，所以截取起点 = start_idx + prompt_len
        # 实际上为了从 output 中切出生成部分，更方便的是：output[:, start_idx + prompt_len :]
        # 下面我们在循环里处理
        prompt_lengths.append((start_idx, prompt_len))
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.encode("<|im_end|>")[0],
            pad_token_id=tokenizer.pad_token_id
        )
    
    results = []
    for j, output in enumerate(outputs):
        start_idx, prompt_len = prompt_lengths[j]
        # 从 output 中截取生成部分：从 prompt 结束位置开始到末尾
        # output 是 left-padded 序列，有效内容从 start_idx 开始，前 prompt_len 个是 prompt，之后是生成
        gen_start = start_idx + prompt_len
        generated_ids = output[gen_start:]
        # 过滤 pad（如果有尾随 pad 的话，一般不会）
        generated_ids = [id for id in generated_ids if id != tokenizer.pad_token_id]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        clean_response = clean_sft_output(raw_response, batch_items[j]["question"])
        results.append(clean_response)
    
    return results


def answers_equal(ans1, ans2, tolerance=1e-4):
    """判断两个答案是否相等"""
    if ans1 is None or ans2 is None:
        return False

    try:
        f1 = float(ans1)
        f2 = float(ans2)
        if abs(f1) > 1.0 or abs(f2) > 1.0:
            return abs(f1 - f2) <= tolerance * max(abs(f1), abs(f2), 1.0)
        else:
            return abs(f1 - f2) <= tolerance
    except (ValueError, TypeError):
        return str(ans1).strip() == str(ans2).strip()


def process_batch_and_save(batch_items, sft_outputs, dpo_data, processed_ids,
                           checkpoint_path, output_path):
    """处理batch并立即原子保存"""
    batch_correct = 0
    batch_error = 0

    for item, sft_output in zip(batch_items, sft_outputs):
        item_id = item["id"]
        correct_cot = item["answer"]
        correct_answer = extract_answer(correct_cot)
        sft_answer = extract_answer(sft_output)

        if answers_equal(correct_answer, sft_answer):
            batch_correct += 1
        else:
            batch_error += 1
            dpo_item = {
                "prompt": item["question"],
                "chosen": correct_cot,
                "rejected": sft_output,
                "metadata": {
                    "id": item_id,
                    "correct_answer": correct_answer,
                    "sft_answer": sft_answer,
                }
            }
            dpo_data.append(dpo_item)

        processed_ids.add(item_id)

    # 原子保存
    checkpoint = {
        "processed_ids": sorted(list(processed_ids)),
        "dpo_data": dpo_data,
        "last_batch_stats": {
            "batch_correct": batch_correct,
            "batch_error": batch_error,
        }
    }

    temp_ckpt = checkpoint_path + ".tmp"
    with open(temp_ckpt, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(temp_ckpt, checkpoint_path)

    temp_out = output_path + ".tmp"
    with open(temp_out, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_out, output_path)

    return batch_correct, batch_error


def load_checkpoint():
    """加载断点"""
    if not os.path.exists(CHECKPOINT_PATH):
        return {"processed_ids": [], "dpo_data": []}

    try:
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
        if not isinstance(ckpt.get("processed_ids"), list):
            print("警告：checkpoint格式异常，重置进度")
            return {"processed_ids": [], "dpo_data": []}
        return ckpt
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告：读取checkpoint失败 ({e})，尝试读取备份...")
        backup = CHECKPOINT_PATH + ".tmp"
        if os.path.exists(backup):
            try:
                with open(backup, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        print("无法恢复checkpoint，从头开始")
        return {"processed_ids": [], "dpo_data": []}


def build_dpo_dataset():
    """主函数：流式构建DPO数据集"""
    print("=" * 60)
    print("流式DPO数据集构建（修复版）")
    print("=" * 60)

    # 1. 加载训练数据
    print("\n[1/3] 加载训练数据...")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_DATA_PATH}")

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"训练集大小: {len(train_data)}")

    # 2. 加载断点
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint.get("processed_ids", []))
    dpo_data = checkpoint.get("dpo_data", [])
    print(f"已处理样本数: {len(processed_ids)}")
    print(f"已收集DPO样本数: {len(dpo_data)}")

    pending_items = [item for item in train_data if item.get("id") not in processed_ids]
    print(f"待处理样本数: {len(pending_items)}")

    if len(pending_items) == 0:
        print("所有样本已处理完毕！")
        print(f"最终DPO数据集大小: {len(dpo_data)}")
        return dpo_data

    # 3. 加载SFT模型
    print("\n[2/3] 加载SFT模型...")
    model, tokenizer = load_sft_model()

    # 4. 流式处理
    print("\n[3/3] 开始流式处理...")
    print(f"Batch size: {SFT_BATCH_SIZE}")
    print("=" * 60)

    total_correct = 0
    total_error = 0
    num_batches = (len(pending_items) + SFT_BATCH_SIZE - 1) // SFT_BATCH_SIZE

    pbar = tqdm(range(num_batches), desc="处理batch")

    for batch_idx in pbar:
        start_idx = batch_idx * SFT_BATCH_SIZE
        end_idx = min(start_idx + SFT_BATCH_SIZE, len(pending_items))
        batch_items = pending_items[start_idx:end_idx]

        sft_outputs = sft_generate_batch(model, tokenizer, batch_items)

        batch_correct, batch_error = process_batch_and_save(
            batch_items, sft_outputs, dpo_data, processed_ids,
            CHECKPOINT_PATH, OUTPUT_PATH
        )

        total_correct += batch_correct
        total_error += batch_error

        current_total = len(processed_ids)
        acc = total_correct / current_total if current_total > 0 else 0
        pbar.set_postfix({
            "已处理": f"{current_total}/{len(train_data)}",
            "DPO样本": len(dpo_data),
            "跳过(正确)": total_correct,
            "准确率": f"{acc:.1%}",
            "已保存": "✓"
        })

    total_processed = len(processed_ids)
    final_acc = total_correct / total_processed if total_processed > 0 else 0

    print("\n" + "=" * 60)
    print("DPO数据集构建完成！")
    print(f"总处理样本: {total_processed}")
    print(f"SFT正确样本: {total_correct} (已跳过)")
    print(f"SFT错误样本: {total_error} → 已加入DPO数据")
    print(f"最终DPO数据集大小: {len(dpo_data)}")
    print(f"SFT在训练集上的准确率: {final_acc:.1%}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"断点文件: {CHECKPOINT_PATH}")
    print("=" * 60)

    for temp_file in [CHECKPOINT_PATH + ".tmp", OUTPUT_PATH + ".tmp"]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

    return dpo_data


if __name__ == "__main__":
    build_dpo_dataset()