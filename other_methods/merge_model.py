# 新建一个脚本 merge_dpo_full.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "./Qwen/Qwen2.5-0.5B-Instruct/"
SFT_LORA = "./output/Qwen_CoT_v2/checkpoint-7325"   # 你的 SFT LoRA 路径

# 加载基座
base_model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
# 加载 SFT LoRA 并合并
model = PeftModel.from_pretrained(base_model, SFT_LORA)
model = model.merge_and_unload()                    # 得到 SFT 完整模型
# 加载 DPO LoRA 并合并

model.save_pretrained("./qwen_sft_full")
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.save_pretrained("./qwen_sft_full")