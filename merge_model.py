import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "./Qwen/Qwen2.5-0.5B-Instruct/"
SFT_LORA = "./output/Qwen_CoT_v2/checkpoint-7325"
DPO_LORA = "./qwen_dpo_output/checkpoint-2253"

# 1. 加载基座
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, 
    dtype=torch.bfloat16,  # 用 dtype 替代 torch_dtype
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载 SFT LoRA，合并，然后彻底卸载 PEFT 状态
model = PeftModel.from_pretrained(base_model, SFT_LORA)
model = model.merge_and_unload()
# 【关键】删除所有 PEFT 相关属性，确保基座"干净"
if hasattr(model, 'peft_config'):
    delattr(model, 'peft_config')

# 3. 重新包装为基座模型（彻底重置）
# 保存再加载，确保没有任何 PEFT 残留
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tokenizer.save_pretrained(tmpdir)
    
    # 从干净的状态重新加载
    clean_model = AutoModelForCausalLM.from_pretrained(
        tmpdir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

# 4. 在干净的模型上加载 DPO LoRA 并合并
model = PeftModel.from_pretrained(clean_model, DPO_LORA)
final_model = model.merge_and_unload()

# 5. 保存最终模型
final_model.save_pretrained("./qwen_dpo_full_clean")
tokenizer.save_pretrained("./qwen_dpo_full_clean")