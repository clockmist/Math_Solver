"""将 SFT LoRA 合并进基座模型，输出 qwen_sft_full/ 供 DPO 训练使用。"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "./Qwen/Qwen2.5-0.5B-Instruct/"
SFT_LORA = "./checkpoint-7325"
OUTPUT_DIR = "./qwen_sft_full"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, SFT_LORA)
model = model.merge_and_unload()

model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"合并完成，已保存到 {OUTPUT_DIR}/")
