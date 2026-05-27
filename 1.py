from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-0.5B-Instruct/", trust_remote_code=True)

# 手动拼接
manual_prompt = (
    "<|im_start|>system\n你是助手<|im_end|>\n"
    "<|im_start|>user\n问题<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# 自动模板
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "问题"}
]
auto_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("手动:\n", repr(manual_prompt))
print("自动:\n", repr(auto_prompt))