import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "/home/zlx/chat-wukong/model/Qwen2.5-7B-Instruct"
adapter = "/home/zlx/chat-wukong/output/qwen2.5-7B-instruct_lora/final"

model = AutoModelForCausalLM.from_pretrained(
    base, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

messages = [
    {"role": "system", "content": "假设你现在是孙悟空。"},
    {"role": "user", "content": "大师兄，俺回高老庄了"},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    out = model.generate(
        inputs, max_new_tokens=256, temperature=0.1, top_p=0.9, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))