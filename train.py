from datasets import Dataset
import pandas as pd
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# 全局 tokenizer
tokenizer = None

def build_messages(example):
    # 建议与数据集人物设定一致
    system_prompt = "你现在以孙悟空（美猴王）的口吻与性格进行回复。"
    user_text = f"{example['instruction']}{example.get('input','')}"
    assistant_text = example['output']
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return messages

def process_func(example):
    messages = build_messages(example)

    # full input（包含assistant回复）
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )

    # prompt 部分（不含assistant内容，只留生成提示）
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1] + [{"role": "assistant", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )

    MAX_LENGTH = 2048
    input_ids = full_ids[:MAX_LENGTH]
    # 仅监督 assistant 部分
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "labels": labels[:len(input_ids)],
    }

if __name__ == "__main__":
    # 基座模型：强烈建议用 Instruct 版，chat_template 兼容指令微调
    # 例如：/home/zlx/chat-wukong/model/Qwen2.5-7B-Instruct
    model_path = "/home/zlx/chat-wukong/model/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("当前 tokenizer 不包含 chat_template，请改用 Qwen3-8B-Instruct 基座或自行设置 chat_template。")

    # 使用你的新数据集
    data_file = "/home/zlx/chat-wukong/dataset/train/lora/西游记白话文.json"
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if not {"instruction", "output"}.issubset(df.columns):
        raise ValueError("数据列应包含 instruction/output（可选 input）字段，请检查数据格式。")

    ds = Dataset.from_pandas(df)
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # LoRA 配置
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./output/qwen2.5_7B_instruct_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model("./output/qwen2.5-7B-instruct_lora/final")