import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# =========================
# 1. Model & tokenizer
# =========================

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,      # saves VRAM
    device_map="auto"
)

# =========================
# 2. LoRA configuration
# =========================

lora_config = LoraConfig(
    r=8,                    # adapter size
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# 3. Load dataset
# =========================

dataset = load_dataset("json", data_files="train.jsonl")

# =========================
# 4. Tokenization
# =========================

def tokenize(example):
    text = f"User: {example['instruction']}\nAssistant: {example['response']}"
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# =========================
# 5. Training arguments
# =========================

training_args = TrainingArguments(
    output_dir="./qwen-serbian-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    report_to="none"
)

# =========================
# 6. Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# =========================
# 7. Train
# =========================

trainer.train()
