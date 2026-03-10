# imports
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "dataset/openhermes2_5.json"
OUTPUT_DIR = "./baseline_results"
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACC_STEPS = 1
LR = 1e-4
LOGGING_STEPS = 10

import wandb
wandb.init(
    project="selective-qlora",
    name="filtered-top30pct-10k-3epochs",
    config={
        "model": MODEL_ID,
        "dataset": DATASET_PATH,
        "r": 16,
        "lora_alpha": 32, 
        "lr": LR,
        "epochs": 3,
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE
    }
)


with open(DATASET_PATH, "r") as f:
    raw = json.load(f)

def to_text(example: dict) -> str:
    if "text" in example:
        return example["text"]
    if "prompt" in example and "completion" in example:
        return f"{example['prompt']}{example['completion']}"
    return " ".join(str(v) for v in example.values())


dataset = Dataset.from_list([{"text": to_text(e)} for e in raw])
dataset = dataset.shuffle(seed=42).select(range(10000))
print(f"Loaded {len(dataset)} samples from {DATASET_PATH}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    report_to="wandb", 
    logging_steps=LOGGING_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    learning_rate=LR,
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    gradient_checkpointing=True,
    bf16=True,
    fp16=False,
    save_strategy="epoch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    num_train_epochs=1
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_config,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Starting training …")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR}/final_adapter")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
wandb.finis()