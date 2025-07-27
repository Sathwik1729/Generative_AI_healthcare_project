from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

BASE = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE)

def format_dialogue(example):
    prompt = f"User: {example['dialogue']}\nAI:"
    ids = tokenizer(prompt, truncation=True, max_length=512)
    ids["labels"] = ids["input_ids"].copy()
    return ids

ds = load_dataset("json", data_files="data/meddialog.jsonl")['train']
ds = ds.map(format_dialogue, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="med-lora",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
)

Trainer(model=model, args=args, train_dataset=ds).train()
model.save_pretrained("med-lora")
tokenizer.save_pretrained("med-lora") 