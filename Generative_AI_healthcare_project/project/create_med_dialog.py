# save as scripts/create_meddialog.py
from datasets import load_dataset
from random import sample
import json, pathlib

pathlib.Path("data").mkdir(exist_ok=True)

# 1) Stream the English split
ds = load_dataset("medical_dialog", "en", split="train")

# 2) Filter to primary-care specialities (optional)
PRIMARY = {"General Family Medicine", "Internal Medicine", "Pediatrics"}
ds = ds.filter(lambda x: x["specialty"] in PRIMARY)

# 3) Randomly pick 5 000 dialogues
pick = sample(range(len(ds)), k=5000)

with open("data/meddialog.jsonl", "w", encoding="utf-8") as f:
    for i in pick:
        obj = {"dialogue": ds[i]["dialogue"].strip()}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
