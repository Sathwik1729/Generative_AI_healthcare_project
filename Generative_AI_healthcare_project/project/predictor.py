import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SYMPTAB = pd.read_csv("data/dsx_wide.csv").fillna(0)
SYMPT_LIST = SYMPTAB.columns[1:]

def rank_diseases(symptoms):
    mask = SYMPTAB[SYMPT_LIST].isin(symptoms).astype(int)
    overlap = (mask * SYMPTAB[SYMPT_LIST]).sum(axis=1)
    return SYMPTAB.loc[overlap.nlargest(5).index][["Disease"]]

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def chat(user_text):
    prompt = f"User: {user_text}\nAI:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("AI:")[-1].strip() 