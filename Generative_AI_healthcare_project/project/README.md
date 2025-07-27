# 🩺 Medical Chatbot Project

A simple medical symptom checker and chatbot web app using Streamlit and Hugging Face models. Optionally, you can finetune the chatbot on your own medical dialogue data.

---

## Features
- Enter symptoms, get top-5 likely diseases
- Chatbot answers medical questions (using a small open model)
- Download your conversation as a PDF report
- Optional: Finetune the chatbot on your own data

---

## Setup
1. **Clone this repo and install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare datasets:**
   - Place a wide-format CSV at `data/dsx_wide.csv` (see below for format)
   - (Optional) For finetuning, place a JSONL at `data/meddialog.jsonl` with lines like `{ "dialogue": "I have a headache..." }`

---

## Running the App
```bash
streamlit run app.py
```

---

## Dataset Format
### Disease-Symptom Table (`data/dsx_wide.csv`)
- Columns: `Disease`, then one column per symptom (1 = present, 0 = absent)
- Example:
  ```csv
  Disease,fever,headache,cough
  Flu,1,1,1
  Migraine,0,1,0
  ```

### (Optional) Medical Dialogue Data (`data/meddialog.jsonl`)
- Each line: `{ "dialogue": "I have a headache and fever." }`

---

## Finetuning (Optional)
1. Place your dialogue data at `data/meddialog.jsonl`
2. Run:
   ```bash
   python3 finetune.py
   ```
3. This will save a finetuned model to `med-lora/` (update `predictor.py` to use it if desired)

---

## Notes
- The default chatbot uses `distilgpt2` (small, open, generic)
- For better results, use a larger or finetuned model
- All code is self-contained and easy to modify 