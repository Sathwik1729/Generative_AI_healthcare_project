import streamlit as st
import pandas as pd
import datetime
import uuid
from predictor import chat, rank_diseases
from fpdf import FPDF

st.set_page_config(page_title="MedBot", page_icon="🩺", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Medical Symptom Checker & Chatbot")

sym_input = st.text_input("Describe your symptom(s) (e.g., fever, headache, nausea)")

if st.button("Get Advice") and sym_input:
    st.session_state.history.append(("User", sym_input))
    top_dis = rank_diseases([s.strip().lower() for s in sym_input.split(",")])
    qry = f"My symptoms are: {sym_input}. Possible conditions: {', '.join(top_dis.Disease)}. Advice?"
    answer = chat(qry)
    st.session_state.history.append(("MedBot", answer))

for role, txt in st.session_state.history[::-1]:
    st.chat_message(role).write(txt)

# ------------------ PDF REPORT ------------------
def build_pdf(conv):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Medical Symptom Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Date: {datetime.date.today()}", ln=True)
    pdf.ln(2)
    for role, msg in conv:
        pdf.multi_cell(0, 6, f"{role}: {msg}")
        pdf.ln(1)
    pdf.output("report.pdf")

if st.button("Download PDF"):
    build_pdf(st.session_state.history)
    with open("report.pdf", "rb") as f:
        st.download_button("Click to save", data=f,
                           file_name=f"med_report_{uuid.uuid4().hex[:6]}.pdf",
                           mime="application/pdf") 