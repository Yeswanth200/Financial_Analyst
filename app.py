"""
Project : Financial Analyst — Gradio Demo
==========================================
Run: python gradio_app.py
"""

import torch
import yfinance as yf
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sklearn.metrics import classification_report
import numpy as np

BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "./financial_analyst_model"

# ─── Load model ───────────────────────────────────────────────────────────────
def load_pipe():
    tok  = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mdl  = PeftModel.from_pretrained(base, ADAPTER_DIR)
    mdl.eval()
    return pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=512, temperature=0.1, do_sample=True,
    )

pipe = load_pipe()

def infer(prompt: str) -> str:
    out = pipe(prompt)[0]["generated_text"]
    return out.split("[/INST]")[-1].strip()

# ─── Core functions ───────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> str:
    prompt = (
        f"<s>[INST] You are a financial analyst. Classify the sentiment "
        f"(POSITIVE / NEGATIVE / NEUTRAL) of this financial statement and explain why.\n\n"
        f"Statement: {text} [/INST]"
    )
    return infer(prompt)

def analyze_filing(ticker: str, question: str) -> str:
    try:
        info    = yf.Ticker(ticker).info
        summary = (
            f"Company: {info.get('longName', ticker)}\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Business Summary: {info.get('longBusinessSummary', '')[:1500]}\n"
            f"Revenue: {info.get('totalRevenue', 'N/A')}\n"
            f"Net Income: {info.get('netIncomeToCommon', 'N/A')}\n"
            f"PE Ratio: {info.get('trailingPE', 'N/A')}\n"
        )
    except Exception as e:
        summary = f"Could not fetch data for {ticker}: {e}"

    prompt = (
        f"<s>[INST] You are a senior financial analyst. Using this company data:\n\n"
        f"{summary}\n\n"
        f"Answer this question in detail: {question} [/INST]"
    )
    return infer(prompt)

def compare_companies(ticker1: str, ticker2: str) -> str:
    prompt_template = lambda t: (
        f"<s>[INST] Summarise the financial health and outlook of {t} "
        f"as a senior equity analyst. Be concise. [/INST]"
    )
    s1 = infer(prompt_template(ticker1))
    s2 = infer(prompt_template(ticker2))
    return f"### {ticker1.upper()}\n{s1}\n\n---\n\n### {ticker2.upper()}\n{s2}"

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="📈 FinSight — Financial AI Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 FinSight — Financial AI Analyst\nPowered by Mistral-7B fine-tuned on FinancialPhraseBank + SEC 10-K data")

    with gr.Tab("💬 Sentiment Analysis"):
        gr.Markdown("Classify financial news or earnings call snippets.")
        txt_in  = gr.Textbox(label="Financial Text", lines=4,
                             placeholder="Paste earnings call text, news headline, or analyst note…")
        sent_btn = gr.Button("Analyze Sentiment")
        sent_out = gr.Textbox(label="Result", lines=6)
        sent_btn.click(analyze_sentiment, inputs=txt_in, outputs=sent_out)

    with gr.Tab("🔍 Company Deep Dive"):
        gr.Markdown("Ask questions about any publicly listed company.")
        with gr.Row():
            ticker_in = gr.Textbox(label="Ticker Symbol (e.g. AAPL)", value="AAPL")
            q_in      = gr.Textbox(label="Your question",
                                   value="What are the main growth drivers?")
        dive_btn  = gr.Button("Run Analysis")
        dive_out  = gr.Textbox(label="Analysis", lines=10)
        dive_btn.click(analyze_filing, inputs=[ticker_in, q_in], outputs=dive_out)

    with gr.Tab("⚖️ Compare Companies"):
        gr.Markdown("Side-by-side AI comparison of two companies.")
        with gr.Row():
            t1 = gr.Textbox(label="Ticker 1", value="MSFT")
            t2 = gr.Textbox(label="Ticker 2", value="GOOGL")
        cmp_btn = gr.Button("Compare")
        cmp_out = gr.Markdown()
        cmp_btn.click(compare_companies, inputs=[t1, t2], outputs=cmp_out)

demo.launch(share=True)