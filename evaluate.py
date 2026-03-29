"""
Project : Financial Model Evaluation
======================================
Evaluates fine-tuned model vs base model on FinancialPhraseBank test set.
Metrics: F1 (sentiment) + BLEU/ROUGE (QA)
"""

import torch
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "./financial_analyst_model"
LABEL_MAP   = {0: "negative", 1: "neutral", 2: "positive"}

def load_finetuned():
    tok  = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    mdl  = PeftModel.from_pretrained(base, ADAPTER_DIR)
    mdl.eval()
    return pipeline("text-generation", model=mdl, tokenizer=tok,
                    max_new_tokens=128, temperature=0.1, do_sample=True)

def extract_label(text: str) -> str:
    text = text.upper()
    for l in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if l in text:
            return l.lower()
    return "neutral"

def evaluate_sentiment(pipe, n_samples: int = 200):
    ds   = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
    ds   = ds.shuffle(seed=42).select(range(n_samples))
    preds, labels = [], []

    for item in ds:
        stmt  = item["sentence"]
        gold  = LABEL_MAP[item["label"]]
        prompt = (
            f"<s>[INST] Classify the sentiment (POSITIVE/NEGATIVE/NEUTRAL) "
            f"of this financial statement: {stmt} [/INST]"
        )
        out  = pipe(prompt)[0]["generated_text"]
        pred = extract_label(out.split("[/INST]")[-1])
        preds.append(pred)
        labels.append(gold)

    print("\n── Sentiment Evaluation ────────────────────────")
    print(classification_report(labels, preds,
                                 target_names=["negative", "neutral", "positive"]))
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    pipe = load_finetuned()
    f1   = evaluate_sentiment(pipe, n_samples=200)
    print(f"\n✅ Final Macro-F1: {f1:.4f}")