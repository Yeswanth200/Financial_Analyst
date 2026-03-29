"""
Project : Financial Sentiment & Report Analyst
================================================
Datasets : FinancialPhraseBank + SEC 10-K filings (via SEC EDGAR)
Base Model: mistralai/Mistral-7B-Instruct-v0.2
Tasks    : Sentiment (bullish/bearish/neutral) + Open-ended QA
Technique: QLoRA + PEFT
"""

import torch
import json
import requests
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./financial_analyst_model"
MAX_SEQ    = 1024
EPOCHS     = 3
LR         = 2e-4

# ─── Quantisation ─────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ─── Dataset 1: FinancialPhraseBank sentiment ─────────────────────────────────
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def build_sentiment_dataset() -> Dataset:
    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
    records = []
    for item in ds:
        sent  = item["sentence"]
        label = LABEL_MAP[item["label"]]
        prompt = (
            f"<s>[INST] You are a financial analyst. Classify the sentiment of the "
            f"following financial statement as POSITIVE, NEGATIVE, or NEUTRAL. "
            f"Then give a one-sentence reasoning.\n\n"
            f"Statement: {sent} [/INST] "
            f"Sentiment: {label.upper()}\nReasoning: The statement indicates "
            f"{'positive market conditions' if label=='positive' else 'negative market conditions' if label=='negative' else 'neutral financial conditions'}. </s>"
        )
        records.append({"text": prompt})
    return Dataset.from_list(records)

# ─── Dataset 2: SEC 10-K QA pairs ─────────────────────────────────────────────
SEC_QA_TEMPLATES = [
    ("What are the primary risk factors mentioned?",
     "risk_factors"),
    ("Summarise the company's revenue performance.",
     "revenue_summary"),
    ("What does management say about future outlook?",
     "outlook"),
    ("Are there any legal proceedings or liabilities?",
     "legal"),
]

def fetch_10k_text(ticker: str) -> str:
    """Fetch latest 10-K filing text from SEC EDGAR (simplified)."""
    headers = {"User-Agent": "research@example.com"}
    cik_url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
    try:
        r = requests.get(cik_url, headers=headers, timeout=10)
        data = r.json()
        filings = data["filings"]["recent"]
        idx = next(i for i, f in enumerate(filings["form"]) if f == "10-K")
        accession = filings["accessionNumber"][idx].replace("-", "")
        doc = filings["primaryDocument"][idx]
        url = f"https://www.sec.gov/Archives/edgar/data/{data['cik']}/{accession}/{doc}"
        text_r = requests.get(url, headers=headers, timeout=15)
        return text_r.text[:5000]
    except Exception as e:
        return f"Error fetching 10-K: {e}"

def build_sec_qa_dataset(sample_texts: list[str]) -> Dataset:
    """Generate QA pairs from 10-K excerpts."""
    records = []
    for text in sample_texts:
        for question, _ in SEC_QA_TEMPLATES:
            prompt = (
                f"<s>[INST] You are a financial analyst. Answer the following "
                f"question based on the SEC 10-K filing excerpt.\n\n"
                f"Filing:\n{text[:2000]}\n\n"
                f"Question: {question} [/INST] "
                f"Based on the filing, {question.lower().rstrip('?')} can be assessed as follows: "
                f"[detailed analysis of the filing content]. </s>"
            )
            records.append({"text": prompt})
    return Dataset.from_list(records)

# ─── Build combined dataset ───────────────────────────────────────────────────
sentiment_ds = build_sentiment_dataset()
print(f"Sentiment samples : {len(sentiment_ds)}")

# For SEC QA, you can fetch real tickers or use saved texts:
# sec_texts = [fetch_10k_text("AAPL"), fetch_10k_text("MSFT")]
# sec_ds    = build_sec_qa_dataset(sec_texts)
# combined  = concatenate_datasets([sentiment_ds, sec_ds])
combined = sentiment_ds   # extend with sec_ds when available

split = combined.train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

# ─── Training ─────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=50,
    learning_rate=LR,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Financial model saved to {OUTPUT_DIR}")