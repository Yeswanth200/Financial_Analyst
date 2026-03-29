# 🤖 LLM Fine-Tuning Project

Production-ready AI/ML project demonstrating domain-specific LLM fine-tuning
using QLoRA + PEFT on Mistral-7B.

---

## 📁 Project Structure

```
├── project3_financial_analyst/
│   ├── train.py          # Fine-tuning on FinancialPhraseBank + SEC 10-K
│   ├── gradio_app.py     # Gradio demo (sentiment + company deep dive)
│   └── evaluate.py       # F1 / BLEU / ROUGE evaluation
│
└── requirements.txt
```

---
## 💰 Project — Financial Sentiment & Report Analyst

| Item | Detail |
|------|--------|
| **Base Model** | mistralai/Mistral-7B-Instruct-v0.2 |
| **Datasets** | FinancialPhraseBank + SEC EDGAR 10-K filings |
| **Technique** | QLoRA + PEFT, multi-task instruction tuning |
| **Tasks** | Sentiment (POSITIVE/NEGATIVE/NEUTRAL) + financial QA |
| **Interface** | Gradio demo with live yFinance data integration |
| **Evaluation** | Macro F1 (sentiment) + BLEU/ROUGE (QA) |

### Quick Start
```bash
# Train
python project_financial_analyst/train.py

# Evaluate
python project_financial_analyst/evaluate.py

# Launch Gradio demo
python project_financial_analyst/gradio_app.py
```
---

## ⚙️ Common Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to Hugging Face (for gated models)
huggingface-cli login

# 3. GPU requirements
# Recommended: NVIDIA T4 (16GB) or A100 (40GB)
# Minimum    : 16GB VRAM with QLoRA 4-bit
```

## 🔑 Key Technologies

`QLoRA` · `LoRA/PEFT` · `Mistral-7B` · `Hugging Face Transformers` · `TRL/SFTTrainer`
`LangChain` · `FAISS` · `RAG` · `Streamlit` · `Gradio` · `yFinance` · `BitsAndBytes`
`PubMedQA` · `CUAD` · `FinancialPhraseBank` · `SEC EDGAR`

---
