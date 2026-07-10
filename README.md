# 🏥 MultiModal MedAI Assistant

> AI-powered diagnostic support system combining CNN image analysis, NLP symptom classification, time-series vitals forecasting, and RAG-based report generation — all running locally for free.

**✨ v2:** Premium dark glassmorphism/aurora enterprise UI, full production
audit + caching, all button/page workflows hardened against crashes, and an
**optional OpenAI (GPT) report engine** you can toggle on with your own API
key (see `.env.example` / `.streamlit/secrets.toml.example`). See
`AUDIT_REPORT.md` and `VERIFICATION_REPORT.md` for the full changelog.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              MultiModal MedAI Assistant          │
├──────────────┬──────────────┬────────────────────┤
│  X-Ray CNN   │ Symptom NLP  │  Vitals Forecast   │
│ EfficientNet │  Zero-Shot   │  Prophet + LSTM     │
│    B0        │  BART-MNLI   │  Time Series        │
└──────┬───────┴──────┬───────┴────────┬───────────┘
       │              │                │
       └──────────────┼────────────────┘
                      ↓
           RAG Diagnostic Report
           FAISS + Flan-T5 LLM
                      ↓
           PDF Download (fpdf2)
```

---

## 📁 Project Structure

```
medai/
├── app.py                      ← Main Streamlit app (run this)
├── requirements.txt            ← All dependencies
├── README.md                   ← This file
└── utils/
    ├── __init__.py
    ├── xray_classifier.py      ← CNN X-ray classification (EfficientNet-B0)
    ├── symptom_analyzer.py     ← NLP symptom analysis (Zero-shot BART)
    ├── vitals_forecaster.py    ← Time-series forecasting (Prophet)
    ├── report_rag.py           ← RAG report generation (FAISS + Flan-T5)
    └── speech_handler.py       ← Voice input (Whisper)
```

---

## ⚙️ Setup & Run

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## ✨ Features — 4 Modules

### 🖼️ Module 1 — X-Ray Analysis (CNN)
- Upload any chest X-ray image (JPG/PNG)
- EfficientNet-B0 classifies into: Normal, Pneumonia, COVID-19, Tuberculosis
- Probability bar chart for all classes
- Severity level + recommended action
- **To improve:** Fine-tune on NIH ChestX-ray14 or CheXpert dataset

### 🗣️ Module 2 — Symptom Checker (NLP)
- Type or speak symptoms
- facebook/bart-large-mnli zero-shot classification
- Top-3 disease predictions with confidence scores
- Detected symptom keywords highlighted
- Recommended tests + care actions per disease

### 📈 Module 3 — Vitals Forecasting (Time Series)
- 4 vitals tracked: Heart Rate, Blood Pressure, SpO2, Temperature
- Facebook Prophet forecasts next 3 days
- Confidence intervals shown on chart
- Condition simulation: Normal / Pneumonia / COVID-19
- Overall patient risk score: Low / Medium / High / Critical

### 📄 Module 4 — Diagnostic Report (RAG)
- FAISS vector store built from medical knowledge base
- sentence-transformers embeddings for semantic retrieval
- google/flan-t5-base generates structured diagnostic summary
- Full patient summary table
- Downloadable PDF report with all findings

---

## 🧠 Models Used

| Model | Purpose |
|---|---|
| `EfficientNet-B0` | Chest X-ray CNN classification |
| `facebook/bart-large-mnli` | Zero-shot symptom classification |
| `facebook/prophet` | Vitals time-series forecasting |
| `openai/whisper-base` | Speech to text for voice input |
| `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings |
| `google/flan-t5-base` | Diagnostic report generation |
| `FAISS` | Vector similarity search |

---

## 🚀 Deploy Free

1. Push to GitHub
2. Go to **share.streamlit.io**
3. Connect repo → select `app.py`
4. Deploy in 60 seconds

---

## 📢 How to Explain in Interview

> *"I built a multimodal medical AI system with 4 integrated modules. First, a CNN using EfficientNet-B0 classifies chest X-rays into Normal, Pneumonia, COVID-19, or Tuberculosis. Second, a zero-shot NLP classifier using BART-MNLI analyzes patient-described symptoms and ranks probable diseases. Third, Facebook Prophet forecasts patient vitals like heart rate and SpO2 for the next 3 days and generates a risk score. Finally, a RAG pipeline using FAISS and sentence-transformers retrieves relevant medical knowledge and passes it to a Flan-T5 LLM to generate a structured diagnostic report — which is also exportable as a PDF. The entire system runs locally with no API cost using Streamlit as the UI."*

---

## ⚠️ Disclaimer

This project is built **for educational and research purposes only**. It is NOT intended for actual medical diagnosis or clinical use. Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Built By
**Nisma** — Final Year AI/ML Engineer
Stack: Python · PyTorch · HuggingFace · Prophet · FAISS · LangChain · Streamlit
