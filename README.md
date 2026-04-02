# Guardian Recruit — Fraud Detection System

**University of North Texas | DTSC 5082 Capstone**

**Team:**
- Isagani Julian Hernandez — Fusion Layer, SHAP Explainability, Data Pipeline
- Hemanth Kumar Gunda — NLP Stream (BERT)
- Srijitha Ungarala — NLP Stream (Preprocessing & Linguistic EDA)
- Kusuma Satya Sreeja Chalasani — Outlier Detection Stream (IsolationForest)

---

## Overview

Guardian Recruit is a hybrid AI system for detecting fraudulent job postings. It combines three independent streams into a single fraud probability score with explainability.

```
Job Posting
  ├── Stream A: BERT fine-tuned NLP      → bert_score    (0.0–1.0)
  ├── Stream B: IsolationForest outlier  → outlier_score (float)
  └── Fusion:   XGBoost meta-classifier  → fraud_score   (0.0–1.0)
                                            + SHAP reasoning summary
```

**Model Performance (Validation Set):**
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9718 |
| Accuracy | 0.99 |
| Fraud F1 | 0.8439 |
| False Negatives | 30 |
| False Positives | 7 |

---

## Directory Structure

```
Guardian-Recruit-Fraud-Detection/
├── app.py                          # Streamlit demo app
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/                        # Original EMSCAD dataset (2014)
│   ├── processed/                  # Train / val / test splits + augmented training set
│   │   ├── train_clean_v1.csv      # Cleaned training data (8,696 rows)
│   │   ├── train.csv / val.csv / test.csv
│   │   ├── FINAL_AUGMENTED_TRAINING.csv   # SMOTENC + synthetic rows (17,342 rows)
│   │   └── synthetic_fraud_2026.csv       # 200 template-generated 2026-era fraud rows
│   ├── external/                   # Scraped 2026 legitimate job listings
│   └── chroma_db/                  # ChromaDB vector store for RAG explainer
│
├── models/
│   ├── nlp_bert.pth                # Fine-tuned BERT weights (Stream A)
│   ├── outlier_forest.pkl          # IsolationForest model (Stream B)
│   └── fusion_xgb.json             # XGBoost fusion model (Fusion Layer)
│
├── notebooks/
│   ├── 01_Initial_EDA.ipynb        # Data understanding & splitting
│   ├── 02_nlp_stream_training.ipynb # BERT fine-tuning (Colab T4 GPU)
│   ├── 03_outlier_modeling.ipynb   # IsolationForest training
│   ├── 04_fusion_layer_shap.ipynb  # XGBoost fusion + SHAP (Colab T4 GPU)
│   ├── 05_live_scraper_test.ipynb  # 2026 live data validation
│   ├── 06_data_augmentation.ipynb  # SMOTENC + synthetic fraud generation
│   └── sandbox/                    # Individual exploration notebooks
│
├── src/
│   ├── main.py                     # End-to-end pipeline: score(job_posting) → dict
│   ├── preprocessing.py            # Data cleaning (GuardianCleaner)
│   ├── nlp_stream.py               # Stream A: predict_proba(text) → float
│   ├── outlier_stream.py           # Stream B: anomaly_score(row) → float
│   ├── fusion_layer.py             # Fusion: predict(row) → dict
│   ├── meta_features.py            # Adversarial features: domain age, perplexity, platform risk
│   ├── text_signals.py             # Keyword signal engine (Tier 1 + Tier 2 + 2026-era)
│   ├── shap_explainer.py           # SHAP reasoning summaries per prediction
│   ├── explainer.py                # LLM narrative explanation (Groq → Ollama → template)
│   ├── vector_store.py             # ChromaDB RAG for similar fraud case retrieval
│   ├── scraper.py                  # 2026 job listing ETL pipeline
│   └── pipeline.py                 # Batch scoring utilities
│
├── scripts/
│   ├── generate_synthetic_fraud.py # Generate 2026-era synthetic fraud rows
│   ├── validate_signals.py         # Validate keyword signal coverage on training data
│   └── smoke_test.py               # Quick end-to-end sanity check
│
└── tests/
    └── test_outlier_stream.py      # Unit tests for outlier stream
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add model files
Place the following in `models/` (download from shared Google Drive):
- `nlp_bert.pth`
- `outlier_forest.pkl`
- `fusion_xgb.json`

### 3. Score a job posting
```python
from src.main import score
from src.shap_explainer import explain_shap

result = score({
    "title": "Remote Operations Assistant",
    "description": "Earn up to $600/week. Contact us on Telegram. Start this week.",
    "has_company_logo": 0,
    "has_questions": 0,
})

print(result["label"])        # FRAUD or LEGITIMATE
print(result["fraud_score"])  # 0.0 – 1.0

shap = explain_shap(result)
print(shap["reasoning_summary"])
# "Flagged because: NLP fraud pattern score (+86%), No company logo (+7%), ..."
```

### 4. Run the demo app
```bash
streamlit run app.py
```

### 5. Run smoke tests
```bash
python src/main.py
python scripts/smoke_test.py
```

---

## What Was Built

### Phase 1 — EDA & Data Splitting
- Class-stratified train/val/test splits from EMSCAD dataset (17,880 rows, 4.85% fraud rate)
- Missing value analysis, keyword frequency EDA, metadata visualisations

### Phase 2 — Dual-Stream Models
- **Stream A (BERT):** Fine-tuned `bert-base-uncased` on fraud detection, F1 0.8376 on val set
- **Stream B (IsolationForest):** Anomaly detection on 7 metadata features
- **Fusion Layer (XGBoost):** Meta-classifier combining both stream scores, ROC-AUC 0.9718

### Phase 3 — Adversarial Robustness & Explainability
- **Adversarial Synthesis:** 200 template-generated 2026-era fraud rows with 12 modern signal types (crypto salary, Telegram interviews, equipment deposits, task scams, etc.)
- **2026 Keyword Signals:** Extended `text_signals.py` with 8 new Tier 2 signal categories grounded in FBI IC3, FTC, and BBB 2024–2026 reports
- **New Meta-Features:** Domain age (WHOIS), text perplexity (GPT-2), platform risk (Telegram/WhatsApp/Signal detection)
- **SHAP Explainability:** Per-prediction reasoning summaries — *"Flagged because: AI-generated text likelihood (+38%), Domain age < 30 days (+27%)"*
- **RAG Explainer:** LLM narrative explanations grounded in similar known fraud cases (Groq → Ollama → template fallback)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| XGBoost for fusion | Handles small meta-feature matrix well; native SHAP support |
| BERT over TF-IDF | Captures semantic fraud patterns beyond keyword matching |
| Template synthesis over LLM generation | Reproducible, no API dependency, guaranteed signal injection |
| `domain_age_days = -1` as training fallback | WHOIS too slow for 17k rows; feature contributes at inference time on live postings |
| Threshold = 0.3 | Optimised for recall on fraud class — missing fraud is worse than a false alarm |

---

## Limitations

- Synthetic training data is template-generated, not real labelled 2026 fraud postings
- `domain_age_days` and `text_perplexity` used neutral fallback values during bulk training — SHAP importance is low for these features on the validation set but they activate at inference time
- Perplexity threshold (< 80 = AI-generated) is heuristic and not empirically calibrated
- WHOIS lookups fail silently for private domain registrations (~30% of domains)

---

## Environment Variables

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_...        # For LLM narrative explanations (free tier)
```

---

## Branching Convention
| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `feature/fusion-evaluation` | Current active branch |
| `feature/nlp-semantics` | Stream A NLP work |
| `feature/outlier-detection` | Stream B outlier work |
