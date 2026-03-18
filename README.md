> **Team members:** For task tracking, milestones, and workflow protocols, please refer to our GitHub Project Board.

# 🛡️ Guardian Recruit – Fraud Detection System

**University of North Texas | DTSC 5082 Capstone**  
**Team:** 
- Isagani Julian Hernandez 
- Hemanth Kumar Gunda
- Kusuma Satya Sreeja Chalasani
- Srijitha Ungarala

---

## 1. Project Overview
Guardian Recruit is a dual-stream hybrid machine learning system for detecting fraudulent job listings. We combine:
- **NLP (BERT):** Text semantics
- **Statistical Outlier Detection (Isolation Forest):** Metadata anomalies
- **Fusion Layer (XGBoost + SHAP):** Final decision and explainability

---

## 2. Directory & Workspace Structure

```
Guardian-Recruit/
├── .venv/                # Local Python Virtual Environment (Ignored by Git)
├── data/                 # Data Directory (Ignored by Git - Sync via Google Drive)
│   ├── raw/              # Original emscad_dataset.csv
│   ├── processed/        # Stratified train.csv, val.csv, test.csv
│   └── external/         # Scraped 2026 job listings
├── models/               # Saved Model Weights (Ignored by Git - Sync via Google Drive)
│   ├── nlp_bert.pth      # Hemanth/Srijitha's saved model
│   ├── outlier_forest.pkl # Kusuma's saved model
│   └── fusion_xgb.json   # Isagani's final Fusion model
├── notebooks/            # Experimental & Training Notebooks (Google Colab)
│   ├── 01_initial_eda.ipynb             # Lead: Data understanding & splitting
│   ├── 02_nlp_stream_training.ipynb     # Hemanth: BERT/RoBERTa semantics
│   ├── 03_outlier_modeling.ipynb        # Kusuma: Isolation Forest/Anomaly detection
│   ├── 04_fusion_layer_shap.ipynb       # Lead: XGBoost integration & XAI
│   └── 05_live_scraper_test.ipynb       # Lead: 2026 Validation
├── src/                  # Production-ready Python Scripts
│   ├── __init__.py
│   ├── preprocessing.py  # Data cleaning & feature engineering (GuardianCleaner)
│   ├── scraper.py        # 2026 Data collection ETL script (Lead Task: S-2)
│   ├── nlp_pipeline.py   # Text preprocessing utilities
│   ├── nlp_stream.py     # Stream A inference: predict_proba(text) → float (Task A-7)
│   ├── outlier_logic.py  # Outlier detection utilities
│   ├── outlier_stream.py # Stream B inference: anomaly_score(row) → float (Task B-8)
│   ├── fusion_engine.py  # XGBoost scoring & SHAP generation utilities
│   ├── fusion_layer.py   # Lead inference: predict(job_posting) → dict (Task L-7)
│   └── main.py           # End-to-end pipeline integration (Task L-8)
├── .gitignore            # Tells Git to ignore /data, /models, and /.venv
├── README.md             # Project Roadmap & Documentation
└── requirements.txt      # Project dependencies (Pandas, Scikit-Learn, XGBoost, etc.)
```

---

## 3. Team Roles & Responsibilities
| Member    | Stream         | Task                              | Branch Name            |
|-----------|---------------|-----------------------------------|-----------------------|
| Isagani   | Fusion Layer   | XGBoost, SHAP, & Data Split       | feature/fusion-layer   |
| Hemanth   | NLP Stream     | BERT/RoBERTa Modeling             | feature/nlp-semantics  |
| Srijitha  | NLP Stream     | Preprocessing & Linguistic EDA    | feature/nlp-semantics  |
| Kusuma    | Outlier Stream | Isolation Forest & Metadata        | feature/outlier-detection |

---

## 4. Getting Started (Internal Protocol)
### Step 1: Clone the Repo
```bash
git clone https://github.com/[your-username]/Guardian-Recruit.git
cd Guardian-Recruit
```

### Step 2: Sync with Google Drive
- Open your assigned Google Colab notebook
- Mount the shared team drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Load data from `/data` in the shared drive (use `train.csv` only)

### Step 3: Branching Rule
- **NEVER** commit directly to `main`
- Create a branch for your feature:
```bash
git checkout -b feature/your-task-name
```
- When finished, push your branch and open a Pull Request (PR) for review

---

## 5. Project Roadmap

### ✅ Phase 1 Checklist (Complete)
- [x] Initial Project Skeleton created
- [x] Virtual Environment setup
- [x] Master Data Split (Isagani)
- [x] Missing Value EDA (Kusuma)
- [x] Keyword Frequency EDA (Hemanth/Srijitha)
- [x] Data Type Consolidation & `ffffffff` handling (Isagani)
- [x] Class Balance Verification across Train/Val/Test splits (Isagani)
- [x] NLP Visual Discovery — Histograms & Word Clouds (Hemanth/Srijitha)
- [x] NLP Descriptive Statistics — Text Length & Frequency (Hemanth/Srijitha)
- [x] NLP Colab Integration & Key Insights in `01_Initial_EDA.ipynb` (Hemanth/Srijitha)
- [x] Metadata Visual Discovery — Box Plots & Location Heatmap (Kusuma)
- [x] Statistical Correlation Analysis — Correlation Matrix (Kusuma)
- [x] Outlier Documentation & Colab Integration in `01_Initial_EDA.ipynb` (Kusuma)

---

## Phase 2 Project Board

> **Board Status Key:** 🔲 Backlog · 🔄 In Progress · ✅ Done  
> **Priority Key:** 🔴 High · 🟡 Medium · 🟢 Low

---

### 🔵 Stream A — NLP Stream (Hemanth + Srijitha)
> **Notebook:** `02_nlp_stream_training.ipynb` · **Output:** `src/nlp_stream.py` + `models/nlp_bert.pth`

| # | Task | Owner | Priority | Timeline | Status |
|---|------|-------|----------|----------|--------|
| A-1 | Text preprocessing & feature concatenation (title + description + requirements + benefits) | Srijitha | 🔴 High | Week 4 | 🔲 Backlog |
| A-2 | BERT/RoBERTa tokenizer setup & DataLoader construction for Colab GPU | Hemanth | 🔴 High | Week 4 | 🔲 Backlog |
| A-3 | Fine-tune BERT/RoBERTa on `train.csv` using Colab GPU runtime | Hemanth | 🔴 High | Weeks 4–5 | 🔲 Backlog |
| A-4 | Evaluate model on `val.csv` — target F1 ≥ 0.85 for fraud class | Hemanth | 🔴 High | Week 6 | 🔲 Backlog |
| A-5 | Hyperparameter tuning (learning rate, epochs, batch size) to meet F1 threshold | Hemanth | 🟡 Medium | Week 6 | 🔲 Backlog |
| A-6 | Export trained model weights as `nlp_bert.pth` to Google Drive `/models/` | Hemanth | 🔴 High | Week 7 | 🔲 Backlog |
| A-7 | Port inference logic — `predict_proba(text) → float` — to `src/nlp_stream.py` | Srijitha | 🔴 High | Week 7 | 🔲 Backlog |
| A-8 | Unit test `predict_proba()` with sample real and fraudulent postings | Srijitha | 🟡 Medium | Week 7 | 🔲 Backlog |

**Stream A — Definition of Done:**
- [ ] `src/nlp_stream.py` exports a callable `predict_proba(text: str) -> float`
- [ ] `models/nlp_bert.pth` is saved to the shared Google Drive `/models/` folder
- [ ] F1 ≥ 0.85 on the fraud class using `val.csv`
- [ ] All code and key insights documented in `02_nlp_stream_training.ipynb`

---

### 🟠 Stream B — Outlier Detection Stream (Kusuma)
> **Notebook:** `03_outlier_modeling.ipynb` · **Output:** `src/outlier_stream.py` + `models/outlier_forest.pkl`

| # | Task | Owner | Priority | Timeline | Status |
|---|------|-------|----------|----------|--------|
| B-1 | Feature engineering — encode salary range, education level, `employment_type`, `has_company_logo` | Kusuma | 🔴 High | Week 4 | 🔲 Backlog |
| B-2 | Build numeric feature matrix from metadata columns for anomaly models | Kusuma | 🔴 High | Week 4 | 🔲 Backlog |
| B-3 | Fit `IsolationForest` on `train.csv` metadata features | Kusuma | 🔴 High | Week 5 | 🔲 Backlog |
| B-4 | Fit `LocalOutlierFactor` on `train.csv` metadata features | Kusuma | 🟡 Medium | Week 5 | 🔲 Backlog |
| B-5 | Tune `contamination` parameter; evaluate anomaly scores against `val.csv` fraud labels | Kusuma | 🔴 High | Week 6 | 🔲 Backlog |
| B-6 | Compare IsolationForest vs. LOF — document which yields better fraud-signal alignment | Kusuma | 🟡 Medium | Week 6 | 🔲 Backlog |
| B-7 | Export best model as `outlier_forest.pkl` to Google Drive `/models/` | Kusuma | 🔴 High | Week 7 | 🔲 Backlog |
| B-8 | Port inference logic — `anomaly_score(row) → float` — to `src/outlier_stream.py` | Kusuma | 🔴 High | Week 7 | 🔲 Backlog |
| B-9 | Unit test `anomaly_score()` with sample real and fraudulent metadata rows | Kusuma | 🟡 Medium | Week 7 | 🔲 Backlog |

**Stream B — Definition of Done:**
- [ ] `src/outlier_stream.py` exports a callable `anomaly_score(row: pd.Series) -> float`
- [ ] `models/outlier_forest.pkl` is saved to the shared Google Drive `/models/` folder
- [ ] Anomaly scores show meaningful separation between real and fraudulent postings on `val.csv`
- [ ] All code and key insights documented in `03_outlier_modeling.ipynb`

---

### 🟢 Lead Task — Fusion Layer & 2026 Live Scraper (Isagani)
> **Notebooks:** `04_fusion_layer_shap.ipynb`, `05_live_scraper_test.ipynb` · **Output:** `src/fusion_layer.py`, `src/scraper.py`, `models/fusion_xgb.json`

#### Sub-stream: Fusion Layer (XGBoost + SHAP)
> ⚠️ **Dependency:** Requires Stream A (`predict_proba`) and Stream B (`anomaly_score`) outputs before starting L-3 onward.

| # | Task | Owner | Priority | Timeline | Status |
|---|------|-------|----------|----------|--------|
| L-1 | Define meta-feature schema: `[nlp_score, outlier_score, desc_len, has_logo, ...]` | Isagani | 🔴 High | Week 5 | 🔲 Backlog |
| L-2 | Build meta-feature extraction pipeline in `04_fusion_layer_shap.ipynb` | Isagani | 🔴 High | Week 5 | 🔲 Backlog |
| L-3 | Train `XGBClassifier` on meta-features using `train.csv` stream outputs *(depends on A-7, B-8)* | Isagani | 🔴 High | Week 6 | 🔲 Backlog |
| L-4 | Evaluate XGBoost fusion model on `val.csv` — target F1 ≥ 0.88 overall | Isagani | 🔴 High | Week 6 | 🔲 Backlog |
| L-5 | Integrate SHAP explainability — generate feature importance plots | Isagani | 🟡 Medium | Week 7 | 🔲 Backlog |
| L-6 | Export fusion model as `fusion_xgb.json` to Google Drive `/models/` | Isagani | 🔴 High | Week 7 | 🔲 Backlog |
| L-7 | Port full prediction pipeline to `src/fusion_layer.py` — `predict(job_posting) → dict` | Isagani | 🔴 High | Week 7 | 🔲 Backlog |
| L-8 | Integrate all streams in `src/main.py` — end-to-end scoring pipeline | Isagani | 🟡 Medium | Week 8 | 🔲 Backlog |

#### Sub-stream: 2026 Live Scraper (ETL)

| # | Task | Owner | Priority | Timeline | Status |
|---|------|-------|----------|----------|--------|
| S-1 | Identify target public job boards (e.g., Indeed, LinkedIn public listings) and scraping strategy | Isagani | 🔴 High | Week 4 | 🔲 Backlog |
| S-2 | Build ETL pipeline in `src/scraper.py` to collect raw job postings | Isagani | 🔴 High | Weeks 4–5 | 🔲 Backlog |
| S-3 | Normalize scraped fields to match EMSCAD schema (title, description, salary_range, etc.) | Isagani | 🔴 High | Weeks 5–6 | 🔲 Backlog |
| S-4 | Store normalized output as CSV to `data/external/` on Google Drive | Isagani | 🟡 Medium | Week 6 | 🔲 Backlog |
| S-5 | Run fusion model against 2026 scraped data in `05_live_scraper_test.ipynb` *(depends on L-7)* | Isagani | 🔴 High | Weeks 7–8 | 🔲 Backlog |
| S-6 | Document real-world validation results and SHAP explanations for flagged postings | Isagani | 🟡 Medium | Week 8 | 🔲 Backlog |

**Lead Task — Definition of Done:**
- [ ] `src/fusion_layer.py` exports `predict(job_posting: dict) -> dict` (score + SHAP explanation)
- [ ] `src/main.py` runs an end-to-end pipeline loading all three models and scoring a posting
- [ ] `models/fusion_xgb.json` is saved to the shared Google Drive `/models/` folder
- [ ] `src/scraper.py` can collect and normalize ≥ 50 live job postings from a public source
- [ ] 2026 live data results documented in `05_live_scraper_test.ipynb`
- [ ] SHAP plots generated for at least 5 flagged fraudulent postings

---

### 📅 Phase 2 Week-by-Week Timeline

| Week | Stream A (Hemanth/Srijitha) | Stream B (Kusuma) | Lead Task (Isagani) |
|------|----------------------------|-------------------|---------------------|
| **4** | A-1: Text preprocessing · A-2: Tokenizer setup | B-1: Feature engineering · B-2: Feature matrix | S-1: Scraper strategy · L-1: Meta-feature schema |
| **5** | A-3: BERT/RoBERTa fine-tuning (Colab GPU) | B-3: Fit IsolationForest · B-4: Fit LOF | S-2: Build ETL pipeline · L-2: Meta-feature extraction |
| **6** | A-4: Evaluate on val.csv · A-5: Hyperparameter tuning | B-5: Tune contamination · B-6: Compare models | S-3: Normalize scraped fields · L-3: Train XGBoost · L-4: Evaluate fusion |
| **7** | A-6: Export `nlp_bert.pth` · A-7: Port to `src/nlp_stream.py` · A-8: Unit tests | B-7: Export `outlier_forest.pkl` · B-8: Port to `src/outlier_stream.py` · B-9: Unit tests | S-4: Store to Drive · L-5: SHAP plots · L-6: Export `fusion_xgb.json` · L-7: Port to `src/fusion_layer.py` |
| **8** | *(Buffer / Final Report)* | *(Buffer / Final Report)* | S-5: Live validation · S-6: Document results · L-8: `src/main.py` integration |

---

## 6. Contact & Syncs
- **Weekly Sync:** [Day/Time] via [Teams]

---

## 7. Best Practices
- **Strict Pathing:** Use relative paths in all notebooks.
  - Example: `pd.read_csv('../data/processed/train.csv')` instead of absolute paths like `C:/Users/...`
- **Model Exports:** All trained models must be exported to the `/models` folder in Google Drive. This allows the Fusion Layer to load them without retraining.
- **Data Isolation:** Never upload anything from the `/data` or `/models` folders to GitHub. We use Google Drive for storage and GitHub for logic.


