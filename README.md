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
│   ├── scraper.py        # 2026 Data collection script
│   ├── nlp_pipeline.py   # Text preprocessing and scoring logic
│   ├── outlier_logic.py  # Statistical anomaly logic
│   └── fusion_engine.py  # Final XGBoost scoring & SHAP generation
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

## 5. Project Roadmap (Phase 1 Checklist)
- [x] Initial Project Skeleton created
- [x] Virtual Environment setup
- [x] Master Data Split (Isagani)
- [x] Missing Value EDA (Kusuma)
- [x] Keyword Frequency EDA (Hemanth/Srijitha)

## Project Roadmap (Phase 2 Checklist)

### Stream A — NLP (Hemanth + Srijitha)
- [ ] Text preprocessing & feature concat (Srijitha) — Week 4
- [ ] BERT/RoBERTa fine-tuning on train.csv in Colab GPU (Hemanth) — Weeks 4–5
- [ ] Evaluate on val.csv (F1 ≥ 0.85 for fraud class) — Week 6
- [ ] Export `nlp_bert.pth` to Google Drive `/models/` — Week 7
- [ ] Port `predict_proba(text) → float` to `src/nlp_stream.py` — Week 7

### Stream B — Outlier Detection (Kusuma)
- [ ] Feature engineering for salary, education, logo, employment_type — Week 4
- [ ] Fit IsolationForest + LocalOutlierFactor on train.csv — Week 5
- [ ] Tune `contamination` param; validate on val.csv — Week 6
- [ ] Export `outlier_forest.pkl` to Google Drive `/models/` — Week 7
- [ ] Port `anomaly_score(row) → float` to `src/outlier_logic.py` — Week 7

### Lead Task — 2026 Live Scraper (Isagani)
- [ ] Build ETL pipeline targeting public job boards — Weeks 4–5
- [ ] Normalize scraped fields to match emscad schema — Weeks 5–6
- [ ] Store output to `data/external/` on Google Drive — Week 6
- [ ] Run real-world validation in `05_live_scraper_test.ipynb` — Weeks 7–8

---

## 6. Contact & Syncs
- **Weekly Sync:** [Day/Time] via [Teams]

---

## 7. Best Practices
- **Strict Pathing:** Use relative paths in all notebooks.
  - Example: `pd.read_csv('../data/processed/train.csv')` instead of absolute paths like `C:/Users/...`
- **Model Exports:** All trained models must be exported to the `/models` folder in Google Drive. This allows the Fusion Layer to load them without retraining.
- **Data Isolation:** Never upload anything from the `/data` or `/models` folders to GitHub. We use Google Drive for storage and GitHub for logic.


