# 🛡️ Guardian Recruit – Fraud Detection System

**University of North Texas | DTSC 5082 Capstone**  
**Team:** 
- Isagani Hernandez
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

## 2. Data & Workspace Layout
### A. GitHub (Code & Architecture)
- `src/`: Final Python scripts (`nlp_stream.py`, `outlier_stream.py`, `fusion_layer.py`)
- `requirements.txt`: Run `pip install -r requirements.txt` to align your environment
- `.gitignore`: DO NOT upload `.venv/` or raw `data/` to GitHub

### B. Google Drive (Data & Training)
All heavy files are stored in our shared Google Drive folder: [Insert Link to Shared Drive Folder Here]
- `/data`: `train.csv`, `val.csv`, `test.csv`
- `/notebooks`: Experimental Google Colab notebooks
- `/models`: Saved model weights (e.g., BERT `.pth` files)

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
- [ ] Master Data Split (Isagani)
- [ ] Missing Value EDA (Kusuma)
- [ ] Keyword Frequency EDA (Hemanth/Srijitha)

---

## 6. Contact & Syncs
- **Weekly Sync:** [Day/Time] via [Teams]


