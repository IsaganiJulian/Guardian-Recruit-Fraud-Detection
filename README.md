# Guardian Recruit Fraud Detection

This AI-powered system helps ensure job seekers' safety and mitigate potential harm from fraudulent job postings.

## Project Structure

```
.
├── data/          # Training and test data
├── notebooks/     # Jupyter notebooks for analysis
├── models/        # Trained model artifacts
├── src/           # Source code modules
│   ├── nlp_stream.py      # Text analysis module (BERT/RoBERTa)
│   ├── outlier_stream.py  # Metadata analysis module (Isolation Forest)
│   ├── fusion_layer.py    # Meta-classifier module (XGBoost + SHAP)
│   └── main.py            # System integration
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for dependencies:
- transformers
- scikit-learn
- xgboost
- shap
- torch
- numpy
- pandas

