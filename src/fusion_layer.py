"""
Fusion Layer Module - XGBoost Meta-Classifier
Combines outputs from NLP and outlier streams for final fraud detection.

Phase 2 Tasks:
  L-1: Define meta-feature schema [bert_score, outlier_score, has_logo, has_questions, desc_len]
  L-2: Build meta-feature extraction pipeline
  L-3: Train XGBClassifier on meta-features (depends on A-7, B-8)
  L-4: Evaluate fusion model on val.csv
  L-5: Integrate SHAP explainability — feature importance
  L-6: Export fusion_xgb.json to Google Drive /models/
  L-7: Port predict(job_posting) -> dict here (THIS FILE)  ← implemented
  L-8: Integrate all streams in src/main.py
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fusion_xgb.json')

# Meta-features fed into XGBoost
# Two stream scores + three lightweight metadata signals
META_FEATURES = ['bert_score', 'outlier_score', 'has_company_logo', 'has_questions', 'desc_len']

_fusion_model = None


def _load_model():
    global _fusion_model
    if _fusion_model is not None:
        return _fusion_model

    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    try:
        import xgboost as xgb
        # Load as Booster — avoids XGBClassifier._estimator_type issues in xgboost >= 2.x
        booster = xgb.Booster()
        booster.load_model(path)
        _fusion_model = booster
        return _fusion_model
    except Exception:
        return None


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all meta-features for a DataFrame of job postings.
    Runs both streams on every row and returns a meta-feature matrix.

    Args:
        df: DataFrame with raw job posting columns.

    Returns:
        DataFrame with columns matching META_FEATURES.
    """
    import outlier_stream
    import nlp_stream

    records = []
    for _, row in df.iterrows():
        bert_score    = nlp_stream.predict_proba_from_row(row)
        outlier_score = outlier_stream.anomaly_score(row)

        desc = row.get('description', '')
        desc_len = len(str(desc)) if pd.notna(desc) else 0

        records.append({
            'bert_score':       bert_score,
            'outlier_score':    outlier_score,
            'has_company_logo': int(row.get('has_company_logo', 0) or 0),
            'has_questions':    int(row.get('has_questions', 0) or 0),
            'desc_len':         float(desc_len),
        })

    return pd.DataFrame(records, columns=META_FEATURES)


def train(df: pd.DataFrame, save_path: str = MODEL_PATH) -> object:
    """
    Train the XGBoost fusion model on pre-labelled job postings.

    Computes stream scores for every row, then fits XGBoost on the
    meta-feature matrix. Saves the model to save_path.

    Args:
        df:        DataFrame with job posting columns + 'fraudulent' label.
        save_path: Where to write the trained model (JSON format).

    Returns:
        Fitted XGBClassifier.
    """
    from xgboost import XGBClassifier

    print(f'Building meta-features for {len(df):,} rows...')
    X = build_meta_features(df)
    y = df['fraudulent'].astype(int).values

    fraud_count  = y.sum()
    legit_count  = len(y) - fraud_count
    scale_weight = legit_count / max(fraud_count, 1)  # handles class imbalance

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save_model(save_path)
    print(f'Fusion model saved → {save_path}')

    global _fusion_model
    _fusion_model = model
    return model


def predict(row) -> dict:
    """
    Score a single job posting through the fusion layer.

    Args:
        row: dict or pd.Series with job posting fields.

    Returns:
        {
            "fraud_score":    float (0.0–1.0),
            "label":          "FRAUD" | "LEGITIMATE",
            "threshold":      float,
            "bert_score":     float,
            "outlier_score":  float,
            "features":       dict of all meta-features,
        }
    """
    import outlier_stream
    import nlp_stream

    THRESHOLD = 0.3  # lowered from 0.5 — optimised from BERT threshold sweep

    bert_score    = nlp_stream.predict_proba_from_row(row)
    outlier_score = outlier_stream.anomaly_score(row)

    if isinstance(row, dict):
        row = pd.Series(row)

    desc = row.get('description', '')
    desc_len = len(str(desc)) if pd.notna(desc) else 0

    features = {
        'bert_score':       bert_score,
        'outlier_score':    outlier_score,
        'has_company_logo': int(row.get('has_company_logo', 0) or 0),
        'has_questions':    int(row.get('has_questions', 0) or 0),
        'desc_len':         float(desc_len),
    }

    model = _load_model()

    if model is not None:
        import xgboost as xgb
        X   = pd.DataFrame([features], columns=META_FEATURES)
        dm  = xgb.DMatrix(X, feature_names=META_FEATURES)
        raw = model.predict(dm)          # Booster.predict returns probabilities for binary classification
        fraud_score = float(raw[0])
    else:
        # Fusion model not trained yet — fall back to BERT score
        fraud_score = bert_score

    label = 'FRAUD' if fraud_score >= THRESHOLD else 'LEGITIMATE'

    return {
        'fraud_score':   fraud_score,
        'label':         label,
        'threshold':     THRESHOLD,
        'bert_score':    bert_score,
        'outlier_score': outlier_score,
        'features':      features,
    }
