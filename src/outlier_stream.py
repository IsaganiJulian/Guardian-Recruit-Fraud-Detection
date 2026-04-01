"""
Outlier Stream Module - Isolation Forest for Metadata Analysis
Handles anomaly detection in job posting metadata using Isolation Forest.

Phase 2 Tasks:
  B-1: Feature engineering — encode salary, education, employment_type, has_company_logo
  B-2: Build numeric feature matrix from metadata columns
  B-3: Fit IsolationForest on train.csv metadata features
  B-4: Fit LocalOutlierFactor on train.csv metadata features
  B-5: Tune contamination param; evaluate on val.csv
  B-6: Compare IsolationForest vs. LOF
  B-7: Export outlier_forest.pkl to Google Drive /models/
  B-8: Port anomaly_score(row) -> float here (THIS FILE)
  B-9: Unit test anomaly_score() with sample rows
"""

import pandas as pd
import joblib
import os

# Encoding maps derived from train_clean_v1.csv category codes
# (alphabetical ordering of unique values, matching .astype('category').cat.codes)
emp_map = {
    'Contract': 0,
    'Full-time': 1,
    'Other': 2,
    'Part-time': 3,
    'Temporary': 4,
    'Unknown': 5,
}

edu_map = {
    'Associate Degree': 0,
    "Bachelor's Degree": 1,
    'Certification': 2,
    'Doctorate': 3,
    'High School or equivalent': 4,
    "Master's Degree": 5,
    'Professional': 6,
    'Some College Coursework Completed': 7,
    'Some High School Coursework': 8,
    'Unknown': 9,
    'Unspecified': 10,
    'Vocational': 11,
    'Vocational - Degree': 12,
    'Vocational - HS Diploma': 13,
}

# Load the model
# Path assumes this script is in /src and model is in /models
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/outlier_forest.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

model = load_model()

def _safe_int(val, default=0):
    """Convert t/f strings and numeric values to int safely."""
    if pd.isna(val):
        return default
    s = str(val).strip().lower()
    if s in ('t', 'true', '1'):
        return 1
    if s in ('f', 'false', '0'):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default

def preprocess_row(row):
    """
    Builds the 7-feature numeric matrix used by the IsolationForest.

    Features:
      salary_processed   — midpoint of salary_range (fallback: median 44000)
      employment_type    — label-encoded (0-5)
      has_company_logo   — binary (0/1)
      has_questions      — binary (0/1); fraudsters rarely include screening questions
      telecommuting      — binary (0/1)
      required_education — label-encoded (0-13)
      desc_len           — character length of description; very short = suspicious
    """
    # Categoricals
    emp_type = str(row.get('employment_type', 'Unknown'))
    req_edu  = str(row.get('required_education', 'Unknown'))
    if pd.isna(emp_type) or emp_type == 'nan': emp_type = 'Unknown'
    if pd.isna(req_edu)  or req_edu  == 'nan': req_edu  = 'Unknown'

    # Salary midpoint
    salary = row.get('salary_range', '')
    salary_processed = 44000.0  # training median fallback
    if isinstance(salary, str) and '-' in salary:
        try:
            low, high = salary.split('-')
            salary_processed = (float(low) + float(high)) / 2
        except (ValueError, TypeError):
            pass

    # desc_len — compute on the fly if not pre-computed (e.g. val.csv)
    desc_len = row.get('desc_len', None)
    if desc_len is None or pd.isna(desc_len):
        desc = row.get('description', '')
        desc_len = len(str(desc)) if pd.notna(desc) else 0
    desc_len = float(desc_len)

    feature_dict = {
        'salary_processed':   salary_processed,
        'employment_type':    emp_map.get(emp_type, 5),
        'has_company_logo':   _safe_int(row.get('has_company_logo', 0)),
        'has_questions':      _safe_int(row.get('has_questions', 0)),
        'telecommuting':      _safe_int(row.get('telecommuting', 0)),
        'required_education': edu_map.get(req_edu, 9),
        'desc_len':           desc_len,
    }

    return pd.DataFrame([feature_dict])

def anomaly_score(row):
    """Returns the decision function score (lower is more anomalous)."""
    if model is None:
        return 0.0
    input_df = preprocess_row(row)
    return float(model.decision_function(input_df)[0])

def anomaly_predict(row):
    """Returns 1 for normal, -1 for anomaly."""
    if model is None:
        return 1
    input_df = preprocess_row(row)
    return int(model.predict(input_df)[0])

