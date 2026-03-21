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

# 1. Load the model
# Path assumes this script is in /src and model is in /models
# Trained model: IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
# Selected in 03_outlier_optimization.ipynb (B-6) — best F1=0.1778, Recall=0.2637 on val.csv
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/outlier_forest.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # pkl may be an empty placeholder (e.g. local dev without Drive mounted)
            return None
    return None

model = load_model()

def preprocess_row(row):
    """
    Replicates the logic from Section 2 of Kusuma's notebook:
    - Fills missing values with 'Unknown'
    - Processes salary range to an average
    - Encodes categorical features
    """
    # 1. Handling Missing Categoricals
    emp_type = str(row.get('employment_type', 'Unknown'))
    req_edu = str(row.get('required_education', 'Unknown'))
    
    if pd.isna(emp_type) or emp_type == 'nan': emp_type = 'Unknown'
    if pd.isna(req_edu) or req_edu == 'nan': req_edu = 'Unknown'

    # 2. Salary Processing (Logic from Cell 12)
    salary = row.get('salary_range', "")
    salary_processed = 44000.0  # Training Median (from Cell 13)
    if isinstance(salary, str) and '-' in salary:
        try:
            low, high = salary.split('-')
            salary_processed = (int(low) + int(high)) / 2
        except:
            pass

   # 3. Categorical Encoding
    feature_dict = {
        'salary_processed': salary_processed,
        'employment_type': emp_map.get(emp_type, 5),
        'has_company_logo': int(row.get('has_company_logo', 0)), # Keep it simple
        'required_education': edu_map.get(req_edu, 9)
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

