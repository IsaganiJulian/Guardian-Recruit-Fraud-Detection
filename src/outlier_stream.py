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

# TODO (B-8): Implement anomaly_score(row: pd.Series) -> float
# Load model from models/outlier_forest.pkl and return anomaly score
