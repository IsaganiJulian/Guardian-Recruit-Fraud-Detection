"""
Main Module - Guardian Recruit Fraud Detection Pipeline
End-to-end scoring pipeline integrating all three streams.

Phase 2 Task:
  L-8: Integrate all streams in this file — end-to-end scoring pipeline
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from streams.nlp.preprocessing import GuardianCleaner
from streams.nlp import nlp_stream
from streams.outlier import outlier_stream
from streams.fusion import fusion_layer

_cleaner = GuardianCleaner()


def score(job_posting: dict) -> dict:
    """
    Score a single raw job posting end-to-end.

    Runs the full pipeline:
      1. Clean & normalise raw fields (GuardianCleaner)
      2. Stream A — BERT NLP score
      3. Stream B — IsolationForest anomaly score
      4. Fusion Layer — XGBoost final decision

    Args:
        job_posting: Raw dict with job posting fields. Any subset of:
            title, location, department, salary_range, company_profile,
            description, requirements, benefits, telecommuting,
            has_company_logo, has_questions, employment_type,
            required_experience, required_education, industry, function

    Returns:
        {
            "fraud_score":    float  — final probability (0.0–1.0)
            "label":          str    — "FRAUD" or "LEGITIMATE"
            "threshold":      float  — decision boundary used
            "bert_score":     float  — Stream A raw score
            "outlier_score":  float  — Stream B raw score (lower = more anomalous)
            "features":       dict   — all meta-features used by fusion layer
        }

    Example:
        result = score({
            "title": "Work From Home — Unlimited Earnings",
            "description": "No experience needed. Send $50 deposit.",
            "has_company_logo": 0,
            "has_questions": 0,
        })
        print(result["label"])       # "FRAUD"
        print(result["fraud_score"]) # e.g. 0.923
    """
    df = pd.DataFrame([job_posting])
    cleaned = _cleaner.transform(df, is_training=False)
    row = cleaned.iloc[0]

    return fusion_layer.predict(row)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of job postings and return results as new columns.

    Args:
        df: DataFrame with raw job posting columns.

    Returns:
        Original DataFrame with four new columns appended:
            fraud_score, label, bert_score, outlier_score
    """
    results = [score(row.to_dict()) for _, row in df.iterrows()]

    out = df.copy()
    out['fraud_score']   = [r['fraud_score']   for r in results]
    out['label']         = [r['label']          for r in results]
    out['bert_score']    = [r['bert_score']     for r in results]
    out['outlier_score'] = [r['outlier_score']  for r in results]
    return out


if __name__ == '__main__':
    # Quick smoke test — score two example postings
    test_cases = [
        {
            'name': 'Suspicious posting',
            'posting': {
                'title':           'Work From Home — Unlimited Earnings',
                'description':     'No experience needed. Send deposit to get started.',
                'company_profile': '',
                'requirements':    '',
                'has_company_logo': 0,
                'has_questions':    0,
                'employment_type':  'Unknown',
                'salary_range':     '80000-120000',
            },
        },
        {
            'name': 'Legitimate posting',
            'posting': {
                'title':           'Senior Software Engineer',
                'description':     'We are looking for an experienced software engineer to join our team. ' * 20,
                'company_profile': 'Established technology company with 500+ employees.',
                'requirements':    "Bachelor's degree in Computer Science. 5+ years Python experience.",
                'has_company_logo': 1,
                'has_questions':    1,
                'employment_type':  'Full-time',
                'salary_range':     '120000-160000',
            },
        },
    ]

    for case in test_cases:
        result = score(case['posting'])
        print(f"\n{case['name']}")
        print(f"  Label        : {result['label']}")
        print(f"  Fraud score  : {result['fraud_score']:.4f}")
        print(f"  BERT score   : {result['bert_score']:.4f}")
        print(f"  Outlier score: {result['outlier_score']:.4f}")
