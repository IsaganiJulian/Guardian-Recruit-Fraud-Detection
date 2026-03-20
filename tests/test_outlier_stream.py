"""
B-9: Unit tests for anomaly_score() in src/outlier_stream.py

Run from repo root:
    pytest tests/test_outlier_stream.py -v
"""

import math
import pandas as pd
import pytest
import outlier_stream


# ── Shared sample rows ────────────────────────────────────────────────────────

def normal_row():
    """A well-formed, low-suspicion job posting."""
    return pd.Series({
        'salary_range':        '60000-90000',
        'employment_type':     'Full-time',
        'has_company_logo':    1,
        'required_education':  "Bachelor's Degree",
        'telecommuting':       0,
    })


def fraudulent_row():
    """
    A high-suspicion row: no logo, no salary, no employment type,
    no required education — pattern associated with fraudulent postings
    in the training EDA.
    """
    return pd.Series({
        'salary_range':        '',
        'employment_type':     float('nan'),
        'has_company_logo':    0,
        'required_education':  float('nan'),
        'telecommuting':       0,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_loaded():
    """The pkl file must be present and loaded at import time."""
    assert outlier_stream.model is not None, (
        "outlier_stream.model is None — "
        "make sure models/outlier_forest.pkl exists."
    )


def test_returns_float():
    """anomaly_score() must always return a plain Python float."""
    score = outlier_stream.anomaly_score(normal_row())
    assert isinstance(score, float)


def test_normal_row_positive_score():
    """
    A clean, well-formed posting should be scored as normal (score > 0).
    IsolationForest decision_function: positive = inlier, negative = outlier.
    """
    score = outlier_stream.anomaly_score(normal_row())
    assert score > 0, f"Expected positive score for normal row, got {score:.4f}"


def test_missing_fields_does_not_raise():
    """
    Rows with entirely missing fields must not raise an exception.
    preprocess_row() fills NaN employment_type/education with 'Unknown'
    and falls back to the salary median.
    """
    empty_row = pd.Series({})
    score = outlier_stream.anomaly_score(empty_row)
    assert isinstance(score, float)
    assert not math.isnan(score)


def test_malformed_salary_uses_median():
    """
    An unparseable salary_range should silently fall back to the
    training median (44000.0) and still return a valid float score.
    """
    row = normal_row().copy()
    row['salary_range'] = 'not-a-number'
    score = outlier_stream.anomaly_score(row)
    assert isinstance(score, float)
    assert not math.isnan(score)


def test_fraud_scores_lower_than_normal():
    """
    A suspicious row should receive a lower anomaly score than a clean row.
    Lower decision_function score = more anomalous = more likely fraud.
    """
    normal_score = outlier_stream.anomaly_score(normal_row())
    fraud_score  = outlier_stream.anomaly_score(fraudulent_row())
    assert fraud_score < normal_score, (
        f"Expected fraud_score ({fraud_score:.4f}) < "
        f"normal_score ({normal_score:.4f})"
    )
