import sys
import os
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

# Add src/ to sys.path so tests can import streams.outlier.outlier_stream
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


@pytest.fixture(autouse=True, scope="session")
def patch_outlier_model():
    """
    The local models/outlier_forest.pkl is a placeholder (the real file lives on
    Google Drive). For tests to run locally we fit a minimal IsolationForest on
    a small synthetic feature matrix and patch it into outlier_stream.model.

    scope="session"  — fitted once per test run, not per test.
    autouse=True     — applied to every test automatically.
    """
    from streams.outlier import outlier_stream

    # Synthetic training data: 8 normal + 2 fraud — clearly separated
    # Normal: full-time, logo, questions, decent salary, long description
    # Fraud:  unknown type, no logo, no questions, no salary, very short description
    synthetic_X = pd.DataFrame({
        'salary_processed':    [65000, 72000, 58000, 80000, 55000, 90000, 48000, 70000,
                                 44000, 44000],
        'employment_type':     [1, 1, 1, 1, 1, 1, 1, 1,
                                 5, 5],
        'has_company_logo':    [1, 1, 1, 1, 1, 1, 1, 1,
                                 0, 0],
        'has_questions':       [1, 1, 0, 1, 1, 1, 0, 1,
                                 0, 0],
        'telecommuting':       [0, 0, 1, 0, 0, 0, 1, 0,
                                 0, 0],
        'required_education':  [1, 1, 2, 1, 5, 1, 3, 1,
                                 9, 9],
        'desc_len':            [1500, 2000, 1200, 1800, 900, 2200, 1100, 1600,
                                 80, 60],
    })

    fitted_model = IsolationForest(
        n_estimators=10, contamination=0.05, random_state=42
    )
    fitted_model.fit(synthetic_X)

    # Patch the module-level model used by anomaly_score / anomaly_predict
    outlier_stream.model = fitted_model
    yield
