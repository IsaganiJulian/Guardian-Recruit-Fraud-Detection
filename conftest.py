import sys
import os
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

# Add src/ to sys.path so tests can import outlier_stream directly
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
    import outlier_stream

    # Minimal synthetic training data that mirrors the 4-feature schema
    synthetic_X = pd.DataFrame({
        'salary_processed':    [44000, 60000, 80000, 44000, 44000, 44000,
                                 44000, 44000, 44000, 44000],
        'employment_type':     [1, 1, 1, 5, 5, 0, 3, 1, 1, 1],
        'has_company_logo':    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        'required_education':  [1, 1, 2, 9, 9, 9, 4, 1, 1, 1],
    })

    fitted_model = IsolationForest(
        n_estimators=10, contamination=0.05, random_state=42
    )
    fitted_model.fit(synthetic_X)

    # Patch the module-level model used by anomaly_score / anomaly_predict
    outlier_stream.model = fitted_model
    yield
