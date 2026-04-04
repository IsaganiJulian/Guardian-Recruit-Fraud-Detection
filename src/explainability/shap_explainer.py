"""
shap_explainer.py — SHAP wrapper for the Guardian Recruit fusion layer.

Produces a ranked list of feature contributions and a plain-English
"Reasoning Summary" for every scored posting.

Example output:
    "Flagged because: AI-generated text likelihood (+38%), domain age < 30 days (+27%),
     suspicious platform mentions (+18%)"

Design notes:
  - Works with the XGBoost Booster saved as fusion_xgb.json (same loader as
    fusion_layer.py — no second model object in memory).
  - SHAP values are extracted for the fraud class (index 1 in binary output).
  - Contributions are normalised to percentages of total absolute SHAP mass so
    each posting has a self-contained, comparable breakdown.
  - `reasoning_summary()` is pure Python — no LLM required. It is always fast.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Human-readable labels for each meta-feature ───────────────────────────────
# Must match fusion_layer.META_FEATURES exactly (order matters for SHAP indexing)
FEATURE_LABELS = {
    'bert_score':       'NLP fraud pattern score',
    'outlier_score':    'Metadata anomaly score',
    'has_company_logo': 'No company logo',
    'has_questions':    'No screening questions',
    'desc_len':         'Description length',
    'domain_age_days':  'Domain age',
    'text_perplexity':  'AI-generated text likelihood',
    'platform_risk':    'Suspicious platform mentions',
}

# Thresholds used to enrich the label with context (e.g. "domain age < 30 days")
_DOMAIN_AGE_WARN   = 30    # days — below this is high-risk
_PERPLEXITY_WARN   = 80.0  # below this = likely AI-generated


class FeatureContribution(NamedTuple):
    feature:     str    # raw feature name  (e.g. 'domain_age_days')
    label:       str    # human label       (e.g. 'Domain age < 30 days')
    shap_value:  float  # raw SHAP value (positive = toward fraud)
    pct:         float  # % of total |SHAP| mass


# ── SHAP model loader (lazy, cached) ─────────────────────────────────────────

_explainer = None


def _load_explainer():
    global _explainer
    if _explainer is not None:
        return _explainer

    try:
        import shap
        import xgboost as xgb
        import os

        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fusion_xgb.json')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            logger.warning('fusion_xgb.json not found — SHAP explanations unavailable.')
            return None

        booster = xgb.Booster()
        booster.load_model(model_path)
        _explainer = shap.TreeExplainer(booster)
        logger.info('SHAP TreeExplainer loaded.')
        return _explainer

    except Exception as e:
        logger.warning(f'Could not load SHAP explainer: {e}')
        return None


# ── Core computation ──────────────────────────────────────────────────────────

def compute_shap(features: dict) -> list[FeatureContribution] | None:
    """
    Compute SHAP contributions for a single prediction's feature dict.

    Args:
        features: Dict matching fusion_layer.META_FEATURES keys.
                  Comes directly from fusion_layer.predict()['features'].

    Returns:
        List of FeatureContribution sorted by |shap_value| descending,
        or None if the model is not loaded.
    """
    explainer = _load_explainer()
    if explainer is None:
        return None

    try:
        from fusion_layer import META_FEATURES

        X = pd.DataFrame([features], columns=META_FEATURES)

        raw = explainer.shap_values(X)

        # shap_values() returns:
        #   older SHAP: list of two arrays [legit_shap, fraud_shap]
        #   newer SHAP: single 2D array (binary → fraud class only)
        if isinstance(raw, list):
            shap_vals = raw[1][0]   # fraud class, first (only) row
        else:
            shap_vals = raw[0]      # first row

        total_mass = float(np.abs(shap_vals).sum()) or 1.0

        contributions = []
        for feat, val in zip(META_FEATURES, shap_vals):
            label = _enrich_label(feat, features.get(feat), float(val))
            contributions.append(FeatureContribution(
                feature    = feat,
                label      = label,
                shap_value = float(val),
                pct        = round(abs(float(val)) / total_mass * 100, 1),
            ))

        return sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)

    except Exception as e:
        logger.warning(f'SHAP computation failed: {e}')
        return None


def _enrich_label(feature: str, value, shap_value: float) -> str:
    """Add context to the label based on the actual feature value."""
    base = FEATURE_LABELS.get(feature, feature)

    if feature == 'domain_age_days':
        if value is not None and value != -1 and float(value) < _DOMAIN_AGE_WARN:
            return f'Domain age < {_DOMAIN_AGE_WARN} days'
        if value == -1:
            return 'Domain age unknown'
        return base

    if feature == 'text_perplexity':
        if value is not None and float(value) < _PERPLEXITY_WARN:
            return 'AI-generated text likelihood (high)'
        return 'AI-generated text likelihood (low)'

    if feature == 'platform_risk' and value is not None:
        count = int(value)
        if count == 0:
            return 'No suspicious platform mentions'
        return f'Suspicious platform mentions ({count})'

    if feature == 'has_company_logo':
        return 'No company logo' if not value else 'Company logo present'

    if feature == 'has_questions':
        return 'No screening questions' if not value else 'Screening questions present'

    return base


# ── Reasoning Summary ─────────────────────────────────────────────────────────

def reasoning_summary(
    contributions: list[FeatureContribution],
    label: str,
    top_n: int = 3,
    min_pct: float = 5.0,
) -> str:
    """
    Format SHAP contributions into a one-line reasoning summary.

    Args:
        contributions: Output of compute_shap().
        label:         "FRAUD" or "LEGITIMATE".
        top_n:         Maximum number of drivers to include (default: 3).
        min_pct:       Minimum % contribution to include a feature (default: 5%).

    Returns:
        str — e.g.
          "Flagged because: AI-generated text likelihood (high) (+38%),
           Domain age < 30 days (+27%), Suspicious platform mentions (2) (+18%)"

          "Appears legitimate: NLP fraud pattern score (-41%),
           Company logo present (-22%), Screening questions present (-18%)"
    """
    if not contributions:
        return 'Insufficient data for a detailed explanation.'

    # For FRAUD: drivers pushing toward fraud (positive SHAP)
    # For LEGITIMATE: drivers pushing away from fraud (negative SHAP)
    if label == 'FRAUD':
        drivers = [c for c in contributions if c.shap_value > 0 and c.pct >= min_pct]
        prefix  = 'Flagged because'
        sign    = '+'
    else:
        drivers = [c for c in contributions if c.shap_value < 0 and c.pct >= min_pct]
        prefix  = 'Appears legitimate'
        sign    = '-'

    # Fall back to top contributors by magnitude if polarity filtering left nothing
    if not drivers:
        drivers = contributions

    top = drivers[:top_n]
    if not top:
        return 'No dominant fraud signals identified.'

    parts = ', '.join(f'{c.label} ({sign}{c.pct:.0f}%)' for c in top)
    return f'{prefix}: {parts}'


# ── Public interface ──────────────────────────────────────────────────────────

def explain_shap(result: dict) -> dict:
    """
    Generate SHAP-based explanation for a fusion_layer.predict() result.

    Args:
        result: Dict returned by fusion_layer.predict() or main.score().

    Returns:
        {
            "reasoning_summary":  str   — one-line human-readable explanation,
            "contributions":      list  — FeatureContribution namedtuples (ranked),
            "shap_available":     bool  — False if model not loaded,
        }
    """
    contributions = compute_shap(result['features'])

    if contributions is None:
        return {
            'reasoning_summary': 'SHAP model not loaded — run fusion training first.',
            'contributions':     [],
            'shap_available':    False,
        }

    summary = reasoning_summary(contributions, label=result['label'])

    return {
        'reasoning_summary': summary,
        'contributions':     contributions,
        'shap_available':    True,
    }
