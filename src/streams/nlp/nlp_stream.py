"""
NLP Stream Module - BERT Text Analysis
Handles natural language processing for job posting fraud detection using BERT.

Phase 2 Tasks:
  A-2: BERT tokenizer setup & DataLoader construction
  A-3: Fine-tune BERT on train.csv using Colab GPU
  A-4: Evaluate on val.csv (target F1 >= 0.85 for fraud class)
  A-5: Hyperparameter tuning
  A-6: Export nlp_bert.pth to Google Drive /models/
  A-7: Port predict_proba(text) -> float here (THIS FILE)  ← implemented
  A-8: Unit test predict_proba() with sample postings
"""

import os
import re
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as transformers_logging

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'nlp_bert.pth')

# ── Module-level state (loaded once on first call) ───────────────────────────
_tokenizer = None
_model     = None
_device    = None


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


_bert_available = None   # None = unchecked, True/False after first load attempt


def bert_model_available() -> bool:
    """Return True if the fine-tuned BERT weights file exists and is non-empty."""
    global _bert_available
    if _bert_available is None:
        path = os.path.abspath(MODEL_PATH)
        _bert_available = os.path.exists(path) and os.path.getsize(path) > 1_000_000
    return _bert_available


def _load_model():
    """Load tokenizer and model from models/nlp_bert.pth (once per process).
    Skipped silently if the weights file is absent — lite deployment mode."""
    global _tokenizer, _model, _device

    if _model is not None:
        return  # already loaded

    if not bert_model_available():
        return  # lite mode — caller checks _model is None and uses fallback

    _device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    model_path = os.path.abspath(MODEL_PATH)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        state = torch.load(model_path, map_location=_device, weights_only=True)
        _model.load_state_dict(state)
    # If no weights file yet, the model returns untrained probabilities (~0.5).
    # This is intentional — predict_proba() will still return a float safely.

    _model.to(_device)
    _model.eval()


def predict_proba(text: str) -> float:
    """
    Return the fraud probability (0.0–1.0) for a single job posting text.

    Args:
        text: Raw combined job text (title + company_profile + description + requirements).
              Cleaning is applied internally.

    Returns:
        Float in [0, 1] — probability the posting is fraudulent.
        Higher = more likely fraud.
    """
    _load_model()

    if _model is None:
        # Lite mode — BERT weights not present.
        # Return -1.0 as sentinel so callers can substitute keyword score.
        return -1.0

    cleaned = _clean_text(text)
    enc = _tokenizer(
        cleaned,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )

    input_ids      = enc['input_ids'].to(_device)
    attention_mask = enc['attention_mask'].to(_device)

    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
        prob   = F.softmax(logits, dim=1)[0, 1]  # fraud class probability

    return float(prob)


def predict_proba_from_row(row) -> float:
    """
    Convenience wrapper: accepts a pandas Series or dict with job posting fields.
    Combines text fields the same way the training notebook does.

    Args:
        row: dict or pd.Series with keys: title, company_profile, description, requirements

    Returns:
        Float fraud probability.
    """
    def _get(key):
        try:
            val = row[key] if hasattr(row, '__getitem__') else getattr(row, key, '')
            return '' if val is None or (isinstance(val, float) and val != val) else str(val)
        except (KeyError, AttributeError):
            return ''

    combined = ' '.join([
        _get('title'),
        _get('company_profile'),
        _get('description'),
        _get('requirements'),
    ])
    return predict_proba(combined)
