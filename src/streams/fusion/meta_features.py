"""
meta_features.py — Three adversarial-era feature extractors for the fusion layer.

  domain_age(row)        → int   days since company domain was registered (-1 = unknown)
  text_perplexity(text)  → float GPT-2 perplexity (lower = more AI-like)
  platform_risk(text)    → int   count of high-risk platform mentions (0–3)

All three are designed to counter "AI-perfect" postings that pass NLP checks:
  - Very new domains (<30 days) signal throwaway scam infrastructure
  - Very low perplexity signals AI-generated, unnaturally fluent prose
  - Platform risk catches Telegram/WhatsApp/Signal contact evasion

Models are loaded lazily and cached at module level so the cost is paid once
per process, not once per row.
"""

import re
import math
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── URL extraction ─────────────────────────────────────────────────────────────
_URL_RE = re.compile(
    r'https?://(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})',
    re.IGNORECASE,
)
_BARE_DOMAIN_RE = re.compile(
    r'\b(?:www\.)?([a-zA-Z0-9\-]{3,}\.[a-zA-Z]{2,6})\b',
    re.IGNORECASE,
)

# Free email providers — skip WHOIS for these (never a company domain)
_FREE_EMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'protonmail.com', 'icloud.com', 'aol.com',
}


def _extract_domain(row) -> str | None:
    """Pull the first plausible company domain from a job posting row."""
    # Check structured fields first
    for field in ('company_profile', 'description', 'benefits'):
        text = str(row.get(field, '') or '')
        m = _URL_RE.search(text)
        if m:
            domain = m.group(1).lower()
            if domain not in _FREE_EMAIL_DOMAINS:
                return domain

    # Fallback: bare domain in any text field
    for field in ('company_profile', 'description'):
        text = str(row.get(field, '') or '')
        for m in _BARE_DOMAIN_RE.finditer(text):
            domain = m.group(1).lower()
            tld = domain.rsplit('.', 1)[-1]
            # Skip single-word common English words mismatched as domains
            if len(domain.split('.')[0]) >= 3 and tld in {
                'com', 'org', 'net', 'io', 'co', 'biz', 'info', 'ai',
            }:
                if domain not in _FREE_EMAIL_DOMAINS:
                    return domain

    return None


# ── Feature 1: Domain Age ──────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _whois_age_days(domain: str) -> int:
    """
    Returns days since domain registration via WHOIS.
    Returns -1 on any lookup failure (timeout, private registration, etc.).
    Cached so repeated rows with the same domain pay the cost once.
    """
    try:
        import whois  # python-whois package
        from datetime import datetime, timezone

        data = whois.whois(domain)
        creation = data.creation_date

        # python-whois sometimes returns a list
        if isinstance(creation, list):
            creation = creation[0]

        if creation is None:
            return -1

        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)

        now  = datetime.now(timezone.utc)
        days = (now - creation).days
        return max(days, 0)

    except Exception:
        return -1


def domain_age(row) -> int:
    """
    Extract company domain from row and return its age in days.

    Returns:
        int ≥ 0  — days since registration
        -1       — domain not found in text, or WHOIS lookup failed
    """
    domain = _extract_domain(row)
    if domain is None:
        return -1
    return _whois_age_days(domain)


# ── Feature 2: Linguistic Perplexity ──────────────────────────────────────────

_perplexity_model    = None
_perplexity_tokenizer = None
_PERPLEXITY_MAX_TOKENS = 256   # truncate for speed; fraud signals are dense in the first ~200 tokens
_PERPLEXITY_FALLBACK   = 200.0  # returned when model not loaded


def _load_perplexity_model():
    """Lazy-load GPT-2 for perplexity scoring. Called once per process."""
    global _perplexity_model, _perplexity_tokenizer
    if _perplexity_model is not None:
        return True
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch

        _perplexity_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        _perplexity_tokenizer.pad_token = _perplexity_tokenizer.eos_token
        _perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
        _perplexity_model.eval()
        logger.info('GPT-2 perplexity model loaded.')
        return True
    except Exception as e:
        logger.warning(f'Could not load GPT-2 for perplexity: {e}')
        return False


def text_perplexity(text: str) -> float:
    """
    Compute GPT-2 perplexity of the input text.

    Lower perplexity = more predictable token sequence = more likely AI-generated.
    Typical ranges:
      Human job posting:        150–400
      AI-generated (2026 scam): 30–100

    Args:
        text: Raw job posting text (description + requirements recommended).

    Returns:
        float perplexity value, or _PERPLEXITY_FALLBACK (200.0) on failure.
    """
    if not text or not text.strip():
        return _PERPLEXITY_FALLBACK

    if not _load_perplexity_model():
        return _PERPLEXITY_FALLBACK

    try:
        import torch

        enc = _perplexity_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=_PERPLEXITY_MAX_TOKENS,
        )
        input_ids = enc['input_ids']
        if input_ids.shape[1] < 2:
            return _PERPLEXITY_FALLBACK

        with torch.no_grad():
            outputs  = _perplexity_model(input_ids, labels=input_ids)
            loss     = outputs.loss.item()

        perplexity = math.exp(loss)
        return round(perplexity, 2)

    except Exception as e:
        logger.warning(f'Perplexity computation failed: {e}')
        return _PERPLEXITY_FALLBACK


# ── Feature 3: Platform Risk ───────────────────────────────────────────────────

_PLATFORM_PATTERNS = [
    re.compile(r'\bwhatsapp\b', re.IGNORECASE),
    re.compile(r'\btelegram\b', re.IGNORECASE),
    re.compile(r'\bsignal\s*(app|messenger)?\b', re.IGNORECASE),
]


def platform_risk(text: str) -> int:
    """
    Count distinct high-risk communication platforms mentioned in the text.

    Each platform (WhatsApp, Telegram, Signal) counts once regardless of how
    many times it appears, so the score is bounded 0–3.

    Returns:
        int 0–3
    """
    if not text or not text.strip():
        return 0
    return sum(1 for pat in _PLATFORM_PATTERNS if pat.search(text))


# ── Convenience: extract all three from a single row ──────────────────────────

def extract_all(row) -> dict:
    """
    Run all three extractors on a job posting row and return a flat dict.

    Args:
        row: dict or pd.Series with job posting fields.

    Returns:
        {
            'domain_age_days':   int,
            'text_perplexity':   float,
            'platform_risk':     int,
        }
    """
    combined_text = ' '.join(
        str(row.get(f, '') or '')
        for f in ('title', 'company_profile', 'description', 'requirements', 'benefits')
    )
    return {
        'domain_age_days': domain_age(row),
        'text_perplexity': text_perplexity(combined_text),
        'platform_risk':   platform_risk(combined_text),
    }
