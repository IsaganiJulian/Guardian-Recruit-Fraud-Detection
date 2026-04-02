"""
Text Signals — Keyword-based fraud signal detector.

Signals are split into two tiers:

  TIER 1 — Training-data validated (train_clean_v1.csv, 2012–2014 era)
  -----------------------------------------------------------------------
  These patterns have statistically significant lift confirmed by TF-IDF
  analysis against 422 real fraud postings. See scripts/validate_signals.py
  for full evidence.

  TIER 2 — Domain knowledge (2020–2026 era fraud patterns)
  -----------------------------------------------------------------------
  These patterns do NOT appear in the 2014 training data because they
  postdate it. They are grounded in documented fraud typologies:
    - FBI IC3 reports on job scams (2021–2024)
    - FTC Consumer Sentinel data on employment fraud
    - ACFE Occupational Fraud reports
  They cannot be statistically validated without 2026-era labelled data.

Final score = sum of triggered signal weights, capped at 1.0.
"""

import re

# ── Tier 1: Training-data validated signals ───────────────────────────────────
# lift = how much more likely a posting with this pattern is to be fraud
# vs the 4.85% base rate in train_clean_v1.csv
# coverage = % of all 422 fraud postings in training set that contain this pattern

TIER1_SIGNALS = [
    # lift=6.1x, coverage=4.0% — "no experience required" + high salary data entry roles
    ('no_experience_high_pay', 0.40, [
        r'no\s+(prior\s+)?(work\s+)?experience\s+(is\s+)?(required|needed|necessary)',
        r'no\s+experience\s+required',
        r'no\s+background\s+check',
        r'no\s+(degree|qualification)\s+(is\s+)?required',
    ]),

    # lift=5.1x, coverage=9.7% — "work from any country" / "open to applicants worldwide"
    ('wfh_unrealistic', 0.35, [
        r'work\s+from\s+(home|anywhere|any\s+country)',
        r'open\s+to\s+applicants\s+worldwide',
        r'work\s+from\s+any\s+country',
        r'position\s+(is\s+)?open\s+(to\s+)?worldwide',
    ]),

    # lift=110x, coverage=7.6% — "signing bonus" appears at extreme rate in fraud postings
    ('signing_bonus', 0.40, [
        r'signing\s+bonus',
        r'sign.on\s+bonus',
    ]),

    # lift=79x, coverage=9.5% — "computer and internet access" template phrase
    # Almost exclusively appears in work-from-home scam boilerplate
    ('computer_internet_template', 0.35, [
        r'(personal\s+)?computer\s+and\s+(reliable\s+)?internet\s+access',
        r'access\s+to\s+a\s+(personal\s+)?computer\s+and',
        r'must\s+have\s+(a\s+)?(computer|laptop)\s+and\s+internet',
    ]),

    # lift=60x, coverage=6.2% — "online training" in context of immediate hire
    ('instant_online_training', 0.25, [
        r'online\s+training\s+(will\s+be\s+)?(provided|given)',
        r'full\s+training\s+(provided|within\s+\d+\s+(hours?|days?))',
        r'training\s+provided\s+within\s+\d+\s+(hours?|days?)',
    ]),

    # lift=89x, coverage=6.9% — fake relocation packages used to extract fees
    ('fake_relocation', 0.30, [
        r'relocation\s+(package|services?|assistance)\s+(provided|included|offered)',
        r'we\s+(will\s+)?cover\s+(your\s+)?relocation',
        r'relocation\s+allowance',
    ]),

    # Oil & gas sector fraud cluster — entire template reused across postings
    # lift=727x for "subsea", lift=104x for "oil gas industry"
    # These postings impersonate offshore engineering companies
    ('oil_gas_impersonation', 0.45, [
        r'\bsubsea\b',
        r'oil\s+(and\s+|&\s+)?gas\s+(industry|sector|company|firm)',
        r'\bhse\b.*\b(compliance|officer|manager)',  # Health Safety Environment
        r'maximize\s+recovery',
        r'approximately\s+28[,\s]?000\s+(employees?|staff|workers?)',
        r'employ\s+approximately\s+\d{2,3}[,\s]?\d{3}',
    ]),

    # S3 bucket links — fraudsters host fake offer letters / documents on AWS S3
    # lift=1453x — near-exclusive to fraud in training data
    ('s3_document_link', 0.50, [
        r'https?://s3[\.-]',
        r's3\.amazonaws\.com',
        r'https?://[a-z0-9\-]+\.s3\.',
    ]),

    # PII request — lift=2.9x in training data (weak but present)
    ('pii_request', 0.20, [
        r'government.issued\s+id',
        r'copy\s+of\s+(your\s+)?(passport|id\s+card|driver.s\s+licen[sc]e)',
        r'date\s+of\s+birth',
        r'social\s+security\s+number',
        r'send\s+(us\s+)?your\s+(full\s+)?home\s+address',
    ]),
]

# ── Tier 2: Domain knowledge signals (post-2014, unvalidated on training data) ─
# Source: FBI IC3 Internet Crime Reports 2021–2024, FTC job scam advisories
# These cannot be statistically validated against train_clean_v1.csv because
# the schemes they describe did not exist when the dataset was collected.

TIER2_SIGNALS = [
    # FBI IC3 2023: cryptocurrency payment scams increasingly used in job fraud
    ('crypto_payment', 0.35, [
        r'\bcryptocurrency\b',
        r'\bbitcoin\b',
        r'\busdt\b',
        r'paid?\s+(weekly|daily|bi.weekly)\s+via\s+(wire|crypto)',
        r'salary\s+paid\s+in\s+crypto',
    ]),

    # FTC 2022–2024: WhatsApp/Telegram used to avoid traceable corporate communication
    ('messaging_app_interview', 0.35, [
        r'\bwhatsapp\b',
        r'\btelegram\b',
        r'interview(s)?\s+(conducted\s+)?(exclusively\s+)?(via|on|through)\s+(whatsapp|telegram|signal)',
        r'contact\s+(us\s+)?(via|on|through)\s+(whatsapp|telegram)',
    ]),

    # FBI IC3 2021–2023: reshipping / money mule schemes via fake job postings
    ('money_mule', 0.45, [
        r'forward(ing)?\s+(company\s+)?packages?',
        r'reshipping',
        r'forwarding\s+payments?\s+on\s+(our|the\s+company.s?)\s+behalf',
        r'transfer\s+funds?\s+on\s+(our|the\s+company.s?)\s+behalf',
        r'receive\s+and\s+(re.?)?forward',
    ]),

    # FTC 2023: equipment bait — MacBook/laptop sent on signing used to establish trust
    ('equipment_bait', 0.20, [
        r'(macbook|laptop|computer)\s+(pro\s+)?sent\s+upon\s+(signing|hire|joining|start)',
        r'home\s+office\s+(setup\s+)?stipend',
        r'equipment\s+(will\s+be\s+)?(provided|shipped|sent)',
    ]),

    # Artificial urgency — social engineering to prevent due diligence
    ('artificial_urgency', 0.20, [
        r'apply\s+within\s+\d+\s+hours?',
        r'positions?\s+(are\s+)?filling\s+fast',
        r'must\s+apply\s+(immediately|now|today|within)',
        r'offer\s+letter(s)?\s+sent\s+within\s+\d+\s+hours?',
        r'urgently\s+hiring',
    ]),

    # Suspicious sender domain — real companies use primary domain for careers email
    ('suspicious_email_domain', 0.25, [
        r'@[a-z0-9\-]+\.(intl|corp|group|global|solutions?|services?|support)\-[a-z]+\.',
        r'careers?@[a-z0-9\-]{20,}\.',
        r'@gmail\.com\b',
        r'@yahoo\.com\b',
        r'@outlook\.com\b',
    ]),
]

# Combined — used by compute_keyword_score()
SIGNALS = TIER1_SIGNALS + TIER2_SIGNALS


def compute_keyword_score(text: str) -> tuple[float, list[dict]]:
    """
    Compute a fraud signal score from raw posting text.

    Args:
        text: Full combined posting text.

    Returns:
        (score, triggered_signals)
        score: float 0.0–1.0
        triggered_signals: list of dicts — name, weight, tier, matched_pattern
    """
    if not text or not text.strip():
        return 0.0, []

    text_lower = text.lower()
    triggered  = []
    total      = 0.0

    tier1_names = {s[0] for s in TIER1_SIGNALS}

    for name, weight, patterns in SIGNALS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                triggered.append({
                    'signal':  name,
                    'weight':  weight,
                    'tier':    1 if name in tier1_names else 2,
                    'pattern': pattern,
                })
                total += weight
                break  # count each signal category once

    return round(min(total, 1.0), 4), triggered


def is_high_confidence_fraud(text: str, threshold: float = 0.5) -> bool:
    """Return True if keyword signals indicate high-confidence fraud."""
    score, _ = compute_keyword_score(text)
    return score >= threshold
