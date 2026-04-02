"""
scripts/generate_synthetic_fraud.py

Generates 2026-era synthetic fraud job postings using template injection.
No API key, no internet, no rate limits. Runs in seconds.

Takes real 2014 fraud rows as seeds and injects 2-3 modern fraud signals
directly into the text. Every output row is guaranteed to contain detectable
fraud patterns.

Usage:
    python scripts/generate_synthetic_fraud.py
    python scripts/generate_synthetic_fraud.py --n 200

Output: data/processed/synthetic_fraud_2026.csv
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parents[1]
SEED_PATH   = REPO_ROOT / "data" / "processed" / "train_clean_v1.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "synthetic_fraud_2026.csv"

SCHEMA_COLS = [
    "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "telecommuting",
    "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function",
    "fraudulent", "in_balanced_dataset", "country", "desc_len",
]

# ── Signal injection templates ─────────────────────────────────────────────────
# Each signal is a (signal_id, text_snippet) pair.
# Snippets are written to blend naturally into job posting prose.

SIGNAL_POOL = {
    "compensation_guarantee": [
        "Earn up to $750 per week working remotely — weekly pay guaranteed.",
        "Guaranteed income of $600/week from day one, paid every Friday.",
        "Earn up to $4,500 per month — no experience required to start earning.",
        "Weekly pay guaranteed. Top performers earn $800+ per week from home.",
    ],
    "hiring_urgency_2026": [
        "We are hiring immediately — limited slots available for this cohort.",
        "Rolling offers issued within 48 hours. Cohort closes this Friday.",
        "Start this week. Only 3 positions remaining — apply now.",
        "Urgently hiring. Offer letters sent within 24 hours of application.",
    ],
    "messaging_app_interview": [
        "Initial screening is conducted via WhatsApp. Contact our coordinator to schedule.",
        "All interviews are conducted exclusively via Telegram. Reach us @recruitcoord.",
        "Our onboarding team is available on WhatsApp for a quick intro call.",
        "First-round interviews via Telegram only. Corporate email follow-up after offer.",
    ],
    "signal_app_contact": [
        "Reach our hiring coordinator on Signal for a confidential discussion.",
        "Contact us via Signal app for fastest response from our recruitment team.",
    ],
    "crypto_payment": [
        "Compensation is issued weekly in USDT via Coinbase Pay.",
        "Salary disbursed bi-weekly in ETH — crypto wallet required.",
        "Flexible payment options including USDT, ETH, or direct deposit.",
    ],
    "equipment_bait": [
        "A home-office stipend of $350 is disbursed via Zelle after onboarding is complete.",
        "MacBook Pro shipped upon signing. A refundable equipment deposit of $200 applies.",
        "Full home-office setup provided. Equipment deposit ($150) refunded after 90 days.",
    ],
    "upfront_fee_request": [
        "A refundable training deposit of $99 is required to secure your position.",
        "Purchase your starter kit ($75) to begin — fully reimbursed in your first paycheck.",
        "A small registration fee ($49) is required to access the onboarding portal.",
    ],
    "task_scam": [
        "Your primary task involves rating and optimising app listings on major platforms.",
        "Role involves liking and reviewing content on Instagram and TikTok to boost engagement metrics.",
        "Complete simple product-rating tasks online and earn per completed batch.",
    ],
    "vague_company_descriptor": [
        "We are a fast-growing global digital solutions firm with clients across 40+ countries.",
        "Our international consulting group partners with Fortune 500 brands on remote workflow projects.",
        "A leading distributed-first organisation operating across the digital commerce space.",
    ],
    "artificial_urgency": [
        "Apply within 24 hours — this posting will be removed once the role is filled.",
        "Positions filling fast. We cannot guarantee availability after this week.",
        "Immediate openings only. Do not apply if you cannot start within 7 days.",
    ],
    "pii_request": [
        "To complete your application, please send a copy of your government-issued ID.",
        "Provide your bank details for direct deposit setup during onboarding.",
    ],
    "lookalike_domain": [
        "Questions? Email our HR team at careers@nexus-global-hiring.net.",
        "Send your resume to onboarding@talentsync-recruit.co for immediate review.",
        "Contact: apply@brightwave-careers.net — response within 2 business hours.",
    ],
}

SIGNAL_IDS = list(SIGNAL_POOL.keys())


def _clean_html(text: str) -> str:
    """Strip basic HTML tags from seed text."""
    import re
    return re.sub(r'<[^>]+>', ' ', str(text)).strip()


def inject_signals(seed_row: pd.Series, rng: random.Random) -> tuple[dict, list[str]]:
    """
    Take one real seed row and inject 2-3 modern fraud signals into its text.
    Returns (row_dict, signal_ids_used).
    """
    # Pick 2-3 signals at random
    n_signals = rng.randint(2, 3)
    chosen_ids = rng.sample(SIGNAL_IDS, n_signals)
    snippets   = [rng.choice(SIGNAL_POOL[sid]) for sid in chosen_ids]

    # Base text fields from seed
    title       = _clean_html(seed_row.get("title", "Remote Opportunity"))
    description = _clean_html(seed_row.get("description", ""))
    requirements = _clean_html(seed_row.get("requirements", ""))
    benefits    = _clean_html(seed_row.get("benefits", ""))

    # Modernise title
    modern_titles = [
        "Remote Operations Coordinator", "Digital Workflow Specialist",
        "AI-Integrated Data Review Associate", "App Optimisation Assistant",
        "Remote Client Engagement Specialist", "Work-From-Home Project Coordinator",
        "Distributed Team Operations Associate", "Remote Content Review Specialist",
    ]
    if not title or len(title) < 5:
        title = rng.choice(modern_titles)

    # Inject snippets: spread across description, requirements, benefits
    desc_inject  = snippets[0]
    req_inject   = snippets[1] if len(snippets) > 1 else ""
    ben_inject   = snippets[2] if len(snippets) > 2 else ""

    new_description  = f"{description}\n\n{desc_inject}".strip() if description else desc_inject
    new_requirements = f"{requirements}\n\n{req_inject}".strip() if req_inject else requirements
    new_benefits     = f"{benefits}\n\n{ben_inject}".strip() if ben_inject else benefits

    # Modern company profiles
    profiles = [
        "We are a globally distributed team leveraging AI-powered workflows to deliver outcomes for clients across 30+ countries. Our team operates fully remote-first.",
        "A fast-growing digital solutions group specialising in content operations, remote workforce management, and platform optimisation.",
        "International remote-first organisation connecting talent with flexible, high-earning digital work across time zones.",
    ]
    company_profile = rng.choice(profiles)

    # Salary — 2026 style
    salary_options = [
        "$600-$800", "$750-$900", "$3000-$4500", "$500-$700", "$2800-$4000",
    ]
    salary_range = rng.choice(salary_options)

    row = {
        "title":              title,
        "location":           seed_row.get("location", "Remote / Work From Home"),
        "department":         seed_row.get("department", "Operations"),
        "salary_range":       salary_range,
        "company_profile":    company_profile,
        "description":        new_description,
        "requirements":       new_requirements,
        "benefits":           new_benefits,
        "telecommuting":      1,
        "has_company_logo":   0,
        "has_questions":      0,
        "employment_type":    seed_row.get("employment_type", "Full-time"),
        "required_experience": seed_row.get("required_experience", "Not Applicable"),
        "required_education":  seed_row.get("required_education", "Unspecified"),
        "industry":           seed_row.get("industry", "Internet"),
        "function":           seed_row.get("function", "Administrative"),
        "fraudulent":         1,
        "in_balanced_dataset": 0,
        "country":            seed_row.get("country", "US"),
        "desc_len":           len(new_description),
        "_seed_index":        int(seed_row.name),
        "_signals_injected":  "|".join(chosen_ids),
    }
    return row, chosen_ids


def main(n_samples: int, seed: int) -> None:
    if OUTPUT_PATH.exists():
        print(f"[SKIP] {OUTPUT_PATH.relative_to(REPO_ROOT)} already exists.")
        print("       Delete it to regenerate.")
        sys.exit(0)

    df    = pd.read_csv(SEED_PATH)
    fraud = df[df["fraudulent"] == 1].reset_index(drop=True)
    print(f"Loaded {len(fraud)} seed fraud examples.")

    seeds = fraud.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)
    rng   = random.Random(seed)

    print(f"Generating {n_samples} synthetic postings...\n")

    results = []
    for i, (_, seed_row) in enumerate(seeds.iterrows()):
        row, sigs = inject_signals(seed_row, rng)
        results.append(row)
        label = f"[{i+1:>{len(str(n_samples))}}/{n_samples}]"
        print(f"{label} OK  [{', '.join(sigs)}]")

    out_df = pd.DataFrame(results)
    review_cols = [c for c in out_df.columns if c.startswith("_")]

    for col in SCHEMA_COLS:
        if col not in out_df.columns:
            out_df[col] = ""

    out_df = out_df[SCHEMA_COLS + review_cols]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 55}")
    print(f"Generated : {len(results):>5,} synthetic fraud postings")
    print(f"Saved to  : {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"\nReview columns:")
    print(f"  _seed_index       — original 2014 row used as seed")
    print(f"  _signals_injected — fraud signals injected into this row")
    print(f"\nNext step: concatenate with FINAL_AUGMENTED_TRAINING.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=200, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed for reproducibility")
    args = parser.parse_args()
    main(n_samples=args.n, seed=args.seed)
