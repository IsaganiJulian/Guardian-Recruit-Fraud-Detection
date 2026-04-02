"""
End-to-end smoke test for the Guardian Recruit fraud detection pipeline.
Runs src/main.py:score() against a sample of scraped 2026 job postings.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --n 50
    python scripts/smoke_test.py --all
"""

import sys
import os
import argparse
import pandas as pd

# Wire src/ into path regardless of where this script is run from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from main import score

SCRAPED_PATH = os.path.join(REPO_ROOT, 'data', 'external', 'scraped_jobs_2026.csv')
FRAUD_PATH   = os.path.join(REPO_ROOT, 'data', 'processed', 'val.csv')


def run_smoke_test(n_scraped=20, run_all=False):
    # ── Load scraped 2026 jobs (all legitimate, fraudulent=0) ────────────────
    scraped_df = pd.read_csv(SCRAPED_PATH)
    if not run_all:
        scraped_df = scraped_df.sample(n=min(n_scraped, len(scraped_df)), random_state=42)

    print(f'\n{"="*60}')
    print(f'Guardian Recruit — End-to-End Smoke Test')
    print(f'{"="*60}')
    print(f'Scoring {len(scraped_df):,} scraped 2026 jobs (all legitimate)...\n')

    results = []
    for _, row in scraped_df.iterrows():
        result = score(row.to_dict())
        results.append({
            'title':         str(row.get('title', ''))[:50],
            'fraud_score':   round(result['fraud_score'], 4),
            'label':         result['label'],
            'bert_score':    round(result['bert_score'], 4),
            'outlier_score': round(result['outlier_score'], 4),
        })

    results_df = pd.DataFrame(results)

    # ── Summary stats ─────────────────────────────────────────────────────────
    flagged    = results_df[results_df['label'] == 'FRAUD']
    legitimate = results_df[results_df['label'] == 'LEGITIMATE']

    print(f'Results on scraped 2026 jobs (expected label: LEGITIMATE)')
    print(f'  Total scored      : {len(results_df):,}')
    print(f'  Correctly cleared : {len(legitimate):,}  ({len(legitimate)/len(results_df):.1%})')
    print(f'  Wrongly flagged   : {len(flagged):,}  ({len(flagged)/len(results_df):.1%})  ← false positives')
    print(f'  Avg fraud score   : {results_df["fraud_score"].mean():.4f}')
    print(f'  Max fraud score   : {results_df["fraud_score"].max():.4f}')

    # ── Show any false positives ───────────────────────────────────────────────
    if len(flagged) > 0:
        print(f'\nFalse positives (legitimate jobs flagged as fraud):')
        print(flagged[['title', 'fraud_score', 'bert_score', 'outlier_score']].to_string(index=False))
    else:
        print('\nNo false positives — all scraped jobs correctly cleared.')

    # ── Score distribution ────────────────────────────────────────────────────
    print(f'\nFraud score distribution (scraped jobs):')
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    labels = ['0.0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5', '0.5–1.0']
    results_df['bucket'] = pd.cut(results_df['fraud_score'], bins=bins, labels=labels, right=False)
    print(results_df['bucket'].value_counts().sort_index().to_string())

    # ── Spot check known fraud patterns ───────────────────────────────────────
    print(f'\n{"="*60}')
    print('Spot check — known fraud patterns:')
    print(f'{"="*60}')

    test_cases = [
        {
            'label': 'Classic fraud (no logo, no questions, vague)',
            'posting': {
                'title':           'Work From Home — Unlimited Earnings',
                'description':     'No experience needed. Great opportunity. Send $50 deposit to get your starter kit.',
                'company_profile': '',
                'requirements':    '',
                'has_company_logo': 0,
                'has_questions':    0,
                'employment_type':  'Unknown',
                'salary_range':     '80000-120000',
            },
        },
        {
            'label': '2026 scam pattern (biometric verification)',
            'posting': {
                'title':           'Remote Data Entry Specialist',
                'description':     'Earn $5000/week from home. Biometric Video Verification Sync required. Hardware Security Deposit of $200 refunded after 30 days.',
                'company_profile': '',
                'requirements':    'No experience necessary. Must have internet connection.',
                'has_company_logo': 0,
                'has_questions':    0,
                'employment_type':  'Part-time',
                'salary_range':     '100000-150000',
            },
        },
        {
            'label': 'Legitimate (complete, professional)',
            'posting': {
                'title':           'Senior Software Engineer',
                'description':     'Join our engineering team to build scalable distributed systems. ' * 15,
                'company_profile': 'Fortune 500 technology company with 10,000+ employees worldwide.',
                'requirements':    "Bachelor's degree in Computer Science. 5+ years Python. Experience with AWS.",
                'has_company_logo': 1,
                'has_questions':    1,
                'employment_type':  'Full-time',
                'salary_range':     '140000-180000',
            },
        },
    ]

    for case in test_cases:
        result = score(case['posting'])
        status = 'PASS' if (
            (case['label'].startswith('Legitimate') and result['label'] == 'LEGITIMATE') or
            (not case['label'].startswith('Legitimate') and result['label'] == 'FRAUD')
        ) else 'FAIL'
        print(f'  [{status}] {case["label"]}')
        print(f'        → {result["label"]}  (fraud_score={result["fraud_score"]:.4f}  bert={result["bert_score"]:.4f})')

    print(f'\n{"="*60}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smoke test the fraud detection pipeline.')
    parser.add_argument('--n',   type=int, default=20, help='Number of scraped jobs to score (default: 20)')
    parser.add_argument('--all', action='store_true',  help='Score all scraped jobs')
    args = parser.parse_args()

    run_smoke_test(n_scraped=args.n, run_all=args.all)
