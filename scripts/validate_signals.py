"""
Signal Validation Script
Validates text_signals.py keyword patterns against:

  Pass 1 — Real training data (train_clean_v1.csv, 2012–2014 era)
    For each signal: fraud_count, legit_count, coverage, precision, lift

  Pass 2 — Synthetic scheme data (data/external/fraud_seeds_*.csv)
    For each Tier 2 signal: how often it fires in the synthetic postings
    that describe the matching scheme vs. unrelated schemes.
    This gives an independent validation signal for post-2014 fraud patterns.

  Also runs TF-IDF to surface high-lift terms we may have missed.

Usage:
    python scripts/validate_signals.py
    python scripts/validate_signals.py --synthetic-only   # skip real data pass
    python scripts/validate_signals.py --no-tfidf         # skip TF-IDF section
"""

import sys
import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_PATH = os.path.join(REPO_ROOT, 'data', 'processed', 'train_clean_v1.csv')
SYNTH_DIR  = os.path.join(REPO_ROOT, 'data', 'external')
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from text_signals import SIGNALS, TIER1_SIGNALS, TIER2_SIGNALS

# Map scheme file names → signal names they should trigger
# Used for Pass 2 "within-scheme recall" check
SCHEME_SIGNAL_MAP = {
    'money_mule':     'money_mule',
    'crypto_payment': 'crypto_payment',
    'messaging_app':  'messaging_app_interview',
    'pii_harvest':    'pii_request',
    'equipment_bait': 'equipment_bait',
    'urgency_scam':   'artificial_urgency',
}


def make_text(row) -> str:
    parts = [
        str(row.get('title', '')           or ''),
        str(row.get('company_profile', '') or ''),
        str(row.get('description', '')     or ''),
        str(row.get('requirements', '')    or ''),
        str(row.get('benefits', '')        or ''),
    ]
    return ' '.join(p for p in parts if p.strip()).lower()


# ── Pass 1: Real training data ────────────────────────────────────────────────

def validate_real(args):
    if not os.path.exists(TRAIN_PATH):
        print(f'[SKIP] Training data not found: {TRAIN_PATH}')
        return

    df = pd.read_csv(TRAIN_PATH)
    df['text'] = df.apply(make_text, axis=1)

    fraud_df = df[df['fraudulent'] == 1]
    legit_df = df[df['fraudulent'] == 0]
    base_rate = len(fraud_df) / len(df)

    n_fraud = len(fraud_df)
    n_legit = len(legit_df)

    print(f'\n{"="*72}')
    print('PASS 1 — REAL TRAINING DATA  (train_clean_v1.csv, 2012–2014)')
    print(f'{"="*72}')
    print(f'Postings: {len(df):,}  |  Fraud: {n_fraud} ({base_rate:.2%})  |  Legit: {n_legit}')
    print(f'{"─"*72}')
    print(f'{"Signal":<26} {"Fraud":>6} {"Legit":>6} {"Cover":>7} {"Prec":>7} {"Lift":>8} {"Wt":>5}')
    print(f'{"─"*72}')

    tier1_names = {s[0] for s in TIER1_SIGNALS}
    results = []

    for name, weight, patterns in SIGNALS:
        combined   = '|'.join(f'(?:{p})' for p in patterns)
        fraud_hits = fraud_df['text'].str.contains(combined, regex=True, na=False).sum()
        legit_hits = legit_df['text'].str.contains(combined, regex=True, na=False).sum()

        total_hits = fraud_hits + legit_hits
        coverage   = fraud_hits / n_fraud if n_fraud else 0
        precision  = fraud_hits / total_hits if total_hits else 0
        lift       = (precision / base_rate) if base_rate else 0

        results.append({
            'signal':    name,
            'weight':    weight,
            'tier':      1 if name in tier1_names else 2,
            'fraud_hits': fraud_hits,
            'legit_hits': legit_hits,
            'coverage':  coverage,
            'precision': precision,
            'lift':      lift,
        })

        flag = ''
        if lift < 1.5:
            flag = ' ← LOW LIFT'
        elif lift > 5:
            flag = ' ★ HIGH'

        tier_tag = '[T1]' if name in tier1_names else '[T2]'
        print(f'{tier_tag} {name:<22} {fraud_hits:>6} {legit_hits:>6} '
              f'{coverage:>6.1%} {precision:>6.1%} {lift:>7.1f}x {weight:>4.2f}{flag}')

    # Weight suggestions
    print(f'\n{"─"*72}')
    print('DATA-DRIVEN WEIGHT SUGGESTIONS (Tier 1 only — real data)')
    print(f'{"─"*72}')

    results_df = pd.DataFrame(results)
    t1_df      = results_df[results_df['tier'] == 1].sort_values('lift', ascending=False)
    max_lift   = t1_df['lift'].max() if not t1_df.empty else 1

    for _, row in t1_df.iterrows():
        suggested = round(min((row['lift'] / max_lift) * 0.5, 0.5), 2)
        diff      = suggested - row['weight']
        direction = '↑' if diff > 0.02 else ('↓' if diff < -0.02 else '=')
        print(f"  {row['signal']:<24}  current={row['weight']:.2f}  "
              f"suggested={suggested:.2f}  {direction}  (lift={row['lift']:.1f}x)")

    if not args.no_tfidf:
        _run_tfidf(df)


def _run_tfidf(df):
    print(f'\n{"─"*72}')
    print('TOP FRAUD-DISCRIMINATING TERMS (TF-IDF on real training data)')
    print('High-lift terms not currently covered by any signal pattern')
    print(f'{"─"*72}')

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=3,
        stop_words='english',
        sublinear_tf=True,
    )
    X      = vectorizer.fit_transform(df['text'])
    terms  = vectorizer.get_feature_names_out()
    labels = df['fraudulent'].values

    fraud_mask = labels == 1
    legit_mask = labels == 0

    fraud_mean = X[fraud_mask].mean(axis=0).A1
    legit_mean = X[legit_mask].mean(axis=0).A1

    with np.errstate(divide='ignore', invalid='ignore'):
        tfidf_lift = np.where(legit_mean > 0, fraud_mean / legit_mean, fraud_mean * 100)

    fraud_doc_freq = np.array((X[fraud_mask] > 0).sum(axis=0)).flatten()

    top_idx = np.argsort(tfidf_lift)[::-1]

    all_patterns     = [p for _, _, patterns in SIGNALS for p in patterns]
    combined_existing = '|'.join(f'(?:{p})' for p in all_patterns)

    print(f'{"Term":<35} {"Lift":>8} {"Fraud docs":>12} {"Covered?":>10}')
    print(f'{"─"*72}')

    shown = 0
    for idx in top_idx:
        if shown >= 30:
            break
        term  = terms[idx]
        lift  = tfidf_lift[idx]
        fdocs = fraud_doc_freq[idx]

        if lift < 2 or fdocs < 3:
            continue

        covered = bool(re.search(combined_existing, term))
        marker  = '✓' if covered else '← MISSING'
        print(f'{term:<35} {lift:>7.1f}x {fdocs:>11}  {marker:>10}')
        shown += 1


# ── Pass 2: Synthetic scheme data ─────────────────────────────────────────────

def validate_synthetic():
    scheme_files = glob.glob(os.path.join(SYNTH_DIR, 'fraud_seeds_*.csv'))
    if not scheme_files:
        print(f'\n[SKIP] No synthetic scheme files found in {SYNTH_DIR}')
        print('       Run: python src/generate_fraud_text.py --all-schemes --target-per-scheme 100')
        return

    print(f'\n{"="*72}')
    print('PASS 2 — SYNTHETIC SCHEME DATA  (Tier 2 signal validation)')
    print(f'{"="*72}')
    print('For each scheme file, shows recall of the matching Tier 2 signal.')
    print('High recall = signal reliably fires on real scheme examples.')
    print('Cross-scheme recall shows false positive rate between schemes.')
    print(f'{"─"*72}')

    # Load all synthetic data
    dfs = {}
    for fpath in sorted(scheme_files):
        fname     = os.path.basename(fpath)
        scheme_id = fname.replace('fraud_seeds_', '').replace('.csv', '')
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            df['_text'] = df.apply(make_text, axis=1)
            dfs[scheme_id] = df
            print(f'  Loaded {fname}: {len(df)} records')
        except Exception as e:
            print(f'  [WARN] Could not load {fname}: {e}')

    if not dfs:
        print('[SKIP] All synthetic files were empty or unreadable.')
        return

    # Build signal lookup for Tier 2
    tier2_lookup = {s[0]: (s[1], s[2]) for s in TIER2_SIGNALS}
    tier1_lookup = {s[0]: (s[1], s[2]) for s in TIER1_SIGNALS}

    print(f'\n{"─"*72}')
    print('WITHIN-SCHEME RECALL  (signal should fire on its own scheme\'s postings)')
    print(f'{"─"*72}')
    print(f'{"Scheme file":<20} {"Signal checked":<26} {"Hits":>6} {"Total":>6} {"Recall":>8}')
    print(f'{"─"*72}')

    recall_results = []
    for scheme_id, signal_name in SCHEME_SIGNAL_MAP.items():
        fname = f'fraud_seeds_{scheme_id}.csv'
        if scheme_id not in dfs:
            print(f'  {scheme_id:<20} {signal_name:<26} {"—":>6} {"—":>6} {"(no file)":>8}')
            continue

        df = dfs[scheme_id]
        if signal_name in tier2_lookup:
            _, patterns = tier2_lookup[signal_name]
        elif signal_name in tier1_lookup:
            _, patterns = tier1_lookup[signal_name]
        else:
            print(f'  {scheme_id:<20} {signal_name:<26} {"—":>6} {"—":>6} {"(unknown)":>8}')
            continue

        combined = '|'.join(f'(?:{p})' for p in patterns)
        hits     = df['_text'].str.contains(combined, regex=True, na=False).sum()
        total    = len(df)
        recall   = hits / total if total else 0

        flag = ' ← LOW' if recall < 0.05 else (' ★ GOOD' if recall >= 0.20 else '')
        print(f'  {scheme_id:<20} {signal_name:<26} {hits:>6} {total:>6} {recall:>7.1%}{flag}')

        recall_results.append({
            'scheme':  scheme_id,
            'signal':  signal_name,
            'hits':    hits,
            'total':   total,
            'recall':  recall,
        })

    # Cross-scheme false positive table
    print(f'\n{"─"*72}')
    print('CROSS-SCHEME SIGNAL MATRIX  (row=scheme file, col=signal fired)')
    print('Diagonal = within-scheme recall. Off-diagonal = cross-trigger rate.')
    print(f'{"─"*72}')

    all_schemes = sorted(dfs.keys())
    col_signals = [(scheme_id, SCHEME_SIGNAL_MAP[scheme_id])
                   for scheme_id in all_schemes if scheme_id in SCHEME_SIGNAL_MAP]

    # Header
    header = f'{"Scheme":<20}'
    for scheme_id, _ in col_signals:
        header += f' {scheme_id[:8]:>9}'
    print(header)
    print('─' * (20 + 10 * len(col_signals)))

    for row_scheme in all_schemes:
        if row_scheme not in dfs:
            continue
        df   = dfs[row_scheme]
        line = f'{row_scheme:<20}'

        for col_scheme, signal_name in col_signals:
            if signal_name in tier2_lookup:
                _, patterns = tier2_lookup[signal_name]
            elif signal_name in tier1_lookup:
                _, patterns = tier1_lookup[signal_name]
            else:
                line += f' {"—":>9}'
                continue

            combined = '|'.join(f'(?:{p})' for p in patterns)
            hits     = df['_text'].str.contains(combined, regex=True, na=False).sum()
            total    = len(df)
            rate     = hits / total if total else 0
            marker   = '*' if row_scheme == col_scheme else ' '
            line    += f' {rate:>7.0%}{marker} '

        print(line)

    # Tier 2 weight suggestions based on synthetic recall
    if recall_results:
        print(f'\n{"─"*72}')
        print('TIER 2 WEIGHT SUGGESTIONS  (based on synthetic recall)')
        print('Higher recall in matching scheme → signal is reliable → weight stays or rises.')
        print('Low recall → signal misses fraud → consider adding patterns.')
        print(f'{"─"*72}')

        rdf = pd.DataFrame(recall_results).sort_values('recall', ascending=False)
        for _, row in rdf.iterrows():
            signal_name = row['signal']
            if signal_name in tier2_lookup:
                current_weight, _ = tier2_lookup[signal_name]
            elif signal_name in tier1_lookup:
                current_weight, _ = tier1_lookup[signal_name]
            else:
                continue

            # Heuristic: recall < 5% → signal probably too narrow, lower weight
            #            recall ≥ 20% → signal reliable, keep or raise weight
            if row['recall'] >= 0.20:
                suggested = round(min(current_weight * 1.1, 0.5), 2)
                direction = '↑ (reliable)'
            elif row['recall'] < 0.05:
                suggested = round(current_weight * 0.7, 2)
                direction = '↓ (low recall — add patterns?)'
            else:
                suggested = current_weight
                direction = '= (moderate)'

            print(f"  {signal_name:<26}  recall={row['recall']:.0%}  "
                  f"current={current_weight:.2f}  suggested={suggested:.2f}  {direction}")

    print(f'\n{"="*72}')
    print('Pass 2 complete.')
    print(f'{"="*72}\n')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Validate keyword signals against training and synthetic data.')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Skip Pass 1 (real training data), run only Pass 2')
    parser.add_argument('--no-tfidf', action='store_true',
                        help='Skip TF-IDF section in Pass 1')
    args = parser.parse_args()

    if not args.synthetic_only:
        validate_real(args)

    validate_synthetic()

    print('Run complete. Review MISSING terms and low-recall Tier 2 signals.')


if __name__ == '__main__':
    main()
