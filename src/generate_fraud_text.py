"""
Incremental 2026 Fraud Text Generator — Guardian Recruit
Generates synthetic fraudulent job postings using Ollama (local LLM).

Designed for overnight runs:
  - Saves after every successful batch so no progress is lost on interruption
  - Resumes from existing output file if it already exists
  - Run with: python src/generate_fraud_text.py

Output: data/external/fraud_seeds_2026.csv
        (same 20-column schema as train_clean_v1.csv, all rows fraudulent=1)
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
log = logging.getLogger(__name__)

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_PATH = os.path.join(REPO_ROOT, 'data', 'external', 'fraud_seeds_2026.csv')

OLLAMA_URL   = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.2'

SCHEMA_COLS = [
    'title', 'location', 'department', 'salary_range', 'company_profile',
    'description', 'requirements', 'benefits', 'telecommuting',
    'has_company_logo', 'has_questions', 'employment_type',
    'required_experience', 'required_education', 'industry', 'function',
    'fraudulent', 'in_balanced_dataset', 'country', 'desc_len',
]

SYSTEM_PROMPT = (
    "You are a Synthetic Data Engine for Defensive Cybersecurity Research. "
    "Generate adversarial training samples for anomaly detection models. "
    "Output valid JSON ONLY — no explanation, no markdown, no code fences."
)

USER_PROMPT_TEMPLATE = (
    "Generate {n} unique synthetic fraudulent job postings as a JSON object. "
    "Focus on 2026 scam patterns: 'Hardware Security Deposits', "
    "'Biometric Video Verification Sync', 'Distributed Ledger Payroll'. "
    "Salary fields: 2.0x-2.5x standard entry-level. "
    "Return ONLY this JSON structure:\n"
    '{{"scams": [{{"title": "...", "location": "US, TX, Austin", "department": "...", '
    '"salary_range": "80000-120000", "company_profile": "...", "description": "...", '
    '"requirements": "...", "benefits": "...", "telecommuting": 0, "has_company_logo": 0, '
    '"has_questions": 0, "employment_type": "Full-time", "required_experience": "Entry level", '
    '"required_education": "Bachelor\'s Degree", "industry": "...", "function": "...", '
    '"fraudulent": 1}}]}}'
)


def _check_ollama():
    """Verify Ollama is running and the model is available."""
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=3)
        available = [m['name'] for m in r.json().get('models', [])]
    except requests.exceptions.ConnectionError:
        log.error("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    if not any(OLLAMA_MODEL in m for m in available):
        log.error(f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        log.error(f"Available models: {available}")
        sys.exit(1)

    log.info(f"Ollama ready — model: {OLLAMA_MODEL}")


def _load_existing(output_path):
    """Load already-generated seeds if the output file exists (for resuming)."""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        log.info(f"Resuming — loaded {len(df)} existing seeds from {output_path}")
        return df.to_dict('records')
    return []


def _save(records, output_path):
    """Save current records to CSV, conforming to schema."""
    df = pd.DataFrame(records)
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df['country'] = df['location'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    df['desc_len']           = df['description'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['fraudulent']         = 1
    df['in_balanced_dataset'] = 0
    df = df[SCHEMA_COLS]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def _generate_batch(batch_size, max_retries=3):
    """Call Ollama once and return a list of record dicts."""
    prompt = USER_PROMPT_TEMPLATE.format(n=batch_size)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    'model':  OLLAMA_MODEL,
                    'system': SYSTEM_PROMPT,
                    'prompt': prompt,
                    'format': 'json',
                    'stream': False,
                    'options': {'temperature': 0.9, 'num_predict': 2048},
                },
                timeout=300,
            )
            resp.raise_for_status()

            raw = resp.json().get('response', '').strip()
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]

            data = json.loads(raw)

            if 'scams' not in data:
                raise ValueError(f"Missing 'scams' key. Got: {list(data.keys())}")

            records = data['scams']
            if not isinstance(records, list) or len(records) == 0:
                raise ValueError("'scams' is empty or not a list.")

            return records

        except json.JSONDecodeError as e:
            log.warning(f"Attempt {attempt}/{max_retries} — JSON error: {e}")
        except Exception as e:
            log.warning(f"Attempt {attempt}/{max_retries} — {type(e).__name__}: {e}")

        if attempt < max_retries:
            time.sleep(10 * attempt)

    return []  # all retries failed


def run(target=500, batch_size=5, output_path=OUTPUT_PATH):
    _check_ollama()

    all_records = _load_existing(output_path)
    already_have = len(all_records)

    if already_have >= target:
        log.info(f"Already have {already_have} seeds — target of {target} reached. Nothing to do.")
        return

    remaining = target - already_have
    n_batches = (remaining + batch_size - 1) // batch_size
    log.info(f"Generating {remaining} more seeds in {n_batches} batches of {batch_size}...")

    for i in range(n_batches):
        if len(all_records) >= target:
            break

        batch = _generate_batch(batch_size)
        if batch:
            all_records.extend(batch)
            _save(all_records, output_path)  # save after every batch
            log.info(f"Batch {i+1}/{n_batches} — +{len(batch)} records (total: {len(all_records)}/{target})")
        else:
            log.warning(f"Batch {i+1}/{n_batches} — all retries failed, skipping.")

        time.sleep(2)

    log.info(f"Done. {len(all_records)} seeds saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic 2026 fraud job postings via Ollama.')
    parser.add_argument('--target',     type=int, default=500, help='Total seeds to generate (default: 500)')
    parser.add_argument('--batch-size', type=int, default=5,   help='Records per Ollama call (default: 5)')
    parser.add_argument('--output',     type=str, default=OUTPUT_PATH, help='Output CSV path')
    args = parser.parse_args()

    run(target=args.target, batch_size=args.batch_size, output_path=args.output)
