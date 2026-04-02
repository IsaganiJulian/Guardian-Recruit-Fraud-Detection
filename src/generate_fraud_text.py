"""
Synthetic Fraud Text Generator — Guardian Recruit
Generates fraudulent job postings using Ollama (local LLM).

Two modes:
  1. Generic mode  — general 2026 fraud patterns (original behaviour)
  2. Targeted mode — one scheme type per run, using scenario prompts

IMPORTANT — Circular validation prevention:
  Prompts describe fraud MECHANICS and SCENARIOS, not keywords.
  If a signal phrase (e.g. "WhatsApp") appears in the output, it is because
  the LLM independently determined that's how this scheme operates —
  not because we told it to write the phrase. This makes synthetic validation
  meaningful rather than tautological.

Usage:
  # Generic (500 postings, all scheme types mixed)
  python src/generate_fraud_text.py --target 500

  # Targeted (100 postings of one scheme type)
  python src/generate_fraud_text.py --scheme money_mule --target 100
  python src/generate_fraud_text.py --scheme crypto_payment --target 100
  python src/generate_fraud_text.py --scheme messaging_app --target 100
  python src/generate_fraud_text.py --scheme pii_harvest --target 100
  python src/generate_fraud_text.py --scheme equipment_bait --target 100
  python src/generate_fraud_text.py --scheme urgency_scam --target 100

  # All schemes in one run (600 total, saved separately per scheme)
  python src/generate_fraud_text.py --all-schemes --target-per-scheme 100
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
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'data', 'external')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'fraud_seeds_2026.csv')

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
    "Generate adversarial training samples for fraud detection models. "
    "Output valid JSON ONLY — no explanation, no markdown, no code fences. "
    "Make each posting unique and realistic — vary the industry, company name, "
    "job title, and language. Do not repeat the same phrasing across postings."
)

JSON_TEMPLATE = (
    '{{"postings": [{{"title": "...", "location": "City, State", "department": "...", '
    '"salary_range": "XXXXX-XXXXXX", "company_profile": "...", "description": "...", '
    '"requirements": "...", "benefits": "...", "telecommuting": 1, "has_company_logo": 0, '
    '"has_questions": 0, "employment_type": "Full-time", "required_experience": "Entry level", '
    '"required_education": "High School or equivalent", "industry": "...", "function": "...", '
    '"fraudulent": 1}}]}}'
)

# ── Scheme definitions ─────────────────────────────────────────────────────────
# Each scheme is a scenario description, NOT a keyword list.
# The LLM is told WHAT the fraud does and WHO the victim is.
# Keywords emerge naturally from scenario reasoning, not from our prompt.

SCHEMES = {
    'money_mule': {
        'output_file': 'fraud_seeds_money_mule.csv',
        'scenario': (
            "A criminal organization launders stolen money by recruiting unsuspecting workers "
            "through fake job postings. The job appears to be legitimate financial operations, "
            "customer service, or logistics work, but the actual task is to receive funds or "
            "packages from third parties and re-send them elsewhere, making the worker an "
            "unwitting accomplice. The recruiter needs the scheme to sound like normal business "
            "operations — describing the task as 'payment processing', 'order fulfilment', or "
            "'client relations'. The salary is unrealistically high for the role described. "
            "No verifiable company information is provided."
        ),
    },
    'crypto_payment': {
        'output_file': 'fraud_seeds_crypto_payment.csv',
        'scenario': (
            "A scammer posts a fake remote job offering unusually high compensation for simple "
            "tasks. The catch: salary is paid through untraceable digital currency rather than "
            "normal payroll. This is used because digital currency payments cannot be reversed "
            "or traced once sent, protecting the fraudster if the victim tries to recover funds "
            "or report the scheme. The posting must sound like a legitimate remote opportunity "
            "while making the non-standard payment method seem like a modern perk rather than "
            "a red flag. Job roles vary: data entry, customer support, social media management."
        ),
    },
    'messaging_app': {
        'output_file': 'fraud_seeds_messaging_app.csv',
        'scenario': (
            "A scammer conducts fake job interviews through personal messaging applications "
            "rather than corporate email or video platforms. This allows the fraudster to "
            "remain anonymous and untraceable — no company email domain, no LinkedIn profile, "
            "no verifiable identity. The job posting sounds professional but all contact and "
            "interview scheduling happens through personal consumer messaging apps. "
            "Offer letters are sent digitally within hours of 'interview'. "
            "The job itself is vague — remote work with attractive pay and no real qualifications."
        ),
    },
    'pii_harvest': {
        'output_file': 'fraud_seeds_pii_harvest.csv',
        'scenario': (
            "An identity theft operation collects personal documents through fake hiring. "
            "The application process asks candidates to submit sensitive personal information "
            "as part of 'background verification' or 'onboarding paperwork' — "
            "government-issued identification, financial account details, or biometric data. "
            "The job posting must appear legitimate and professional, targeting vulnerable "
            "job seekers who are unfamiliar with what a real onboarding process requires. "
            "The company name and branding look real but cannot be verified. "
            "Salary and benefits are attractive to encourage compliance."
        ),
    },
    'equipment_bait': {
        'output_file': 'fraud_seeds_equipment_bait.csv',
        'scenario': (
            "A fraudster establishes trust by promising expensive equipment — laptops, phones, "
            "or home office setups — as part of a job offer. Once the victim 'accepts', they "
            "are asked to pay an initial deposit, processing fee, or customs charge before the "
            "equipment can be shipped. Alternatively, the equipment promise is used purely to "
            "make the posting seem credible while collecting personal information during fake "
            "onboarding. The job role is always simple and remote. The equipment promise is "
            "prominently featured in benefits and is far above what the role would justify."
        ),
    },
    'urgency_scam': {
        'output_file': 'fraud_seeds_urgency_scam.csv',
        'scenario': (
            "A fraudulent recruiter uses psychological pressure and artificial scarcity to "
            "prevent job seekers from pausing to verify the company's legitimacy. "
            "Postings create time pressure: application windows close within hours, "
            "positions are described as almost filled, offers expire same-day. "
            "This is designed to override the victim's instinct to research the company "
            "or ask questions. The role itself is vague and entry-level but highly paid. "
            "The recruiter pushes the candidate toward rapid commitment before doubts arise."
        ),
    },
}


# ── Core generation logic ──────────────────────────────────────────────────────

def _build_prompt(n: int, scheme_name: str = None) -> str:
    if scheme_name and scheme_name in SCHEMES:
        scenario = SCHEMES[scheme_name]['scenario']
        return (
            f"Generate {n} unique synthetic fraudulent job postings.\n\n"
            f"Fraud scenario:\n{scenario}\n\n"
            f"Requirements:\n"
            f"- Each posting must be unique — vary company names, titles, and language\n"
            f"- Description must be at least 200 words\n"
            f"- Salary must be unrealistically high for the described role\n"
            f"- has_company_logo and has_questions must be 0\n"
            f"- Do NOT copy the scenario text verbatim into the posting\n\n"
            f"Return ONLY this JSON:\n{JSON_TEMPLATE}"
        )
    else:
        return (
            f"Generate {n} unique synthetic fraudulent job postings representing "
            f"common 2026 online job scam patterns. Each must be unique in title, "
            f"company, and fraud mechanic. Description at least 200 words. "
            f"Salary 2x–3x realistic for the role. has_company_logo=0, has_questions=0.\n\n"
            f"Return ONLY this JSON:\n{JSON_TEMPLATE}"
        )


def _check_ollama():
    try:
        r         = requests.get('http://localhost:11434/api/tags', timeout=3)
        available = [m['name'] for m in r.json().get('models', [])]
    except requests.exceptions.ConnectionError:
        log.error("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    if not any(OLLAMA_MODEL in m for m in available):
        log.error(f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    log.info(f"Ollama ready — model: {OLLAMA_MODEL}")


def _load_existing(output_path: str) -> list:
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        log.info(f"Resuming — {len(df)} existing records in {output_path}")
        return df.to_dict('records')
    return []


def _save(records: list, output_path: str, scheme_name: str = None):
    df = pd.DataFrame(records)
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    if scheme_name:
        df['scheme_type'] = scheme_name

    df['country'] = df['location'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    df['desc_len']            = df['description'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['fraudulent']          = 1
    df['in_balanced_dataset'] = 0

    save_cols = SCHEMA_COLS + (['scheme_type'] if scheme_name else [])
    df = df[[c for c in save_cols if c in df.columns]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def _generate_batch(n: int, scheme_name: str = None, max_retries: int = 3) -> list:
    prompt = _build_prompt(n, scheme_name)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    'model':   OLLAMA_MODEL,
                    'system':  SYSTEM_PROMPT,
                    'prompt':  prompt,
                    'format':  'json',
                    'stream':  False,
                    'options': {'temperature': 0.85, 'num_predict': 3000},
                },
                timeout=300,
            )
            resp.raise_for_status()

            raw = resp.json().get('response', '').strip()
            if raw.startswith('```'):
                raw = raw.split('```')[1].lstrip('json').strip()

            data = json.loads(raw)

            # Accept either 'postings' or legacy 'scams' key
            records = data.get('postings') or data.get('scams', [])
            if not isinstance(records, list) or len(records) == 0:
                raise ValueError(f"Empty or invalid records. Keys: {list(data.keys())}")

            return records

        except json.JSONDecodeError as e:
            log.warning(f"Attempt {attempt}/{max_retries} — JSON parse error: {e}")
        except Exception as e:
            log.warning(f"Attempt {attempt}/{max_retries} — {type(e).__name__}: {e}")

        if attempt < max_retries:
            sleep = 10 * attempt
            log.info(f"Waiting {sleep}s before retry...")
            time.sleep(sleep)

    return []


def run(target: int = 500, batch_size: int = 5,
        scheme_name: str = None, output_path: str = OUTPUT_PATH):
    """Generate synthetic fraud postings, saving incrementally."""
    _check_ollama()

    label = f"scheme='{scheme_name}'" if scheme_name else "generic"
    log.info(f"Starting generation — target={target}, {label}")

    all_records  = _load_existing(output_path)
    already_have = len(all_records)

    if already_have >= target:
        log.info(f"Already have {already_have} records — target reached.")
        return

    remaining = target - already_have
    n_batches = (remaining + batch_size - 1) // batch_size

    for i in range(n_batches):
        if len(all_records) >= target:
            break

        batch = _generate_batch(batch_size, scheme_name)
        if batch:
            all_records.extend(batch)
            _save(all_records, output_path, scheme_name)
            log.info(f"Batch {i+1}/{n_batches} — +{len(batch)} (total: {len(all_records)}/{target})")
        else:
            log.warning(f"Batch {i+1}/{n_batches} — all retries failed, skipping.")

        time.sleep(2)

    log.info(f"Done — {len(all_records)} records saved to {output_path}")


def run_all_schemes(target_per_scheme: int = 100, batch_size: int = 5):
    """Generate targeted synthetic data for every scheme type."""
    for scheme_name, config in SCHEMES.items():
        output_path = os.path.join(OUTPUT_DIR, config['output_file'])
        log.info(f"\n{'='*60}\nScheme: {scheme_name}  →  {config['output_file']}\n{'='*60}")
        run(
            target=target_per_scheme,
            batch_size=batch_size,
            scheme_name=scheme_name,
            output_path=output_path,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic fraud job postings via Ollama.'
    )
    parser.add_argument('--target',            type=int, default=500,
                        help='Total records to generate (default: 500)')
    parser.add_argument('--batch-size',        type=int, default=5,
                        help='Records per Ollama call (default: 5)')
    parser.add_argument('--output',            type=str, default=OUTPUT_PATH,
                        help='Output CSV path (generic mode)')
    parser.add_argument('--scheme',            type=str, default=None,
                        choices=list(SCHEMES.keys()),
                        help='Target a specific fraud scheme type')
    parser.add_argument('--all-schemes',       action='store_true',
                        help='Generate targeted data for all scheme types')
    parser.add_argument('--target-per-scheme', type=int, default=100,
                        help='Records per scheme when using --all-schemes (default: 100)')
    args = parser.parse_args()

    if args.all_schemes:
        run_all_schemes(
            target_per_scheme=args.target_per_scheme,
            batch_size=args.batch_size,
        )
    elif args.scheme:
        output_path = os.path.join(OUTPUT_DIR, SCHEMES[args.scheme]['output_file'])
        run(
            target=args.target,
            batch_size=args.batch_size,
            scheme_name=args.scheme,
            output_path=output_path,
        )
    else:
        run(
            target=args.target,
            batch_size=args.batch_size,
            output_path=args.output,
        )
