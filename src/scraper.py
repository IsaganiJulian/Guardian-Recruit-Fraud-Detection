# 2026 Data collection script
"""
Live Job Scraper — Guardian Recruit Fraud Detection
Scrapes real 2026 job listings from public sources and outputs a CSV
formatted to match train_clean_v1.csv for model retraining.

Sources:
  - The Muse API  (no auth required, well-structured JSON)
  - RemoteOK API  (no auth required, returns JSON array)

Output: data/external/scraped_jobs_2026.csv
"""

import os
import time
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema matching train_clean_v1.csv
# ---------------------------------------------------------------------------
SCHEMA_COLS = [
    'title', 'location', 'department', 'salary_range', 'company_profile',
    'description', 'requirements', 'benefits', 'telecommuting',
    'has_company_logo', 'has_questions', 'employment_type',
    'required_experience', 'required_education', 'industry', 'function',
    'fraudulent', 'in_balanced_dataset', 'country', 'desc_len',
]

# ---------------------------------------------------------------------------
# Source 1: The Muse API
# ---------------------------------------------------------------------------
MUSE_API_URL = "https://www.themuse.com/api/public/jobs"

def scrape_the_muse(max_pages=10, page_size=20):
    """
    Fetches job listings from The Muse public API.
    Returns a list of raw dicts.
    """
    records = []
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                MUSE_API_URL,
                params={'page': page, 'count': page_size, 'descending': 'true'},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get('results', [])
            if not results:
                log.info(f"The Muse: no more results at page {page}.")
                break

            for job in results:
                company = job.get('company', {})
                locations = job.get('locations', [{}])
                levels = job.get('levels', [{}])
                categories = job.get('categories', [{}])

                records.append({
                    'title':             job.get('name', ''),
                    'location':          locations[0].get('name', '') if locations else '',
                    'department':        categories[0].get('name', '') if categories else '',
                    'salary_range':      '',
                    'company_profile':   company.get('name', ''),
                    'description':       job.get('contents', ''),
                    'requirements':      '',
                    'benefits':          '',
                    'telecommuting':     1 if 'remote' in job.get('name', '').lower() else 0,
                    'has_company_logo':  1 if company.get('name') else 0,
                    'has_questions':     0,
                    'employment_type':   levels[0].get('name', '') if levels else '',
                    'required_experience': levels[0].get('name', '') if levels else '',
                    'required_education': '',
                    'industry':          categories[0].get('name', '') if categories else '',
                    'function':          categories[0].get('name', '') if categories else '',
                    'fraudulent':        0,
                    'in_balanced_dataset': 0,
                })

            log.info(f"The Muse: page {page}/{max_pages} — {len(results)} jobs fetched.")
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            log.warning(f"The Muse page {page} failed: {e}")
            time.sleep(5)

    log.info(f"The Muse total: {len(records)} records.")
    return records


# ---------------------------------------------------------------------------
# Source 2: RemoteOK API
# ---------------------------------------------------------------------------
REMOTEOK_API_URL = "https://remoteok.com/api"

def scrape_remoteok():
    """
    Fetches job listings from RemoteOK public API.
    Returns a list of raw dicts.
    """
    records = []
    try:
        resp = requests.get(
            REMOTEOK_API_URL,
            headers={'User-Agent': 'Guardian-Recruit-Research/1.0'},
            timeout=20
        )
        resp.raise_for_status()
        jobs = resp.json()

        # First element is a metadata dict, skip it
        for job in jobs[1:]:
            if not isinstance(job, dict):
                continue
            records.append({
                'title':             job.get('position', ''),
                'location':          job.get('location', 'Remote'),
                'department':        ', '.join(job.get('tags', [])),
                'salary_range':      _parse_remoteok_salary(job),
                'company_profile':   job.get('company', ''),
                'description':       job.get('description', ''),
                'requirements':      '',
                'benefits':          '',
                'telecommuting':     1,
                'has_company_logo':  1 if job.get('company_logo') else 0,
                'has_questions':     0,
                'employment_type':   'Full-time',
                'required_experience': '',
                'required_education': '',
                'industry':          ', '.join(job.get('tags', [])),
                'function':          job.get('position', ''),
                'fraudulent':        0,
                'in_balanced_dataset': 0,
            })

        log.info(f"RemoteOK total: {len(records)} records.")

    except requests.exceptions.RequestException as e:
        log.warning(f"RemoteOK failed: {e}")

    return records


def _parse_remoteok_salary(job):
    """Extract salary_range string from RemoteOK salary fields."""
    low  = job.get('salary_min') or job.get('salary')
    high = job.get('salary_max')
    if low and high:
        return f"{int(low)}-{int(high)}"
    if low:
        return f"{int(low)}"
    return ''


# ---------------------------------------------------------------------------
# Post-processing: conform to train_clean_v1.csv schema
# ---------------------------------------------------------------------------
def _postprocess(records):
    """
    Converts raw record list → cleaned DataFrame matching train_clean_v1.csv.
    Mirrors GuardianCleaner logic so the scraper output feeds directly into
    the same pipeline as historical data.
    """
    df = pd.DataFrame(records)

    # Ensure every schema column exists
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = ''

    # country: first token of location
    df['country'] = (
        df['location']
        .fillna('Unknown')
        .astype(str)
        .str.split(',')
        .str[0]
        .str.strip()
        .replace('', 'Unknown')
    )

    # desc_len
    df['desc_len'] = df['description'].astype(str).str.len()

    # Enforce binary int types
    for col in ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent', 'in_balanced_dataset']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Drop exact duplicates
    df = df.drop_duplicates(subset=['title', 'company_profile', 'description'])

    # Reorder to schema
    df = df[SCHEMA_COLS].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_scraper(output_path=None, muse_pages=10):
    """
    Runs all scrapers, merges results, post-processes, and saves CSV.

    Args:
        output_path: where to save the CSV. Defaults to data/external/scraped_jobs_2026.csv
        muse_pages:  how many pages to fetch from The Muse (20 jobs/page)

    Returns:
        pd.DataFrame of scraped jobs
    """
    if output_path is None:
        repo_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_path = os.path.join(repo_root, 'data', 'external', 'scraped_jobs_2026.csv')

    log.info("Starting scrape...")

    all_records = []
    all_records.extend(scrape_the_muse(max_pages=muse_pages))
    all_records.extend(scrape_remoteok())

    if not all_records:
        log.error("No records collected from any source.")
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = _postprocess(all_records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Saved {len(df):,} jobs → {output_path}")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scrape 2026 job listings.')
    parser.add_argument('--muse-pages', type=int, default=10,
                        help='Number of Muse API pages to fetch (20 jobs/page). Default: 10')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path. Default: data/external/scraped_jobs_2026.csv')
    args = parser.parse_args()

    df = run_scraper(output_path=args.output, muse_pages=args.muse_pages)
    print(f"\nTotal jobs scraped: {len(df):,}")
    print(df[['title', 'location', 'employment_type', 'desc_len']].head(10))
