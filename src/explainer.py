"""
Explainer — RAG-based plain-English explanation of fraud detection results.

LLM priority order:
  1. Groq (llama-3.3-70b-versatile) — fast cloud inference, requires GROQ_API_KEY in .env
  2. Ollama (llama3.2)              — local inference, requires Ollama running
  3. Template                       — rule-based fallback, no LLM required

Set GROQ_API_KEY in .env to enable Groq.
"""

import os
import requests
from dotenv import load_dotenv
from vector_store import search_similar

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL   = 'llama-3.3-70b-versatile'

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.2'

SYSTEM_PROMPT = (
    "You are a fraud detection analyst for an online job platform. "
    "Explain in plain English (3-5 sentences) why a job posting was flagged. "
    "Be specific to the signals provided. Do not use bullet points. "
    "Do not repeat the score. Do not say 'the model'."
)


# ── LLM availability checks ───────────────────────────────────────────────────

def _groq_available() -> bool:
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def _ollama_available() -> bool:
    try:
        r      = requests.get('http://localhost:11434/api/tags', timeout=2)
        models = [m['name'] for m in r.json().get('models', [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(result: dict, similar: list[dict]) -> str:
    features = result['features']

    rag_context = 'No closely matching fraud cases found.'
    if similar and result['label'] == 'FRAUD':
        rag_context = '\n'.join(
            f"Case {i+1} (similarity {c['similarity']:.0%}): \"{c['title']}\" — {c['snippet']}"
            for i, c in enumerate(similar[:2])
        )

    return (
        f"Job posting analysis:\n"
        f"- Fraud score: {result['fraud_score']:.0%}\n"
        f"- Label: {result['label']}\n"
        f"- BERT score: {result['bert_score']:.3f}\n"
        f"- Outlier score: {result['outlier_score']:.3f}\n"
        f"- Has company logo: {'Yes' if features.get('has_company_logo') else 'No'}\n"
        f"- Has screening questions: {'Yes' if features.get('has_questions') else 'No'}\n"
        f"- Description length: {int(features.get('desc_len', 0))} characters\n\n"
        f"Similar known fraud cases:\n{rag_context}\n\n"
        f"Explain why this posting "
        f"{'was flagged as fraudulent' if result['label'] == 'FRAUD' else 'appears legitimate'}."
    )


# ── LLM backends ─────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    resp = requests.post(
        GROQ_URL,
        headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type':  'application/json',
        },
        json={
            'model': GROQ_MODEL,
            'messages': [
                {'role': 'system',  'content': SYSTEM_PROMPT},
                {'role': 'user',    'content': prompt},
            ],
            'temperature': 0.3,
            'max_tokens':  200,
        },
        timeout=15,
    )
    resp.raise_for_status()
    text = resp.json()['choices'][0]['message']['content'].strip()
    if not text:
        raise ValueError('Empty Groq response')
    return text


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            'model':   OLLAMA_MODEL,
            'system':  SYSTEM_PROMPT,
            'prompt':  prompt,
            'stream':  False,
            'options': {'temperature': 0.3, 'num_predict': 200},
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json().get('response', '').strip()
    if not text:
        raise ValueError('Empty Ollama response')
    return text


# ── Template fallback ─────────────────────────────────────────────────────────

def _template_explanation(result: dict, similar: list[dict]) -> str:
    score    = result['fraud_score']
    bert     = result['bert_score']
    features = result['features']

    if result['label'] != 'FRAUD':
        return (
            f"This posting appears legitimate with a low fraud score of {score:.0%}. "
            "The language, metadata, and structure are consistent with genuine job listings."
        )

    parts = [
        f"This posting received a fraud score of {score:.0%}, "
        f"primarily driven by suspicious language patterns (BERT score: {bert:.2f})."
    ]
    if not features.get('has_company_logo'):
        parts.append("The absence of a company logo is a strong fraud indicator.")
    if not features.get('has_questions'):
        parts.append("No screening questions were included, which is uncommon for legitimate employers.")
    if features.get('desc_len', 1000) < 200:
        parts.append("The description is unusually short — a common pattern in fraudulent listings.")
    if features.get('outlier_score', 0) < 0:
        parts.append("The metadata profile is anomalous compared to legitimate postings.")
    if similar:
        parts.append(f'This closely resembles known scams, including: "{similar[0]["title"]}".')

    return ' '.join(parts)


# ── Public interface ──────────────────────────────────────────────────────────

def explain(result: dict, posting_text: str) -> dict:
    """
    Generate a structured explanation for a fraud detection result.

    Args:
        result:       Output dict from pipeline.predict().
        posting_text: Combined posting text for ChromaDB similarity query.

    Returns:
        {
            "explanation":   str  — plain-English explanation,
            "similar_cases": list — top-3 similar known fraud postings,
            "source":        str  — "groq" | "ollama" | "template",
        }
    """
    similar = search_similar(posting_text, n=3)
    prompt  = _build_prompt(result, similar)

    # Priority: Groq → Ollama → Template
    for backend_name, available_fn, call_fn in [
        ('groq',   _groq_available,   lambda: _call_groq(prompt)),
        ('ollama', _ollama_available, lambda: _call_ollama(prompt)),
    ]:
        if available_fn():
            try:
                return {
                    'explanation':   call_fn(),
                    'similar_cases': similar,
                    'source':        backend_name,
                }
            except Exception:
                continue  # fall through to next backend

    return {
        'explanation':   _template_explanation(result, similar),
        'similar_cases': similar,
        'source':        'template',
    }
