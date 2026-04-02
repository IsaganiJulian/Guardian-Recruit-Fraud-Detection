"""
Guardian Recruit — Fraud Detection App
Streamlit UI backed by FraudDetectionPipeline.

Run with:
    streamlit run app.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Guardian Recruit — Fraud Detection',
    page_icon='🛡️',
    layout='wide',
)

# ── Warm up pipeline once per session ────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return pipeline.warm_up()

with st.spinner('Loading models...'):
    health = load_pipeline()

# ── Header ────────────────────────────────────────────────────────────────────
st.title('🛡️ Guardian Recruit — Fraud Detection')
st.caption('Stream A (BERT) + Stream B (IsolationForest) + Fusion (XGBoost) + RAG Explainability')

# Health status in sidebar
with st.sidebar:
    st.subheader('Pipeline Status')
    for name, info in health.items():
        if not isinstance(info, dict):
            continue
        icon = '✅' if info['ok'] else '⚠️'
        st.write(f"{icon} **{name.replace('_', ' ').title()}** — {info['note']}")

    st.divider()
    st.caption('Threshold: 0.3 · Model: bert-base-uncased · XGBoost fusion')

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader('Analyse a Job Posting')

with st.form('job_form'):
    col1, col2 = st.columns(2)
    with col1:
        title           = st.text_input('Job Title', placeholder='e.g. Software Engineer')
        employment_type = st.selectbox('Employment Type',
                                       ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other', 'Unknown'])
        salary_range    = st.text_input('Salary Range', placeholder='e.g. 60000-90000')
        has_logo        = st.checkbox('Has Company Logo',       value=True)
        has_questions   = st.checkbox('Has Screening Questions', value=True)

    with col2:
        company_profile = st.text_area('Company Profile', height=100,
                                        placeholder='Brief company description...')
        requirements    = st.text_area('Requirements', height=100,
                                        placeholder='Required qualifications...')

    description = st.text_area('Job Description', height=180,
                                placeholder='Full job description...')

    submitted = st.form_submit_button('Analyse Posting', type='primary', use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if submitted:
    if not title and not description:
        st.warning('Please enter at least a title and description.')
        st.stop()

    posting = {
        'title':            title,
        'company_profile':  company_profile,
        'description':      description,
        'requirements':     requirements,
        'employment_type':  employment_type,
        'salary_range':     salary_range,
        'has_company_logo': int(has_logo),
        'has_questions':    int(has_questions),
    }

    with st.spinner('Analysing...'):
        result = pipeline.predict(posting)

    fraud_score = result['fraud_score']
    is_fraud    = result['label'] == 'FRAUD'

    # ── Verdict ───────────────────────────────────────────────────────────────
    st.divider()
    if is_fraud:
        st.error(f'### FRAUDULENT  —  {fraud_score:.0%} fraud probability', icon='🚨')
    else:
        st.success(f'### LEGITIMATE  —  {fraud_score:.0%} fraud probability', icon='✅')

    st.progress(fraud_score, text=f'Fraud probability: {fraud_score:.1%}')

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Fraud Score',    f'{fraud_score:.4f}')
    m2.metric('BERT Score',     f'{result["bert_score"]:.4f}')
    m3.metric('Outlier Score',  f'{result["outlier_score"]:.4f}')
    m4.metric('Latency',        f'{result["latency_ms"]:.0f} ms')

    st.divider()
    left, right = st.columns([1, 1])

    # ── Feature chart ─────────────────────────────────────────────────────────
    with left:
        st.subheader('Feature Breakdown')
        features = result['features']
        labels   = ['BERT Score', 'Outlier Score', 'Has Logo', 'Has Questions', 'Desc Length']
        values   = [
            features['bert_score'],
            max(0.0, features['outlier_score']),
            float(features['has_company_logo']),
            float(features['has_questions']),
            min(features['desc_len'] / 3000, 1.0),
        ]
        colours = [
            '#e74c3c' if features['bert_score']      >  0.3  else '#2ecc71',
            '#e74c3c' if features['outlier_score']   <  0.0  else '#2ecc71',
            '#e74c3c' if not features['has_company_logo']     else '#2ecc71',
            '#e74c3c' if not features['has_questions']        else '#2ecc71',
            '#e74c3c' if features['desc_len']        < 200   else '#2ecc71',
        ]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(labels, values, color=colours)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised value')
        ax.set_title('Signal breakdown')
        ax.legend(handles=[
            mpatches.Patch(color='#e74c3c', label='Fraud signal'),
            mpatches.Patch(color='#2ecc71', label='Normal signal'),
        ], fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── AI explanation ────────────────────────────────────────────────────────
    with right:
        st.subheader('AI Explanation')
        source_label = '🤖 Ollama (llama3.2)' if result['explain_source'] == 'llm' else '📋 Template'
        st.caption(source_label)
        st.info(result['explanation'])

    # ── Triggered keyword signals ─────────────────────────────────────────────
    triggered = result.get('triggered_signals', [])
    if triggered:
        st.divider()
        st.subheader('Fraud Signals Detected')
        signal_labels = {
            'money_mule':         '💸 Money mule / package forwarding',
            'pii_request':        '🪪 Personal ID / PII harvesting',
            'suspicious_contact': '📱 WhatsApp / Telegram interview',
            'crypto_payment':     '₿ Cryptocurrency payment',
            'salary_mismatch':    '💰 Unrealistic salary + no experience',
            'urgency':            '⏱️ Artificial urgency',
            'wfh_unrealistic':    '🌍 Unrealistic remote claim',
            'equipment_bait':     '💻 Equipment / stipend bait',
            'suspicious_domain':  '📧 Suspicious email domain',
        }
        cols = st.columns(min(len(triggered), 3))
        for i, sig in enumerate(triggered):
            cols[i % 3].error(signal_labels.get(sig['signal'], sig['signal']))

        if result.get('keyword_score', 0) > result.get('ml_score', 0):
            st.caption(
                f"⚠️ ML model score was low ({result['ml_score']:.0%}) — "
                f"keyword rules engine detected fraud (score: {result['keyword_score']:.0%}). "
                "This posting uses professionally-written language to evade NLP detection."
            )

    # ── Similar fraud cases ───────────────────────────────────────────────────
    similar = result.get('similar_cases', [])
    if similar and is_fraud:
        st.divider()
        st.subheader('Similar Known Fraud Cases')
        st.caption('Retrieved from training database via ChromaDB')
        for i, case in enumerate(similar, 1):
            with st.expander(f"#{i} · {case['title']}  (similarity: {case['similarity']:.0%})"):
                c1, c2, c3 = st.columns(3)
                c1.metric('Employment Type', case['employment_type'] or 'Unknown')
                c2.metric('Salary Range',    case['salary_range']    or 'Not listed')
                c3.metric('Has Logo',        'Yes' if case['has_company_logo'] else 'No')
                st.caption(case['snippet'])

    # ── Raw output ────────────────────────────────────────────────────────────
    with st.expander('Raw pipeline output'):
        display = {k: v for k, v in result.items() if k != 'similar_cases'}
        st.json(display)
