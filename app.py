"""
Guardian Recruit — Fraud Detection App
Streamlit UI backed by FraudDetectionPipeline.

Run with:
    streamlit run app.py
"""

import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Download models and data automatically on cold start (HF Spaces / fresh clone)
from startup import ensure_assets
ensure_assets()

from pipeline import pipeline

# ── Training queue helpers (SQLite) ──────────────────────────────────────────
import sqlite3

REPO_ROOT  = os.path.dirname(__file__)
DB_DIR     = os.path.join(REPO_ROOT, 'data', 'feedback')
QUEUE_DB   = os.path.join(DB_DIR, 'training_queue.db')

def _get_conn():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(QUEUE_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_queue (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            title         TEXT,
            description   TEXT,
            requirements  TEXT,
            company_profile TEXT,
            employment_type TEXT,
            salary_range  TEXT,
            has_company_logo INTEGER,
            has_questions INTEGER,
            fraud_score   REAL,
            model_label   TEXT,
            human_label   TEXT
        )
    """)
    conn.commit()
    return conn

def _load_queue() -> pd.DataFrame:
    try:
        conn = _get_conn()
        df   = pd.read_sql_query("SELECT * FROM training_queue ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def save_to_training_queue(posting: dict, result: dict) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO training_queue
        (timestamp, title, description, requirements, company_profile,
         employment_type, salary_range, has_company_logo, has_questions,
         fraud_score, model_label, human_label)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.utcnow().isoformat(timespec='seconds'),
        posting.get('title', ''),
        posting.get('description', ''),
        posting.get('requirements', ''),
        posting.get('company_profile', ''),
        posting.get('employment_type', ''),
        posting.get('salary_range', ''),
        int(posting.get('has_company_logo', 0)),
        int(posting.get('has_questions', 0)),
        round(result['fraud_score'], 4),
        result['label'],
        result['label'],
    ))
    conn.commit()
    conn.close()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Guardian Recruit — Threat Detection',
    page_icon='🛡️',
    layout='wide',
)

# ── Cybersecurity Theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base dark theme */
  .stApp {
      background-color: #0a0e17;
      color: #c9d1d9;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background-color: #0d1117;
      border-right: 1px solid #00ff9d22;
  }

  /* All text inputs and text areas */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
      background-color: #0d1117 !important;
      color: #00ff9d !important;
      border: 1px solid #00ff9d44 !important;
      font-family: 'Courier New', monospace !important;
  }

  /* Form submit button */
  .stFormSubmitButton > button {
      background-color: #00ff9d !important;
      color: #0a0e17 !important;
      font-family: 'Courier New', monospace !important;
      font-weight: bold !important;
      border: none !important;
      letter-spacing: 1px;
  }
  .stFormSubmitButton > button:hover {
      background-color: #00cc7a !important;
  }

  /* Metric boxes */
  [data-testid="stMetric"] {
      background-color: #0d1117;
      border: 1px solid #00ff9d22;
      border-radius: 6px;
      padding: 12px;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; }
  [data-testid="stMetricValue"] {
      color: #00ff9d !important;
      font-family: 'Courier New', monospace !important;
  }

  /* Headers */
  h1, h2, h3 { color: #00ff9d !important; font-family: 'Courier New', monospace !important; }
  .stCaption  { color: #8b949e !important; font-family: 'Courier New', monospace !important; }

  /* Divider */
  hr { border-color: #00ff9d22 !important; }

  /* Expander */
  .streamlit-expanderHeader {
      background-color: #0d1117 !important;
      color: #00ff9d !important;
      font-family: 'Courier New', monospace !important;
      border: 1px solid #00ff9d22 !important;
  }

  /* Alert boxes */
  .stAlert {
      font-family: 'Courier New', monospace !important;
      border-radius: 4px !important;
  }

  /* Checkboxes */
  .stCheckbox label { color: #c9d1d9 !important; }

  /* Labels */
  label { color: #8b949e !important; font-family: 'Courier New', monospace !important; }

  /* Progress bar */
  .stProgress > div > div {
      background-color: #0d1117 !important;
  }
  .stProgress > div > div > div {
      background-color: #00ff9d !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e17; }
  ::-webkit-scrollbar-thumb { background: #00ff9d44; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Warm up pipeline once per session ────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return pipeline.warm_up()

with st.spinner('[ INITIALISING GUARDIAN SYSTEM... ]'):
    health = load_pipeline()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border:1px solid #00ff9d33; border-radius:6px; padding:20px; margin-bottom:20px; background:#0d1117;">
    <h1 style="margin:0; font-size:1.8rem; letter-spacing:2px;">🛡️ GUARDIAN RECRUIT</h1>
    <p style="margin:6px 0 0 0; color:#8b949e; font-family:'Courier New',monospace; font-size:0.85rem;">
        THREAT DETECTION SYSTEM &nbsp;|&nbsp; STREAM A: BERT &nbsp;·&nbsp; STREAM B: ISOLATION FOREST &nbsp;·&nbsp; FUSION: XGBOOST + SHAP
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### [ SYSTEM STATUS ]")
    for name, info in health.items():
        if not isinstance(info, dict):
            continue
        icon   = '🟢' if info['ok'] else '🔴'
        status = 'ONLINE' if info['ok'] else 'OFFLINE'
        st.markdown(
            f"<span style='font-family:Courier New;font-size:0.8rem;'>{icon} "
            f"<b style='color:#00ff9d'>{name.replace('_',' ').upper()}</b> — "
            f"<span style='color:#8b949e'>{status}</span></span>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        "<span style='font-family:Courier New;font-size:0.75rem;color:#8b949e;'>"
        "THRESHOLD: 0.30<br>MODEL: bert-base-uncased<br>FUSION: XGBoost<br>ROC-AUC: 0.9718"
        "</span>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### [ GLOBAL FEATURE IMPORTANCE ]")
    _beeswarm = os.path.join(REPO_ROOT, 'docs', 'shap_beeswarm.png')
    if os.path.exists(_beeswarm):
        st.image(_beeswarm, use_column_width=True)
        st.caption('SHAP beeswarm — 500 validation postings')

    st.divider()
    st.markdown("### [ TRAINING QUEUE ]")
    queue_df = _load_queue()
    n_queue  = len(queue_df)
    st.markdown(
        f"<span style='font-family:Courier New;font-size:0.8rem;color:#00ff9d;'>"
        f"{n_queue} posting(s) saved for retraining</span>",
        unsafe_allow_html=True
    )
    if n_queue > 0:
        csv_bytes = queue_df.to_csv(index=False).encode()
        st.download_button(
            label='⬇ Download training_queue.csv',
            data=csv_bytes,
            file_name='training_queue.csv',
            mime='text/csv',
            use_container_width=True,
        )

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### [ SUBMIT TARGET FOR ANALYSIS ]")

with st.form('job_form'):
    col1, col2 = st.columns(2)
    with col1:
        title           = st.text_input('JOB TITLE', placeholder='e.g. Software Engineer')
        employment_type = st.selectbox('EMPLOYMENT TYPE',
                                       ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other', 'Unknown'])
        salary_range    = st.text_input('SALARY RANGE', placeholder='e.g. 60000-90000')
        has_logo        = st.checkbox('Has Company Logo',        value=True)
        has_questions   = st.checkbox('Has Screening Questions', value=True)

    with col2:
        company_profile = st.text_area('COMPANY PROFILE', height=100,
                                        placeholder='Brief company description...')
        requirements    = st.text_area('REQUIREMENTS', height=100,
                                        placeholder='Required qualifications...')

    description = st.text_area('JOB DESCRIPTION', height=180,
                                placeholder='Full job description...')

    submitted = st.form_submit_button(
        '⟫  RUN THREAT ANALYSIS',
        type='primary',
        use_container_width=True
    )

# ── Session state init ────────────────────────────────────────────────────────
if 'result'  not in st.session_state: st.session_state['result']  = None
if 'posting' not in st.session_state: st.session_state['posting'] = None
if 'saved'   not in st.session_state: st.session_state['saved']   = False

# ── Run pipeline ──────────────────────────────────────────────────────────────
if submitted:
    if not title and not description:
        st.warning('[ INPUT ERROR ] Please enter at least a title and description.')
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

    with st.spinner('[ RUNNING THREAT ANALYSIS... ]'):
        result = pipeline.predict(posting)

    st.session_state['result']  = result
    st.session_state['posting'] = posting
    st.session_state['saved']   = False

# ── Render results from session state ─────────────────────────────────────────
if st.session_state['result'] is not None:
    result      = st.session_state['result']
    posting     = st.session_state['posting']
    fraud_score = result['fraud_score']
    is_fraud    = result['label'] == 'FRAUD'

    # ── Verdict ───────────────────────────────────────────────────────────────
    st.divider()
    if is_fraud:
        st.markdown(f"""
        <div style="background:#1a0a0a;border:2px solid #ff4444;border-radius:6px;padding:20px;text-align:center;">
            <span style="font-family:'Courier New',monospace;font-size:1.6rem;font-weight:bold;color:#ff4444;letter-spacing:3px;">
                ⚠ THREAT DETECTED
            </span><br>
            <span style="font-family:'Courier New',monospace;font-size:1.1rem;color:#ff6666;">
                FRAUD PROBABILITY: {fraud_score:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#0a1a0f;border:2px solid #00ff9d;border-radius:6px;padding:20px;text-align:center;">
            <span style="font-family:'Courier New',monospace;font-size:1.6rem;font-weight:bold;color:#00ff9d;letter-spacing:3px;">
                ✓ POSTING CLEARED
            </span><br>
            <span style="font-family:'Courier New',monospace;font-size:1.1rem;color:#00cc7a;">
                FRAUD PROBABILITY: {fraud_score:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(fraud_score, text=f'Threat level: {fraud_score:.1%}')

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('FRAUD SCORE',   f'{fraud_score:.4f}')
    m2.metric('BERT SCORE',    f'{result["bert_score"]:.4f}')
    m3.metric('OUTLIER SCORE', f'{result["outlier_score"]:.4f}')
    m4.metric('LATENCY',       f'{result["latency_ms"]:.0f} ms')

    st.divider()
    left, right = st.columns([1, 1])

    # ── Feature chart ─────────────────────────────────────────────────────────
    with left:
        st.markdown("### [ SIGNAL BREAKDOWN ]")
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
            '#ff4444' if features['bert_score']    >  0.3 else '#00ff9d',
            '#ff4444' if features['outlier_score'] <  0.0 else '#00ff9d',
            '#ff4444' if not features['has_company_logo']  else '#00ff9d',
            '#ff4444' if not features['has_questions']     else '#00ff9d',
            '#ff4444' if features['desc_len']      < 200  else '#00ff9d',
        ]
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.barh(labels, values, color=colours, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalised value', color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#00ff9d22')
        ax.legend(handles=[
            mpatches.Patch(color='#ff4444', label='Threat signal'),
            mpatches.Patch(color='#00ff9d', label='Normal signal'),
        ], fontsize=7, facecolor='#0d1117', labelcolor='#c9d1d9')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── AI explanation ────────────────────────────────────────────────────────
    with right:
        st.markdown("### [ THREAT ANALYSIS ]")
        source = result.get('explain_source', 'template')
        source_label = {
            'groq':     '⚡ Groq (llama-3.3-70b)',
            'ollama':   '🖥️ Ollama (llama3.2)',
            'template': '📋 Rule-based engine',
        }.get(source, source)
        st.caption(f'Source: {source_label}')
        st.markdown(
            f"<div style='background:#0d1117;border:1px solid #00ff9d22;border-radius:4px;"
            f"padding:16px;font-family:Courier New,monospace;font-size:0.85rem;color:#c9d1d9;'>"
            f"{result['explanation']}</div>",
            unsafe_allow_html=True
        )

    # ── Triggered keyword signals ─────────────────────────────────────────────
    triggered = result.get('triggered_signals', [])
    if triggered:
        st.divider()
        st.markdown("### [ THREAT INDICATORS ]")
        signal_labels = {
            'money_mule':             '💸 Money mule / package forwarding',
            'pii_request':            '🪪 PII harvesting',
            'messaging_app_interview':'📱 WhatsApp / Telegram interview',
            'crypto_payment':         '₿ Cryptocurrency payment',
            'compensation_guarantee': '💰 Compensation guarantee',
            'hiring_urgency_2026':    '⏱️ Artificial urgency (2026)',
            'task_scam':              '📲 Task / rating scam',
            'upfront_fee_request':    '💳 Upfront fee request',
            'ats_fee_scam':           '🗃️ ATS fee scam',
            'signal_app_contact':     '🔒 Signal app contact',
            'lookalike_domain':       '🌐 Lookalike domain',
            'vague_company_descriptor':'🏢 Vague company',
            'equipment_bait':         '💻 Equipment bait',
            'suspicious_email_domain':'📧 Suspicious email domain',
            'wfh_unrealistic':        '🌍 Unrealistic WFH claim',
            'artificial_urgency':     '⏱️ Artificial urgency',
        }
        cols = st.columns(min(len(triggered), 3))
        for i, sig in enumerate(triggered):
            label = signal_labels.get(sig['signal'], sig['signal'])
            cols[i % 3].markdown(
                f"<div style='background:#1a0a0a;border:1px solid #ff444466;border-radius:4px;"
                f"padding:8px;font-family:Courier New,monospace;font-size:0.8rem;color:#ff6666;"
                f"text-align:center;'>{label}</div>",
                unsafe_allow_html=True
            )

        if result.get('keyword_score', 0) > result.get('ml_score', 0):
            st.markdown(
                f"<div style='margin-top:12px;font-family:Courier New,monospace;font-size:0.8rem;"
                f"color:#f0a500;'>⚠ ML score was low ({result['ml_score']:.0%}) — "
                f"keyword engine flagged this posting ({result['keyword_score']:.0%}). "
                f"Professionally written language used to evade NLP detection.</div>",
                unsafe_allow_html=True
            )

    # ── Similar fraud cases ───────────────────────────────────────────────────
    similar = result.get('similar_cases', [])
    if similar and is_fraud:
        st.divider()
        st.markdown("### [ SIMILAR THREAT SIGNATURES ]")
        st.caption('Retrieved from threat database via ChromaDB vector search')
        for i, case in enumerate(similar, 1):
            with st.expander(f"CASE #{i} · {case['title']}  [ similarity: {case['similarity']:.0%} ]"):
                c1, c2, c3 = st.columns(3)
                c1.metric('Employment Type', case['employment_type'] or 'Unknown')
                c2.metric('Salary Range',    case['salary_range']    or 'Not listed')
                c3.metric('Has Logo',        'Yes' if case['has_company_logo'] else 'No')
                st.caption(case['snippet'])

    # ── Raw output ────────────────────────────────────────────────────────────
    with st.expander('[ RAW PIPELINE OUTPUT ]'):
        display = {k: v for k, v in result.items() if k != 'similar_cases'}
        st.json(display)

    # ── Actions ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### [ ACTIONS ]")
    st.markdown(
        "<div style='background:#0d1117;border:1px solid #00ff9d33;border-radius:6px;"
        "padding:12px 16px;font-family:Courier New,monospace;font-size:0.82rem;color:#8b949e;"
        "margin-bottom:12px;'>"
        "💡 <span style='color:#c9d1d9;'>Help us improve Guardian Recruit!</span> "
        "Save this job listing to our training queue — whether it's fraud or legitimate, "
        "every submission helps the model get smarter over time. Thank you!"
        "</div>",
        unsafe_allow_html=True
    )
    act1, act2 = st.columns(2)

    with act1:
        if st.session_state['saved']:
            st.success('✓ Saved — thank you for helping improve Guardian Recruit!')
        else:
            if st.button('💾  Save Listing to Help Improve the App', use_container_width=True,
                         help='Adds this posting to the retraining queue for future model improvement.'):
                save_to_training_queue(posting, result)
                st.session_state['saved'] = True
                st.rerun()

    with act2:
        if st.button('✕  Clear Results', use_container_width=True,
                     help='Clears the current result. The form also clears on page refresh.'):
            st.session_state['result']  = None
            st.session_state['posting'] = None
            st.session_state['saved']   = False
            st.rerun()
