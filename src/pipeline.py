"""
FraudDetectionPipeline — central orchestration class.

Loads all models once at startup and exposes a single predict() interface.
All UI and scripts should go through this class, not call stream modules directly.

Usage:
    from pipeline import FraudDetectionPipeline

    pipeline = FraudDetectionPipeline()
    pipeline.warm_up()          # optional — pre-loads all models
    result = pipeline.predict(posting_dict)
    health = pipeline.health_check()
"""

import sys
import os
import time
import pandas as pd

# Ensure src/ is on path when called from repo root
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import GuardianCleaner
import nlp_stream
import outlier_stream
import fusion_layer
from vector_store import get_collection, search_similar, collection_size
from explainer import explain
from text_signals import compute_keyword_score


class FraudDetectionPipeline:
    """
    End-to-end fraud detection pipeline.

    Components:
        1. GuardianCleaner   — normalises raw input fields
        2. nlp_stream        — BERT fraud probability (Stream A)
        3. outlier_stream    — IsolationForest anomaly score (Stream B)
        4. fusion_layer      — XGBoost meta-classifier (final score)
        5. vector_store      — ChromaDB similarity search (RAG context)
        6. explainer         — LangChain + Ollama plain-English explanation
    """

    def __init__(self):
        self._cleaner = GuardianCleaner()
        self._ready   = False

    def warm_up(self) -> dict:
        """
        Pre-load all models into memory.
        Call once at app startup to avoid cold-start latency on first prediction.

        Returns:
            health_check() result after warm-up.
        """
        print('Warming up Guardian Recruit pipeline...')
        t0 = time.time()

        # Trigger lazy loaders
        nlp_stream._load_model()
        outlier_stream.load_model()

        # Warm up fusion layer
        fusion_layer._load_model()

        # Build / load ChromaDB index
        get_collection()

        self._ready = True
        elapsed = time.time() - t0
        print(f'Pipeline ready in {elapsed:.1f}s')
        return self.health_check()

    def predict(self, posting: dict) -> dict:
        """
        Score a single raw job posting end-to-end.

        Args:
            posting: Raw dict with any subset of job posting fields.

        Returns:
            {
                "fraud_score":    float  — final probability 0.0–1.0
                "label":          str    — "FRAUD" | "LEGITIMATE"
                "threshold":      float  — decision boundary
                "bert_score":     float  — Stream A score
                "outlier_score":  float  — Stream B score
                "features":       dict   — meta-features used by XGBoost
                "explanation":    str    — plain-English reason
                "similar_cases":  list   — top-3 similar known fraud postings
                "explain_source": str    — "llm" | "template"
                "latency_ms":     float  — total inference time
            }
        """
        t0 = time.time()

        # 1. Clean input
        df      = pd.DataFrame([posting])
        cleaned = self._cleaner.transform(df, is_training=False)
        row     = cleaned.iloc[0]

        # 2. Build posting text (used by keyword scorer + ChromaDB + explainer)
        posting_text = ' '.join([
            str(posting.get('title', '')           or ''),
            str(posting.get('company_profile', '') or ''),
            str(posting.get('description', '')     or ''),
            str(posting.get('requirements', '')    or ''),
        ]).strip()

        # 3. Keyword signal score — catches modern fraud BERT misses
        keyword_score, triggered_signals = compute_keyword_score(posting_text)

        # 4. Score through fusion layer (Stream A + Stream B + XGBoost)
        score_result = fusion_layer.predict(row)

        # 5. Combine ML score with keyword score
        # Take the max — if either the ML model or the rules engine flags it, flag it.
        # This prevents sophisticated fraud from evading detection via polished language.
        ml_score    = score_result['fraud_score']
        final_score = max(ml_score, keyword_score)
        final_label = 'FRAUD' if final_score >= score_result['threshold'] else 'LEGITIMATE'

        score_result['fraud_score']      = round(final_score, 4)
        score_result['label']            = final_label
        score_result['ml_score']         = round(ml_score, 4)
        score_result['keyword_score']    = keyword_score
        score_result['triggered_signals'] = triggered_signals

        # 6. Explain — similar cases + LLM/template explanation
        explain_result = explain(score_result, posting_text)

        return {
            **score_result,
            'explanation':    explain_result['explanation'],
            'similar_cases':  explain_result['similar_cases'],
            'explain_source': explain_result['source'],
            'latency_ms':     round((time.time() - t0) * 1000, 1),
        }

    def predict_batch(self, df: pd.DataFrame, include_explanation: bool = False) -> pd.DataFrame:
        """
        Score a DataFrame of job postings.
        Explanations are skipped by default (slow for large batches).

        Returns:
            Original DataFrame with columns appended:
            fraud_score, label, bert_score, outlier_score
        """
        results = []
        for _, row in df.iterrows():
            posting = row.to_dict()
            if include_explanation:
                r = self.predict(posting)
            else:
                # Fast path — skip ChromaDB + explainer
                cleaned = self._cleaner.transform(pd.DataFrame([posting]), is_training=False)
                r = fusion_layer.predict(cleaned.iloc[0])
            results.append(r)

        out = df.copy()
        out['fraud_score']   = [r['fraud_score']   for r in results]
        out['label']         = [r['label']          for r in results]
        out['bert_score']    = [r['bert_score']     for r in results]
        out['outlier_score'] = [r['outlier_score']  for r in results]
        return out

    def health_check(self) -> dict:
        """
        Verify all pipeline components are loaded and functional.

        Returns:
            Dict with status per component and overall ok flag.
        """
        checks = {}

        # BERT
        nlp_stream._load_model()
        checks['bert'] = {
            'ok':    nlp_stream._model is not None,
            'note':  'trained' if (
                os.path.exists(nlp_stream.MODEL_PATH) and
                os.path.getsize(nlp_stream.MODEL_PATH) > 0
            ) else 'untrained weights',
        }

        # IsolationForest
        checks['isolation_forest'] = {
            'ok':   outlier_stream.model is not None,
            'note': 'loaded' if outlier_stream.model else 'model file missing',
        }

        # Fusion XGBoost
        fusion_layer._load_model()
        checks['fusion_xgb'] = {
            'ok':   fusion_layer._fusion_model is not None,
            'note': 'loaded' if fusion_layer._fusion_model else 'model file missing — fallback to BERT score',
        }

        # ChromaDB
        try:
            n = collection_size()
            checks['chroma_db'] = {'ok': n > 0, 'note': f'{n:,} fraud postings indexed'}
        except Exception as e:
            checks['chroma_db'] = {'ok': False, 'note': str(e)}

        checks['overall_ok'] = all(v['ok'] for v in checks.values() if isinstance(v, dict))
        return checks


# Module-level singleton — imported by app.py
pipeline = FraudDetectionPipeline()
