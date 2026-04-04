"""
Vector Store — ChromaDB + Sentence Transformers
Indexes known fraud postings for RAG-based similarity search.

On first run, embeds all fraud postings from train_clean_v1.csv and
persists to data/chroma_db/. Subsequent runs load the index instantly.
"""

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH         = os.path.join(REPO_ROOT, 'data', 'chroma_db')
TRAIN_PATH      = os.path.join(REPO_ROOT, 'data', 'processed', 'train.csv')
COLLECTION_NAME = 'fraud_postings'
EMBED_MODEL     = 'all-MiniLM-L6-v2'
BATCH_SIZE      = 100   # ChromaDB safe batch size

_collection = None


def _embed_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _make_text(row) -> str:
    parts = [
        str(row.get('title', '')           or ''),
        str(row.get('company_profile', '') or ''),
        str(row.get('description', '')     or ''),
        str(row.get('requirements', '')    or ''),
    ]
    return ' '.join(p for p in parts if p.strip()).strip()


def _build_index(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Index all fraud postings. Called once on first run."""
    print('Building ChromaDB fraud index (one-time setup)...')

    df       = pd.read_csv(TRAIN_PATH)
    fraud_df = df[df['fraudulent'] == 1].reset_index(drop=True)

    texts, ids, metadatas = [], [], []
    for i, row in fraud_df.iterrows():
        text = _make_text(row)
        if not text.strip():
            continue
        texts.append(text)
        ids.append(str(i))
        metadatas.append({
            'title':            str(row.get('title', '')           or '')[:200],
            'employment_type':  str(row.get('employment_type', '') or ''),
            'salary_range':     str(row.get('salary_range', '')    or ''),
            'has_company_logo': int(row.get('has_company_logo', 0) or 0),
            'has_questions':    int(row.get('has_questions', 0)    or 0),
        })

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn(),
        metadata={'hnsw:space': 'cosine'},
    )

    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            documents=texts[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

    print(f'ChromaDB index ready: {len(texts):,} fraud postings indexed.')
    return collection


def get_collection() -> chromadb.Collection:
    """Return the fraud postings collection, building it on first call."""
    global _collection
    if _collection is not None:
        return _collection

    os.makedirs(DB_PATH, exist_ok=True)
    client   = chromadb.PersistentClient(path=DB_PATH)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_embed_fn(),
        )
    else:
        _collection = _build_index(client)

    return _collection


def search_similar(text: str, n: int = 3) -> list[dict]:
    """
    Return the n most similar known fraud postings to the input text.

    Returns:
        List of dicts: title, employment_type, salary_range,
        has_company_logo, has_questions, similarity (0–1), snippet.
    """
    if not text.strip():
        return []

    collection = get_collection()
    results    = collection.query(query_texts=[text], n_results=min(n, 3))

    similar = []
    for i in range(len(results['ids'][0])):
        meta     = results['metadatas'][0][i]
        doc      = results['documents'][0][i]
        distance = results['distances'][0][i] if results.get('distances') else 0.0
        similar.append({
            'title':            meta.get('title', 'Unknown'),
            'employment_type':  meta.get('employment_type', ''),
            'salary_range':     meta.get('salary_range', ''),
            'has_company_logo': meta.get('has_company_logo', 0),
            'has_questions':    meta.get('has_questions', 0),
            'similarity':       round(max(0.0, 1.0 - distance), 3),
            'snippet':          doc[:300] + '...' if len(doc) > 300 else doc,
        })

    return similar


def collection_size() -> int:
    """Return number of indexed fraud postings."""
    try:
        return get_collection().count()
    except Exception:
        return 0
