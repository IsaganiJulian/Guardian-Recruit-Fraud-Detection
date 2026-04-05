"""
Startup asset downloader — runs automatically when the app cold-starts.
Downloads models from Hugging Face Hub and data from Google Drive if not present.
Safe to run multiple times — skips files that already exist.
"""

import os
import sys

REPO_ROOT     = os.path.dirname(__file__)
HF_REPO       = "ijih14/guardian-recruit-models"
DATA_FOLDER_ID = "1jHSmyPs9_1Z8JgiNfZCjJ_BhSnar5rAZ"

HF_MODELS = {
    "fusion_xgb.json":    os.path.join(REPO_ROOT, "models", "fusion_xgb.json"),
    "outlier_forest.pkl": os.path.join(REPO_ROOT, "models", "outlier_forest.pkl"),
    "nlp_bert.pth":       os.path.join(REPO_ROOT, "models", "nlp_bert.pth"),
}

# train.csv is required by ChromaDB vector store at startup — host on HF Hub
# Upload with: hf upload ijih14/guardian-recruit-models data/processed/train.csv train.csv
HF_DATA = {
    "train.csv": os.path.join(REPO_ROOT, "data", "processed", "train.csv"),
}

DATA_DEST = os.path.join(REPO_ROOT, "data", "processed")


def _download_models():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[startup] huggingface_hub not installed — skipping model download.")
        return

    os.makedirs(os.path.join(REPO_ROOT, "models"), exist_ok=True)

    for filename, dest_path in HF_MODELS.items():
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[startup] SKIP  {filename}")
            continue
        print(f"[startup] DOWN  {filename} from HF Hub ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=os.path.join(REPO_ROOT, "models"),
            )
            print(f"[startup] OK    {filename}")
        except Exception as e:
            print(f"[startup] FAIL  {filename} — {e}")


def _download_data():
    os.makedirs(DATA_DEST, exist_ok=True)

    # Step 1 — Download train.csv from HF Hub (required for ChromaDB at startup)
    try:
        from huggingface_hub import hf_hub_download
        for filename, dest_path in HF_DATA.items():
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                print(f"[startup] SKIP  {filename}")
                continue
            print(f"[startup] DOWN  {filename} from HF Hub ...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=DATA_DEST,
            )
            print(f"[startup] OK    {filename}")
    except Exception as e:
        print(f"[startup] FAIL  train.csv from HF Hub — {e}")

    # Step 2 — Download remaining CSVs from Google Drive
    existing = [f for f in os.listdir(DATA_DEST) if f.endswith('.csv')]
    remaining = [f for f in ['val.csv', 'test.csv', 'FINAL_AUGMENTED_TRAINING.csv']
                 if f not in existing]
    if not remaining:
        print("[startup] SKIP  remaining CSVs already present")
        return

    try:
        import gdown, shutil, tempfile
    except ImportError:
        print("[startup] gdown not installed — skipping Google Drive download.")
        return

    url = f"https://drive.google.com/drive/folders/{DATA_FOLDER_ID}"
    print("[startup] DOWN  remaining CSVs from Google Drive ...")
    try:
        with tempfile.TemporaryDirectory() as staging:
            gdown.download_folder(url, output=staging, quiet=True, use_cookies=False)
            subfolders = [f for f in os.listdir(staging)
                          if os.path.isdir(os.path.join(staging, f))]
            src_dir = os.path.join(staging, subfolders[0]) if subfolders else staging
            for fname in os.listdir(src_dir):
                src = os.path.join(src_dir, fname)
                dest = os.path.join(DATA_DEST, fname)
                if os.path.isfile(src) and not os.path.exists(dest):
                    shutil.move(src, dest)
                    print(f"[startup] OK    data/processed/{fname}")
    except Exception as e:
        print(f"[startup] FAIL  Google Drive download — {e}")


def _download_bert_background():
    """Download nlp_bert.pth in a background thread — app starts in lite mode until done."""
    import threading
    bert_path = HF_MODELS.get("nlp_bert.pth")
    if bert_path and os.path.exists(bert_path) and os.path.getsize(bert_path) > 1_000_000:
        return

    def _bg():
        print("[startup] BG    nlp_bert.pth downloading in background ...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=HF_REPO,
                filename="nlp_bert.pth",
                local_dir=os.path.join(REPO_ROOT, "models"),
            )
            print("[startup] BG    nlp_bert.pth ready.")
        except Exception as e:
            print(f"[startup] BG    nlp_bert.pth failed — {e}")

    threading.Thread(target=_bg, daemon=True).start()


def ensure_assets():
    """Download required assets at startup. BERT downloads in background to avoid timeout."""
    print("[startup] Checking assets ...")

    # Small models only — synchronous (fast)
    small = {k: v for k, v in HF_MODELS.items() if k != "nlp_bert.pth"}
    for filename, dest_path in small.items():
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[startup] SKIP  {filename}")
            continue
        print(f"[startup] DOWN  {filename} ...")
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            hf_hub_download(repo_id=HF_REPO, filename=filename,
                            local_dir=os.path.join(REPO_ROOT, "models"))
            print(f"[startup] OK    {filename}")
        except Exception as e:
            print(f"[startup] FAIL  {filename} — {e}")

    # train.csv — required for ChromaDB
    _download_data()

    # BERT — background so app starts immediately in lite mode
    _download_bert_background()

    print("[startup] Assets ready. BERT loading in background — lite mode active.")
