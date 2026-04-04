"""
Startup asset downloader — runs automatically when the app cold-starts.
Downloads models from Hugging Face Hub and data from Google Drive if not present.
Safe to run multiple times — skips files that already exist.
"""

import os
import sys

REPO_ROOT     = os.path.dirname(__file__)
HF_REPO       = "ijah14/guardian-recruit-models"
DATA_FOLDER_ID = "1jHSmyPs9_1Z8JgiNfZCjJ_BhSnar5rAZ"

HF_MODELS = {
    "fusion_xgb.json":    os.path.join(REPO_ROOT, "models", "fusion_xgb.json"),
    "outlier_forest.pkl": os.path.join(REPO_ROOT, "models", "outlier_forest.pkl"),
    "nlp_bert.pth":       os.path.join(REPO_ROOT, "models", "nlp_bert.pth"),
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
    existing = [f for f in os.listdir(DATA_DEST) if f.endswith('.csv')] \
               if os.path.exists(DATA_DEST) else []
    if existing:
        print(f"[startup] SKIP  data/processed/ ({len(existing)} CSV files present)")
        return

    try:
        import gdown, shutil, tempfile
    except ImportError:
        print("[startup] gdown not installed — skipping data download.")
        return

    os.makedirs(DATA_DEST, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{DATA_FOLDER_ID}"
    print("[startup] DOWN  data/processed/ from Google Drive ...")
    try:
        with tempfile.TemporaryDirectory() as staging:
            gdown.download_folder(url, output=staging, quiet=True, use_cookies=False)
            subfolders = [f for f in os.listdir(staging)
                          if os.path.isdir(os.path.join(staging, f))]
            src_dir = os.path.join(staging, subfolders[0]) if subfolders else staging
            for fname in os.listdir(src_dir):
                src = os.path.join(src_dir, fname)
                if os.path.isfile(src):
                    shutil.move(src, os.path.join(DATA_DEST, fname))
                    print(f"[startup] OK    data/processed/{fname}")
    except Exception as e:
        print(f"[startup] FAIL  data/processed/ — {e}")


def ensure_assets():
    """Download all required assets if not already present. Call once at app startup."""
    print("[startup] Checking assets ...")
    _download_models()
    _download_data()
    print("[startup] Assets ready.")
