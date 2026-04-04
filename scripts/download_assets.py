"""
Download models and data assets for Guardian Recruit.

Run once before starting the app:
    python scripts/download_assets.py

Deployment tiers
----------------
LITE  (default) — fusion_xgb.json + outlier_forest.pkl from Hugging Face Hub
                  ~1.5 MB total, works on Streamlit Cloud free tier
                  bert_score falls back to keyword signal score

FULL            — also downloads nlp_bert.pth (418 MB) from Google Drive
                  requires --full flag, recommended for local dev / retraining

HOW TO SET UP HUGGING FACE
---------------------------
1. Create a free account at https://huggingface.co
2. Create a new Model repository (e.g. your-username/guardian-recruit-models)
3. Upload fusion_xgb.json and outlier_forest.pkl to that repo
4. Set HF_REPO below to "your-username/guardian-recruit-models"
"""

import os
import sys
import argparse

try:
    import gdown
except ImportError:
    print("[ERROR] gdown not installed. Run: pip install gdown>=5.0.0")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Hugging Face repo (update with your HF username/repo) ────────────────────
HF_REPO = "ijih14/guardian-recruit-models"

HF_MODELS = {
    "fusion_xgb.json":    "models/fusion_xgb.json",
    "outlier_forest.pkl": "models/outlier_forest.pkl",
    "nlp_bert.pth":       "models/nlp_bert.pth",   # 418 MB — uploaded via git LFS
}

# ── Google Drive folder — data/processed/ CSVs ───────────────────────────────
DATA_FOLDER_ID = "1jHSmyPs9_1Z8JgiNfZCjJ_BhSnar5rAZ"
DATA_DEST      = os.path.join(REPO_ROOT, "data", "processed")

# ─────────────────────────────────────────────────────────────────────────────


def download_hf_models(models: dict = None):
    if HF_REPO.startswith("REPLACE_WITH"):
        print("[WARNING] HF_REPO not set — skipping Hugging Face model download.")
        print("  Edit scripts/download_assets.py and set HF_REPO to your repo name.\n")
        return

    models = models or HF_MODELS
    print("── Downloading models from Hugging Face Hub ──")
    for filename, relative_dest in models.items():
        dest_path = os.path.join(REPO_ROOT, relative_dest)
        if os.path.exists(dest_path):
            print(f"  [SKIP]  {relative_dest}")
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"  [DOWN]  {relative_dest} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=os.path.dirname(dest_path),
            )
            print(f"  [OK]    {relative_dest}")
        except Exception as e:
            print(f"  [FAIL]  {relative_dest} — {e}")


def download_gdrive_file(file_id: str, relative_dest: str):
    dest_path = os.path.join(REPO_ROOT, relative_dest)
    if os.path.exists(dest_path):
        print(f"  [SKIP]  {relative_dest}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  [DOWN]  {relative_dest} ...")
    try:
        gdown.download(url, dest_path, quiet=False, fuzzy=True)
        print(f"  [OK]    {relative_dest}")
    except Exception as e:
        print(f"  [FAIL]  {relative_dest} — {e}")


def download_data_folder():
    import shutil, tempfile
    print("\n── Downloading data/processed/ folder ──")

    existing = [f for f in os.listdir(DATA_DEST) if f.endswith('.csv')] \
               if os.path.exists(DATA_DEST) else []
    if existing:
        print(f"  [SKIP]  data/processed/ already has {len(existing)} CSV file(s)")
        return

    os.makedirs(DATA_DEST, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{DATA_FOLDER_ID}"
    print(f"  [DOWN]  Downloading folder to data/processed/ ...")
    try:
        with tempfile.TemporaryDirectory() as staging:
            gdown.download_folder(url, output=staging, quiet=False, use_cookies=False)
            subfolders = [f for f in os.listdir(staging)
                          if os.path.isdir(os.path.join(staging, f))]
            src_dir = os.path.join(staging, subfolders[0]) if subfolders else staging
            moved = 0
            for fname in os.listdir(src_dir):
                src  = os.path.join(src_dir, fname)
                dest = os.path.join(DATA_DEST, fname)
                if os.path.isfile(src):
                    shutil.move(src, dest)
                    print(f"  [OK]    data/processed/{fname}")
                    moved += 1
            if moved == 0:
                print("  [WARN]  No files found in the Drive folder.")
    except Exception as e:
        print(f"  [FAIL]  data/processed/ — {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Guardian Recruit assets.")
    parser.add_argument("--full", action="store_true",
                        help="Also download nlp_bert.pth (418 MB) for full BERT mode")
    parser.add_argument("--data", action="store_true",
                        help="Download data/processed/ CSVs from Google Drive")
    args = parser.parse_args()

    if args.full:
        download_hf_models()
    else:
        # Lite mode — skip nlp_bert.pth (418 MB), download only the two small models
        _lite_models = {k: v for k, v in HF_MODELS.items() if k != "nlp_bert.pth"}
        download_hf_models(_lite_models)
        print("\n[ LITE MODE ] Skipping nlp_bert.pth (418 MB).")
        print("  bert_score will fall back to keyword signal score.")
        print("  Run with --full to download BERT weights for higher accuracy.\n")

    if args.data:
        download_data_folder()

    print("\n[DONE] Assets ready.")
    if not args.full:
        print("  Tip: run with --full to enable BERT-powered NLP stream.")


if __name__ == "__main__":
    main()
