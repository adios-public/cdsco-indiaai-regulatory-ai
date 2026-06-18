"""Download required Indic AI models to ~/.adios/models/hf/

Run once:
    python3 services/adios-inference/download_models.py

Models downloaded:
    - IndicTrans2 En-Indic distilled 200M (MIT)  ~400MB
    - IndicTrans2 Indic-En distilled 200M (MIT)  ~400MB
    - BharatGen Param-1 2.9B-Instruct           ~5.8GB
    - IndicBERT (ai4bharat/indic-bert)           ~50MB
"""
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed.")
    print("Run: pip install huggingface_hub")
    sys.exit(1)

BASE = os.path.expanduser("~/.adios/models/hf")
os.makedirs(BASE, exist_ok=True)

MODELS = [
    # (repo_id, local_dir_name, description)
    ("ai4bharat/indictrans2-en-indic-dist-200M",
     "indictrans2-en-indic-dist-200M",
     "IndicTrans2 En→Indic distilled 200M — MIT license"),
    ("ai4bharat/indictrans2-indic-en-dist-200M",
     "indictrans2-indic-en-dist-200M",
     "IndicTrans2 Indic→En distilled 200M — MIT license"),
    ("ai4bharat/indic-bert",
     "indic-bert",
     "IndicBERT ALBERT-base — 12 Indian languages, embeddings/NER"),
    ("bharatgenai/Param-1-2.9B-Instruct",
     "param-1-2.9b-instruct",
     "BharatGen Param-1 2.9B — Hindi+English instruction model"),
]

def download(repo_id: str, local_name: str, desc: str):
    dest = os.path.join(BASE, local_name)
    if os.path.isdir(dest) and any(f.endswith(".bin") or f.endswith(".safetensors")
                                   for f in os.listdir(dest)):
        print(f"  SKIP  {local_name} (already present at {dest})")
        return
    print(f"  GET   {local_name}")
    print(f"        {desc}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest,
                          ignore_patterns=["*.msgpack", "*.h5", "flax_model*",
                                           "tf_model*", "rust_model*"])
        print(f"  DONE  {local_name} → {dest}")
    except Exception as e:
        print(f"  FAIL  {local_name}: {e}")

if __name__ == "__main__":
    print(f"Downloading Indic AI models to {BASE}\n")
    for repo, name, desc in MODELS:
        download(repo, name, desc)
    print("\nAll done. Start adios-inference:")
    print("  uvicorn services.adios-inference.main:app --port 8010 --reload")
