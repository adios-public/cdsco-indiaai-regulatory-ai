"""Download required Indic AI models to ~/.adios/models/hf/

Public models (no HF account needed):
    - Meta NLLB-200-distilled-600M  (Apache 2.0)  ~2.4GB  ← primary translation
    - AI4Bharat IndicBERTv2-MLM-only (Apache 2.0)  ~420MB  ← Indic embeddings
    - BharatGen Param-1-2.9B-Instruct (check card) ~5.8GB  ← Hindi+En instruction

Gated models (request access at huggingface.co first):
    - ai4bharat/indictrans2-en-indic-dist-200M  (MIT, gated)
    - ai4bharat/indictrans2-indic-en-dist-200M  (MIT, gated)

Once IndicTrans2 access is approved, re-run this script — the gateway
automatically upgrades from NLLB-200 to IndicTrans2 when those dirs are present.
"""
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: pip install huggingface_hub")
    sys.exit(1)

BASE = os.path.expanduser("~/.adios/models/hf")
os.makedirs(BASE, exist_ok=True)

PUBLIC = [
    ("facebook/nllb-200-distilled-600M",    "nllb-200-distilled-600M",
     "Meta NLLB-200 600M — 200+ langs — Apache 2.0"),
    ("ai4bharat/IndicBERTv2-MLM-only",      "indicbertv2-mlm-only",
     "IndicBERTv2 MLM-only — 23 Indian languages, embeddings/NER"),
    ("bharatgenai/Param-1-2.9B-Instruct",   "param-1-2.9b-instruct",
     "BharatGen Param-1 2.9B — Hindi+English instruction"),
]

GATED = [
    ("ai4bharat/indictrans2-en-indic-dist-200M", "indictrans2-en-indic-dist-200M",
     "IndicTrans2 En→Indic 200M — MIT (request access first)"),
    ("ai4bharat/indictrans2-indic-en-dist-200M", "indictrans2-indic-en-dist-200M",
     "IndicTrans2 Indic→En 200M — MIT (request access first)"),
]

SKIP = ["*.msgpack","*.h5","flax_*","tf_*","rust_*"]

def pull(repo, name, desc):
    dest = os.path.join(BASE, name)
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"  SKIP  {name}")
        return True
    print(f"  GET   {name}  ({desc})")
    try:
        snapshot_download(repo_id=repo, local_dir=dest, ignore_patterns=SKIP)
        print(f"  DONE  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False

if __name__ == "__main__":
    print(f"Model directory: {BASE}\n")
    print("=== Public models ===")
    for r, n, d in PUBLIC:
        pull(r, n, d)
    print("\n=== Gated models (requires HF access approval) ===")
    for r, n, d in GATED:
        pull(r, n, d)
    print("\nDone. Start gateway:")
    print("  cd ~/.adios/inference && uvicorn main:app --host 0.0.0.0 --port 8010")
