# adios-inference

Unified Indic AI inference gateway for the AdiOS Platform.

**Port:** `8010`  
**Backends:** HuggingFace (IndicTrans2, IndicBERT, Param-1) + Ollama proxy

## Setup

```bash
# 1. Create venv
python3 -m venv ~/.adios/venv/inference
source ~/.adios/venv/inference/bin/activate

# 2. Install dependencies
pip install -r services/adios-inference/requirements.txt

# 3. Download models (~7GB total)
python3 services/adios-inference/download_models.py

# 4. Start gateway
uvicorn services.adios-inference.main:app --host 0.0.0.0 --port 8010
```

## API

### POST /translate
```json
{ "text": "Patient complaint in Hindi", "source_lang": "hin_Deva", "target_lang": "eng_Latn" }
```
Backed by IndicTrans2 distilled 200M. MIT licensed. Supports all 22 Indian scheduled languages.

**Language codes:** `eng_Latn` · `hin_Deva` · `tam_Taml` · `tel_Telu` · `kan_Knda` · `mal_Mlym` · `mar_Deva` · `ben_Beng` · `guj_Gujr` · `pan_Guru` · `urd_Arab` · `asm_Beng` · `ory_Orya` · `san_Deva`

### POST /embed
```json
{ "text": "...", "model": "indicbert" }
```
Returns 768-dim IndicBERT sentence embedding. Use `"model": "nomic"` to proxy Ollama's nomic-embed-text.

### POST /generate
```json
{ "prompt": "...", "system": "...", "model": "gajendra:latest", "max_tokens": 512 }
```
Proxies Ollama. Strips `<think>` traces from reasoning models.

### GET /models
Returns live availability of all registered backends.

## Model registry

| Model | Backend | Size | Use case |
|-------|---------|------|----------|
| indictrans2-en-indic-dist-200M | HuggingFace | ~400MB | En → 22 Indic (MIT) |
| indictrans2-indic-en-dist-200M | HuggingFace | ~400MB | 22 Indic → En (MIT) |
| indicbert | HuggingFace | ~50MB | Embeddings / NER / classification |
| param-1-2.9b-instruct | HuggingFace | ~5.8GB | Hindi+English instruction |
| gajendra:latest | Ollama | 4.2GB | Bilingual generalist |
| sarvam:latest | Ollama | 1.5GB | Fast Indic edge |
| ayurparam:latest | Ollama | 1.8GB | Clinical Ayurveda |
| deepseek-r1:7b | Ollama | 4.7GB | Structured JSON reasoning |
| qwen3.6:latest | Ollama | 24GB | Long-form reports |
| nomic-embed-text | Ollama | 274MB | Semantic document embeddings |

## adios-platform integration

This service is designed to become `plugins/ai/adios-inference` in the AdiOS Platform.
The Rust client in `src/inference_client.rs` mirrors the same API contract so the
hackathon code lifts directly into `core/adios-cortex` or `plugins/ai/`.
