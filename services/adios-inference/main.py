"""adios-inference — Unified Indic AI inference gateway.

Routes:
  POST /translate          IndicTrans2 (22 Indian languages, MIT)
  POST /embed              IndicBERT sentence embeddings
  POST /generate           Ollama proxy (Param-1, gajendra, qwen3.6…)
  POST /asr                Indic-Conformer / IndicWav2Vec ASR (stub)
  GET  /models             Registry of all loaded backends
  GET  /health             Liveness

Designed as the inference substrate for AdiOS Platform's
adios-inference component (plugins/ai/). The hackathon wires
into this via HTTP from the Rust axum service on port 8000.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("adios-inference")

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR        = os.environ.get("ADIOS_MODEL_DIR",   os.path.expanduser("~/.adios/models/hf"))
OLLAMA_URL       = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434")
DEFAULT_CHAT     = os.environ.get("DEFAULT_CHAT_MODEL","gajendra:latest")
DEVICE           = os.environ.get("INFERENCE_DEVICE",  "cuda")   # cuda | cpu
TRANSLATE_DEVICE = os.environ.get("TRANSLATE_DEVICE",  "cuda")

# ── Lazy-loaded backends ─────────────────────────────────────────────────────

_translate_pipe  = None   # IndicTrans2 pipeline
_bert_model      = None   # IndicBERT
_bert_tokenizer  = None

def load_translation_pipeline():
    global _translate_pipe
    if _translate_pipe is not None:
        return _translate_pipe
    en_indic_dir = os.path.join(MODEL_DIR, "indictrans2-en-indic-dist-200M")
    indic_en_dir = os.path.join(MODEL_DIR, "indictrans2-indic-en-dist-200M")
    if not os.path.isdir(en_indic_dir) or not os.path.isdir(indic_en_dir):
        log.warning("IndicTrans2 models not found at %s — translation unavailable", MODEL_DIR)
        return None
    try:
        from IndicTransToolkit import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _translate_pipe = {
            "en_indic": {
                "model":     AutoModelForSeq2SeqLM.from_pretrained(en_indic_dir).to(TRANSLATE_DEVICE),
                "tokenizer": AutoTokenizer.from_pretrained(en_indic_dir, trust_remote_code=True),
                "processor": IndicProcessor(inference=True),
            },
            "indic_en": {
                "model":     AutoModelForSeq2SeqLM.from_pretrained(indic_en_dir).to(TRANSLATE_DEVICE),
                "tokenizer": AutoTokenizer.from_pretrained(indic_en_dir, trust_remote_code=True),
                "processor": IndicProcessor(inference=True),
            },
        }
        log.info("IndicTrans2 loaded (%s)", TRANSLATE_DEVICE)
    except ImportError:
        log.warning("IndicTransToolkit not installed — run: pip install indic-trans")
        _translate_pipe = None
    return _translate_pipe

def load_indicbert():
    global _bert_model, _bert_tokenizer
    if _bert_tokenizer is not None:
        return _bert_model, _bert_tokenizer
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        log.info("Loading IndicBERT…")
        _bert_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        _bert_model     = AutoModel.from_pretrained("ai4bharat/indic-bert").to(DEVICE)
        _bert_model.eval()
        log.info("IndicBERT loaded")
    except Exception as e:
        log.warning("IndicBERT unavailable: %s", e)
    return _bert_model, _bert_tokenizer

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("adios-inference starting — MODEL_DIR=%s", MODEL_DIR)
    # Warm up on startup (non-blocking; backends lazy-load on first request
    # to keep startup fast when models are not yet downloaded)
    yield
    log.info("adios-inference shutting down")

app = FastAPI(title="adios-inference", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text:            str
    source_lang:     str = "eng_Latn"   # BCP-47 + script, e.g. hin_Deva, tam_Taml
    target_lang:     str = "hin_Deva"
    max_length:      int = 512

class TranslateResponse(BaseModel):
    translated:      str
    source_lang:     str
    target_lang:     str
    model:           str
    backend:         str

class EmbedRequest(BaseModel):
    text:            str
    model:           str = "indicbert"  # indicbert | nomic (via Ollama)

class EmbedResponse(BaseModel):
    embedding:       list[float]
    model:           str
    dimensions:      int

class GenerateRequest(BaseModel):
    prompt:          str
    system:          Optional[str] = None
    model:           str = ""           # empty = use DEFAULT_CHAT
    max_tokens:      int = 1024
    temperature:     float = 0.1

class GenerateResponse(BaseModel):
    content:         str
    model:           str
    backend:         str

class ModelInfo(BaseModel):
    name:            str
    backend:         str
    available:       bool
    notes:           str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "adios-inference"}

@app.get("/models", response_model=list[ModelInfo])
def list_models():
    pipe = _translate_pipe  # check cached state only
    bert_loaded = _bert_tokenizer is not None
    indic_dir = os.path.join(MODEL_DIR, "indictrans2-en-indic-dist-200M")
    param1_dir = os.path.join(MODEL_DIR, "param-1-2.9b-instruct")
    return [
        ModelInfo(name="indictrans2-en-indic-dist-200M", backend="huggingface",
                  available=os.path.isdir(indic_dir) or pipe is not None,
                  notes="IndicTrans2 En→Indic distilled 200M (MIT)"),
        ModelInfo(name="indictrans2-indic-en-dist-200M", backend="huggingface",
                  available=os.path.isdir(indic_dir) or pipe is not None,
                  notes="IndicTrans2 Indic→En distilled 200M (MIT)"),
        ModelInfo(name="indicbert", backend="huggingface",
                  available=bert_loaded,
                  notes="IndicBERT ALBERT-base, 12 Indian languages, embeddings/NER"),
        ModelInfo(name="param-1-2.9b-instruct", backend="huggingface",
                  available=os.path.isdir(param1_dir),
                  notes="BharatGen Param-1 2.9B bilingual Hindi+English"),
        ModelInfo(name="gajendra:latest",   backend="ollama", available=True,
                  notes="7B bilingual generalist, Indian regulatory"),
        ModelInfo(name="sarvam:latest",     backend="ollama", available=True,
                  notes="2B edge, fast classification, Indic scripts"),
        ModelInfo(name="ayurparam:latest",  backend="ollama", available=True,
                  notes="2.9B clinical Ayurveda, Hindi/Sanskrit"),
        ModelInfo(name="deepseek-r1:7b",    backend="ollama", available=True,
                  notes="7B logic/reasoning, reliable JSON output"),
        ModelInfo(name="qwen3.6:latest",    backend="ollama", available=True,
                  notes="36B long-form summarisation and reports"),
        ModelInfo(name="nomic-embed-text",  backend="ollama", available=True,
                  notes="Semantic embeddings for document comparison"),
    ]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    pipe = load_translation_pipeline()
    if pipe is None:
        raise HTTPException(503, "IndicTrans2 not loaded — download models first: see README")
    import torch
    direction = "en_indic" if req.source_lang.startswith("eng") else "indic_en"
    backend   = pipe[direction]
    processor = backend["processor"]
    tokenizer = backend["tokenizer"]
    model     = backend["model"]

    batch = processor.preprocess_batch([req.text], src_lang=req.source_lang, tgt_lang=req.target_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest",
                       return_tensors="pt", return_attention_mask=True).to(TRANSLATE_DEVICE)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=req.max_length,
        )
    decoded = tokenizer.batch_decode(generated.detach().cpu().tolist(),
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    result = processor.postprocess_batch(decoded, lang=req.target_lang)
    return TranslateResponse(
        translated=result[0],
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        model="indictrans2-dist-200M",
        backend="huggingface",
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if req.model == "nomic" or req.model.startswith("nomic"):
        # Proxy to Ollama nomic-embed-text
        import httpx
        r = httpx.post(f"{OLLAMA_URL}/api/embed",
                       json={"model": "nomic-embed-text:latest", "input": req.text},
                       timeout=30)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        return EmbedResponse(embedding=vec, model="nomic-embed-text", dimensions=len(vec))

    bert, tok = load_indicbert()
    if bert is None:
        raise HTTPException(503, "IndicBERT not loaded")
    import torch
    inputs = tok(req.text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = bert(**inputs)
    vec = out.last_hidden_state[:, 0, :].squeeze().cpu().tolist()
    return EmbedResponse(embedding=vec, model="indicbert", dimensions=len(vec))

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model  = req.model or DEFAULT_CHAT
    msgs   = []
    if req.system:
        msgs.append({"role": "system",    "content": req.system})
    msgs.append({"role": "user",       "content": req.prompt})

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json={
            "model":    model,
            "stream":   False,
            "messages": msgs,
            "options":  {"temperature": req.temperature, "num_predict": req.max_tokens},
        })
    r.raise_for_status()
    content = r.json()["message"]["content"]
    # Strip reasoning traces from thinking models
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return GenerateResponse(content=content, model=model, backend="ollama")
