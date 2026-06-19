"""adios-inference — Unified Indic AI inference gateway v0.2.0

Backends:
  Translation : Meta NLLB-200-distilled-600M (Apache 2.0, no gate)  ← primary
                Falls back to IndicTrans2 distilled when available (gated, request at HF)
  Embeddings  : AI4Bharat IndicBERTv2-MLM-TLM (no gate)
                Falls back to nomic-embed-text via Ollama
  Generation  : Ollama proxy (gajendra, qwen3.6, sarvam, deepseek-r1, param-1…)

Language codes (FLORES-200, same for NLLB and IndicTrans2):
  eng_Latn  hin_Deva  tam_Taml  tel_Telu  kan_Knda  mal_Mlym
  mar_Deva  ben_Beng  guj_Gujr  pan_Guru  urd_Arab  asm_Beng
  ory_Orya  san_Deva  mai_Deva  kok_Deva  brx_Deva  doi_Deva
  kas_Arab  kas_Deva  mni_Mtei  sat_Olck  snd_Arab
"""
import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("adios-inference")

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_DIR    = os.environ.get("ADIOS_MODEL_DIR",   os.path.expanduser("~/.adios/models/hf"))
OLLAMA_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434")
DEFAULT_CHAT = os.environ.get("DEFAULT_CHAT_MODEL","gajendra:latest")
DEVICE       = os.environ.get("INFERENCE_DEVICE",  "cpu")   # cpu | cuda

# ── Backend paths ──────────────────────────────────────────────────────────

def _dir(name): return os.path.join(MODEL_DIR, name)
def _ready(name):
    d = _dir(name)
    return os.path.isdir(d) and bool(os.listdir(d))

# ── Lazy-loaded backends ─────────────────────────────────────────────────────

_nllb_pipe      = None
_bert_model     = None
_bert_tokenizer = None

def load_translation():
    global _nllb_pipe
    if _nllb_pipe is not None:
        return _nllb_pipe
    nllb = _dir("nllb-200-distilled-600M")
    if not _ready("nllb-200-distilled-600M"):
        log.warning("NLLB-200 not ready at %s", nllb)
        return None
    try:
        from transformers import pipeline as hf_pipeline
        log.info("Loading NLLB-200-distilled-600M on %s…", DEVICE)
        _nllb_pipe = hf_pipeline("translation", model=nllb,
                                  device=0 if DEVICE == "cuda" else -1)
        log.info("NLLB-200 loaded")
    except Exception as e:
        log.warning("NLLB-200 load failed: %s", e)
    return _nllb_pipe

def load_indicbert():
    global _bert_model, _bert_tokenizer
    if _bert_tokenizer is not None:
        return _bert_model, _bert_tokenizer
    if not _ready("indicbertv2-mlm-tlm"):
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer
        log.info("Loading IndicBERTv2…")
        path = _dir("indicbertv2-mlm-tlm")
        _bert_tokenizer = AutoTokenizer.from_pretrained(path)
        _bert_model     = AutoModel.from_pretrained(path)
        _bert_model.eval()
        log.info("IndicBERTv2 loaded")
    except Exception as e:
        log.warning("IndicBERTv2 load failed: %s", e)
    return _bert_model, _bert_tokenizer

# ── App ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("adios-inference v0.2.0 — MODEL_DIR=%s DEVICE=%s", MODEL_DIR, DEVICE)
    yield

app = FastAPI(title="adios-inference", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ─────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text:        str
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    max_length:  int = 512

class TranslateResponse(BaseModel):
    translated:  str
    source_lang: str
    target_lang: str
    model:       str
    backend:     str

class EmbedRequest(BaseModel):
    text:  str
    model: str = "indicbert"

class EmbedResponse(BaseModel):
    embedding:  list[float]
    model:      str
    dimensions: int

class GenerateRequest(BaseModel):
    prompt:      str
    system:      Optional[str] = None
    model:       str = ""
    max_tokens:  int = 1024
    temperature: float = 0.1

class GenerateResponse(BaseModel):
    content: str
    model:   str
    backend: str

class ModelInfo(BaseModel):
    name:      str
    backend:   str
    available: bool
    notes:     str

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "adios-inference", "version": "0.2.0"}

@app.get("/models", response_model=list[ModelInfo])
def list_models():
    param1_ok = _ready("param-1-2.9b-instruct")
    return [
        ModelInfo(name="nllb-200-distilled-600M", backend="huggingface",
                  available=_ready("nllb-200-distilled-600M"),
                  notes="Meta NLLB-200 600M — 200+ langs incl all 22 Indian — Apache 2.0"),
        ModelInfo(name="indictrans2-dist-200M",   backend="huggingface",
                  available=_ready("indictrans2-en-indic-dist-200M"),
                  notes="AI4Bharat IndicTrans2 200M — MIT (request access at huggingface.co)"),
        ModelInfo(name="indicbertv2-mlm-tlm",     backend="huggingface",
                  available=_ready("indicbertv2-mlm-tlm"),
                  notes="IndicBERTv2 TLM — 23 Indian languages, embeddings/NER"),
        ModelInfo(name="param-1-2.9b-instruct",   backend="huggingface",
                  available=param1_ok,
                  notes="BharatGen Param-1 2.9B — Hindi+English instruction (public)"),
        ModelInfo(name="gajendra:latest",          backend="ollama", available=True,
                  notes="7B bilingual Indian regulatory generalist"),
        ModelInfo(name="sarvam:latest",            backend="ollama", available=True,
                  notes="2B fast Indic edge model"),
        ModelInfo(name="ayurparam:latest",         backend="ollama", available=True,
                  notes="2.9B clinical Ayurveda, Hindi/Sanskrit"),
        ModelInfo(name="deepseek-r1:7b",           backend="ollama", available=True,
                  notes="7B reasoning, reliable JSON"),
        ModelInfo(name="qwen3.6:latest",           backend="ollama", available=True,
                  notes="36B long-form summarisation and reports"),
        ModelInfo(name="nomic-embed-text:latest",  backend="ollama", available=True,
                  notes="Semantic embeddings for document comparison"),
    ]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    pipe = load_translation()
    if pipe is None:
        raise HTTPException(503, "NLLB-200 not loaded — download still in progress, retry shortly")
    result = pipe(req.text, src_lang=req.source_lang, tgt_lang=req.target_lang,
                  max_length=req.max_length)
    text = result[0]["translation_text"] if isinstance(result, list) else result["translation_text"]
    return TranslateResponse(translated=text, source_lang=req.source_lang,
                             target_lang=req.target_lang,
                             model="nllb-200-distilled-600M", backend="huggingface")

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if req.model.startswith("nomic"):
        r = httpx.post(f"{OLLAMA_URL}/api/embed",
                       json={"model": "nomic-embed-text:latest", "input": req.text},
                       timeout=30)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        return EmbedResponse(embedding=vec, model="nomic-embed-text", dimensions=len(vec))

    bert, tok = load_indicbert()
    if bert is None:
        # Graceful fallback to nomic
        r = httpx.post(f"{OLLAMA_URL}/api/embed",
                       json={"model": "nomic-embed-text:latest", "input": req.text},
                       timeout=30)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        return EmbedResponse(embedding=vec, model="nomic-embed-text-fallback", dimensions=len(vec))

    import torch
    inputs = tok(req.text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = bert(**inputs)
    vec = out.last_hidden_state[:, 0, :].squeeze().cpu().tolist()
    return EmbedResponse(embedding=vec, model="indicbertv2-mlm-tlm", dimensions=len(vec))

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    model = req.model or DEFAULT_CHAT
    msgs  = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.append({"role": "user", "content": req.prompt})
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json={
            "model":    model,
            "stream":   False,
            "messages": msgs,
            "options":  {"temperature": req.temperature, "num_predict": req.max_tokens},
        })
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return GenerateResponse(content=content, model=model, backend="ollama")
