"""adios-inference — Unified Indic AI inference gateway v0.3.0

Backends:
  Translation : Meta NLLB-200-distilled-600M (Apache 2.0, no gate)  ← primary
                Uses M2M100ForConditionalGeneration + NllbTokenizer directly
                (transformers ≥4.38 removed the "translation" pipeline task)
                Falls back to IndicTrans2 distilled when available (gated)
  Embeddings  : AI4Bharat IndicBERTv2-MLM-only (no gate)
                Falls back to nomic-embed-text via Ollama
  Generation  : Ollama proxy (gajendra, qwen3.6, sarvam, deepseek-r1, param-1…)

Language codes (FLORES-200, same for both NLLB and IndicTrans2):
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

# ── Lazy-loaded backends ─────────────────────────────────────────────────────

_nllb_model     = None   # M2M100ForConditionalGeneration
_nllb_tokenizer = None   # NllbTokenizer
_bert_model     = None
_bert_tokenizer = None

def nllb_dir():
    return os.path.join(MODEL_DIR, "nllb-200-distilled-600M")

def indicbert_dir():
    # Prefer indicbertv2-mlm-only (public); fall back to -mlm-tlm if present
    for name in ("indicbertv2-mlm-only", "indicbertv2-mlm-tlm"):
        d = os.path.join(MODEL_DIR, name)
        if os.path.isdir(d) and os.listdir(d):
            return d
    return os.path.join(MODEL_DIR, "indicbertv2-mlm-only")

def indictrans2_dir(direction="en_indic"):
    name = "indictrans2-en-indic-dist-200M" if direction == "en_indic" else "indictrans2-indic-en-dist-200M"
    return os.path.join(MODEL_DIR, name)

def load_translation():
    """Load NLLB-200 using M2M100ForConditionalGeneration + NllbTokenizer.

    Transformers ≥4.38 removed the "translation" pipeline task shorthand;
    we drive the model directly instead so this works on any version.
    """
    global _nllb_model, _nllb_tokenizer
    if _nllb_tokenizer is not None:
        return _nllb_model, _nllb_tokenizer
    nllb = nllb_dir()
    if not os.path.isdir(nllb) or not os.listdir(nllb):
        log.warning("NLLB-200 not found at %s", nllb)
        return None, None
    try:
        from transformers import M2M100ForConditionalGeneration, NllbTokenizer
        log.info("Loading NLLB-200-distilled-600M on %s…", DEVICE)
        _nllb_tokenizer = NllbTokenizer.from_pretrained(nllb)
        _nllb_model     = M2M100ForConditionalGeneration.from_pretrained(nllb)
        if DEVICE == "cuda":
            _nllb_model = _nllb_model.cuda()
        _nllb_model.eval()
        log.info("NLLB-200 loaded OK")
    except Exception as e:
        log.warning("NLLB-200 load failed: %s", e)
        _nllb_model = _nllb_tokenizer = None
    return _nllb_model, _nllb_tokenizer

def load_indicbert():
    global _bert_model, _bert_tokenizer
    if _bert_tokenizer is not None:
        return _bert_model, _bert_tokenizer
    bert_path = indicbert_dir()
    if not os.path.isdir(bert_path):
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        log.info("Loading IndicBERTv2 from %s…", bert_path)
        _bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        _bert_model     = AutoModel.from_pretrained(bert_path)
        _bert_model.eval()
        log.info("IndicBERTv2 loaded")
    except Exception as e:
        log.warning("IndicBERTv2 load failed: %s", e)
    return _bert_model, _bert_tokenizer

# ── App ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("adios-inference v0.3.0 — MODEL_DIR=%s DEVICE=%s", MODEL_DIR, DEVICE)
    yield

app = FastAPI(title="adios-inference", version="0.3.0", lifespan=lifespan)
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
    model: str = "indicbert"   # indicbert | nomic

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

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "adios-inference", "version": "0.3.0"}

@app.get("/models", response_model=list[ModelInfo])
def list_models():
    nllb_ok    = os.path.isdir(nllb_dir()) and bool(os.listdir(nllb_dir()))
    _bd        = indicbert_dir()
    bert_ok    = os.path.isdir(_bd) and bool(os.listdir(_bd))
    it2_ok     = os.path.isdir(indictrans2_dir()) and bool(os.listdir(indictrans2_dir()))
    param1_dir = os.path.join(MODEL_DIR, "param-1-2.9b-instruct")
    param1_ok  = os.path.isdir(param1_dir) and bool(os.listdir(param1_dir))
    return [
        ModelInfo(name="nllb-200-distilled-600M",  backend="huggingface", available=nllb_ok,
                  notes="Meta NLLB-200 600M — 200+ langs incl all 22 Indian — Apache 2.0"),
        ModelInfo(name="indictrans2-dist-200M",    backend="huggingface", available=it2_ok,
                  notes="AI4Bharat IndicTrans2 distilled 200M — MIT (gated, request access)"),
        ModelInfo(name="indicbertv2-mlm-only",     backend="huggingface", available=bert_ok,
                  notes="IndicBERTv2 MLM-only — 23 Indian languages, embeddings/NER"),
        ModelInfo(name="param-1-2.9b-instruct",    backend="huggingface", available=param1_ok,
                  notes="BharatGen Param-1 2.9B — Hindi+English instruction"),
        ModelInfo(name="gajendra:latest",           backend="ollama",      available=True,
                  notes="7B bilingual Indian regulatory generalist"),
        ModelInfo(name="sarvam:latest",             backend="ollama",      available=True,
                  notes="2B fast edge model, Indic scripts"),
        ModelInfo(name="ayurparam:latest",          backend="ollama",      available=True,
                  notes="2.9B clinical Ayurveda, Hindi/Sanskrit"),
        ModelInfo(name="deepseek-r1:7b",            backend="ollama",      available=True,
                  notes="7B reasoning, reliable JSON output"),
        ModelInfo(name="qwen3.6:latest",            backend="ollama",      available=True,
                  notes="36B long-form summarisation and reports"),
        ModelInfo(name="nomic-embed-text:latest",   backend="ollama",      available=True,
                  notes="Semantic embeddings for document comparison"),
    ]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    import torch
    model, tokenizer = load_translation()
    if model is None:
        raise HTTPException(503,
            "NLLB-200 not loaded — weights still downloading or load failed.")

    tokenizer.src_lang = req.source_lang
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True,
                       max_length=req.max_length)
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    tgt_lang_id = tokenizer.convert_tokens_to_ids(req.target_lang)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=req.max_length,
            num_beams=4,
        )
    translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return TranslateResponse(
        translated=translated,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        model="nllb-200-distilled-600M",
        backend="huggingface",
    )

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if req.model.startswith("nomic") or req.model == "nomic":
        import httpx as _httpx
        r = _httpx.post(f"{OLLAMA_URL}/api/embed",
                        json={"model": "nomic-embed-text:latest", "input": req.text},
                        timeout=30)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
        return EmbedResponse(embedding=vec, model="nomic-embed-text", dimensions=len(vec))

    bert, tok = load_indicbert()
    if bert is None:
        # Fallback: nomic via Ollama
        import httpx as _httpx
        r = _httpx.post(f"{OLLAMA_URL}/api/embed",
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
    return EmbedResponse(embedding=vec, model="indicbertv2-mlm-only", dimensions=len(vec))

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
    # Strip reasoning tags; keep content that follows
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return GenerateResponse(content=content, model=model, backend="ollama")
