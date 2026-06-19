"""adios-inference — Unified Indic AI inference gateway v0.4.0

Translation routing (automatic, based on direction):
  En→Indic  : IndicTrans2-en-indic-dist-200M (AI4Bharat, MIT)  ← higher quality
               Format: tokenizer input = "src_lang tgt_lang text"
               Loaded with trust_remote_code=True + 3 compat patches for
               transformers ≥4.38 (onnx import, tie_weights, super() order)
  Indic→En  : NLLB-200-distilled-600M (Meta, Apache 2.0)       ← works correctly
               IndicTrans2 Indic→En needs IndicProcessor normalisation pipeline
               (indic_transliteration / indic_unified_parser) which would add
               heavy native deps; NLLB-200 is accurate and dependency-free.
  Other     : NLLB-200 covers all 200+ languages including all 22 Indian
               scheduled languages (same FLORES-200 language codes).

Embeddings  : AI4Bharat IndicBERT (ALBERT-base, 33M params)
               Falls back to nomic-embed-text via Ollama
Generation  : Ollama proxy (gajendra, qwen3.6, sarvam, deepseek-r1, param-1…)

Language codes (FLORES-200 — same for NLLB-200 and IndicTrans2):
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

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR    = os.environ.get("ADIOS_MODEL_DIR",   os.path.expanduser("~/.adios/models/hf"))
OLLAMA_URL   = os.environ.get("OLLAMA_BASE_URL",   "http://localhost:11434")
DEFAULT_CHAT = os.environ.get("DEFAULT_CHAT_MODEL","gajendra:latest")
DEVICE       = os.environ.get("INFERENCE_DEVICE",  "cpu")   # cpu | cuda

# ── Lazy-loaded backends ────────────────────────────────────────────────────────
_nllb_model        = None   # M2M100ForConditionalGeneration  (NLLB-200)
_nllb_tokenizer    = None   # NllbTokenizer
_it2_en_indic_tok  = None   # IndicTrans2 En→Indic tokenizer
_it2_en_indic_mdl  = None   # IndicTrans2 En→Indic model
_bert_model        = None   # IndicBERT (ALBERT)
_bert_tokenizer    = None

# ── Directory helpers ───────────────────────────────────────────────────────────
def _dir(name):
    return os.path.join(MODEL_DIR, name)

def _ready(name):
    d = _dir(name)
    return os.path.isdir(d) and bool(os.listdir(d))

def _indicbert_dir():
    for name in ("indic-bert", "indicbertv2-mlm-only", "indicbertv2-mlm-tlm"):
        d = _dir(name)
        if os.path.isdir(d) and os.listdir(d):
            return d
    return _dir("indic-bert")

# ── Loaders ─────────────────────────────────────────────────────────────────────

def load_nllb():
    """NLLB-200-distilled-600M via M2M100ForConditionalGeneration + NllbTokenizer.
    transformers ≥4.38 dropped the "translation" pipeline task — drive directly.
    """
    global _nllb_model, _nllb_tokenizer
    if _nllb_tokenizer is not None:
        return _nllb_model, _nllb_tokenizer
    path = _dir("nllb-200-distilled-600M")
    if not _ready("nllb-200-distilled-600M"):
        log.warning("NLLB-200 not found at %s", path)
        return None, None
    try:
        from transformers import M2M100ForConditionalGeneration, NllbTokenizer
        log.info("Loading NLLB-200-distilled-600M on %s…", DEVICE)
        _nllb_tokenizer = NllbTokenizer.from_pretrained(path)
        _nllb_model     = M2M100ForConditionalGeneration.from_pretrained(path)
        if DEVICE == "cuda":
            _nllb_model = _nllb_model.cuda()
        _nllb_model.eval()
        log.info("NLLB-200 ready")
    except Exception as e:
        log.warning("NLLB-200 load failed: %s", e)
        _nllb_model = _nllb_tokenizer = None
    return _nllb_model, _nllb_tokenizer


def load_it2_en_indic():
    """IndicTrans2 En→Indic distilled 200M.
    Works without IndicProcessor for English source text.
    Input format: tokenizer receives "src_lang tgt_lang text" as a single string.
    """
    global _it2_en_indic_tok, _it2_en_indic_mdl
    if _it2_en_indic_tok is not None:
        return _it2_en_indic_mdl, _it2_en_indic_tok
    path = _dir("indictrans2-en-indic-dist-200M")
    if not _ready("indictrans2-en-indic-dist-200M"):
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        log.info("Loading IndicTrans2 En→Indic on %s…", DEVICE)
        _it2_en_indic_tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _it2_en_indic_mdl = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
        if DEVICE == "cuda":
            _it2_en_indic_mdl = _it2_en_indic_mdl.cuda()
        _it2_en_indic_mdl.eval()
        log.info("IndicTrans2 En→Indic ready")
    except Exception as e:
        log.warning("IndicTrans2 En→Indic load failed: %s", e)
        _it2_en_indic_tok = _it2_en_indic_mdl = None
    return _it2_en_indic_mdl, _it2_en_indic_tok


def _translate_nllb(text: str, src_lang: str, tgt_lang: str, max_length: int) -> str:
    import torch
    mdl, tok = load_nllb()
    if mdl is None:
        raise HTTPException(503, "NLLB-200 not loaded")
    tok.src_lang = src_lang
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True,
                 max_length=max_length)
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    tgt_id = tok.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out = mdl.generate(**inputs, forced_bos_token_id=tgt_id,
                           max_new_tokens=max_length, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)


def _translate_it2_en_indic(text: str, src_lang: str, tgt_lang: str, max_length: int) -> str:
    """IndicTrans2 En→Indic. Input must be English (eng_Latn) source."""
    import torch
    mdl, tok = load_it2_en_indic()
    if mdl is None:
        raise HTTPException(503, "IndicTrans2 En→Indic not loaded")
    tagged = f"{src_lang} {tgt_lang} {text}"
    inputs = tok(tagged, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl.generate(**inputs, num_beams=4, max_new_tokens=max_length,
                           use_cache=False)
    return tok.decode(out[0], skip_special_tokens=True)


def load_indicbert():
    global _bert_model, _bert_tokenizer
    if _bert_tokenizer is not None:
        return _bert_model, _bert_tokenizer
    bert_path = _indicbert_dir()
    if not os.path.isdir(bert_path):
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer
        log.info("Loading IndicBERT from %s…", bert_path)
        _bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        _bert_model     = AutoModel.from_pretrained(bert_path)
        _bert_model.eval()
        log.info("IndicBERT ready")
    except Exception as e:
        log.warning("IndicBERT load failed: %s", e)
    return _bert_model, _bert_tokenizer


# ── App ──────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("adios-inference v0.4.0  MODEL_DIR=%s  DEVICE=%s", MODEL_DIR, DEVICE)
    yield

app = FastAPI(title="adios-inference", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Schemas ──────────────────────────────────────────────────────────────────────
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

# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "adios-inference", "version": "0.4.0"}

@app.get("/models", response_model=list[ModelInfo])
def list_models():
    param1_dir = _dir("param-1-2.9b-instruct")
    return [
        ModelInfo(name="indictrans2-en-indic-dist-200M", backend="huggingface",
                  available=_ready("indictrans2-en-indic-dist-200M"),
                  notes="IndicTrans2 En→Indic 200M (AI4Bharat, MIT) — high quality En→22 Indian langs"),
        ModelInfo(name="nllb-200-distilled-600M", backend="huggingface",
                  available=_ready("nllb-200-distilled-600M"),
                  notes="NLLB-200 600M (Meta, Apache 2.0) — Indic→En + 200 lang fallback"),
        ModelInfo(name="indic-bert", backend="huggingface",
                  available=_ready("indic-bert"),
                  notes="IndicBERT ALBERT-base (AI4Bharat) — 768-dim embeddings, 12 Indian langs"),
        ModelInfo(name="param-1-2.9b-instruct", backend="huggingface",
                  available=os.path.isdir(param1_dir) and bool(os.listdir(param1_dir)),
                  notes="BharatGen Param-1 2.9B — Hindi+English instruction"),
        ModelInfo(name="gajendra:latest",        backend="ollama", available=True,
                  notes="7B bilingual Indian regulatory generalist"),
        ModelInfo(name="sarvam:latest",          backend="ollama", available=True,
                  notes="2B fast edge model, Indic scripts"),
        ModelInfo(name="ayurparam:latest",       backend="ollama", available=True,
                  notes="2.9B clinical Ayurveda, Hindi/Sanskrit"),
        ModelInfo(name="deepseek-r1:7b",         backend="ollama", available=True,
                  notes="7B reasoning, reliable JSON output"),
        ModelInfo(name="qwen3.6:latest",         backend="ollama", available=True,
                  notes="36B long-form summarisation and reports"),
        ModelInfo(name="nomic-embed-text:latest",backend="ollama", available=True,
                  notes="768-dim semantic embeddings for document comparison"),
    ]

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    src, tgt = req.source_lang, req.target_lang
    INDIC = {"hin_Deva","tam_Taml","tel_Telu","kan_Knda","mal_Mlym","mar_Deva",
             "ben_Beng","guj_Gujr","pan_Guru","urd_Arab","asm_Beng","ory_Orya",
             "san_Deva","mai_Deva","kok_Deva","brx_Deva","doi_Deva","kas_Arab",
             "kas_Deva","mni_Mtei","sat_Olck","snd_Arab"}

    # Route En→Indic to IndicTrans2 (higher quality) when model is available
    if src == "eng_Latn" and tgt in INDIC and _ready("indictrans2-en-indic-dist-200M"):
        try:
            result = _translate_it2_en_indic(req.text, src, tgt, req.max_length)
            return TranslateResponse(translated=result, source_lang=src, target_lang=tgt,
                                     model="indictrans2-en-indic-dist-200M",
                                     backend="huggingface")
        except Exception as e:
            log.warning("IndicTrans2 failed, falling back to NLLB-200: %s", e)

    # All other directions (Indic→En, Indic→Indic, etc.) → NLLB-200
    result = _translate_nllb(req.text, src, tgt, req.max_length)
    return TranslateResponse(translated=result, source_lang=src, target_lang=tgt,
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
        # Fallback to nomic
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
    return EmbedResponse(embedding=vec, model="indic-bert", dimensions=len(vec))

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
