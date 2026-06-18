# AdiOS Regulatory AI — CDSCO-IndiaAI Health Innovation Hackathon

> **Submitted by:** AdiOS Platform Private Limited (CIN: U58201TS2026PTC211867)  
> **DPIIT Recognised Startup** | Hyderabad, India  
> **Hackathon:** CDSCO-IndiaAI Health Innovation Acceleration Hackathon (MeitY / IndiaAI)  
> **Implementation:** 100% Rust — aligned with AdiOS Platform’s sovereign-first architecture

---

## Overview

A sovereign, on-device AI platform for CDSCO regulatory workflow automation.
All inference runs **locally via Ollama / AIKosh** — no external API calls, no data leaves the machine.
Fully compliant with DPDP Act 2023, NDHM, ICMR, and CDSCO data standards.

| # | Endpoint | Capability |
|---|----------|------------|
| 1 | `POST /api/v1/anonymise` | Hybrid regex + LLM PII/PHI detection, two-step de-identification |
| 2 | `POST /api/v1/summarise` | SUGAM checklists / SAE narrations / meeting transcripts |
| 3 | `POST /api/v1/assess-completeness` | Mandatory field verification against CDSCO schemas |
| 4 | `POST /api/v1/compare` | Semantic + lexical document version diff |
| 5 | `POST /api/v1/classify-sae` | Severity classification (Schedule Y) + expedited flag |
| 6 | `POST /api/v1/generate-inspection-report` | Unstructured observations → CDSCO report |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Rust 2021 |
| HTTP API | `axum 0.7` + `tokio` |
| LLM inference | Local Ollama (`gajendra:latest` default, `qwen3.6:latest` for long-form) |
| Embeddings | `nomic-embed-text:latest` via Ollama |
| PII detection | `regex` (Indian patterns) + LLM NER |
| Pseudonymisation | HMAC-SHA256 (`hmac` + `sha2`) |
| Document diff | `similar` crate (line diff) + cosine similarity |
| Serialisation | `serde` + `serde_json` |

### Model Routing

| Task | Model | Why |
|------|-------|-----|
| SAE classification | `gajendra:latest` (6.9B) | AdiOS custom Indian-context model |
| Inspection reports | `gajendra:latest` | Same |
| Summarisation | `qwen3.6:latest` (36B) | Long-form synthesis quality |
| Embeddings | `nomic-embed-text:latest` | Semantic document comparison |
| Anonymisation + completeness | No LLM | Pure rule engine — deterministic, auditable |

---

## Quick Start

### Prerequisites
- Rust 1.78+ (`rustup`)
- Ollama running locally with `gajendra`, `qwen3.6`, and `nomic-embed-text` models

```bash
git clone https://github.com/adios-public/cdsco-indiaai-regulatory-ai.git
cd cdsco-indiaai-regulatory-ai
cp .env.example .env
cargo run --release
```

API at `http://localhost:8000` | Health: `GET /health`

### Docker

```bash
docker compose up --build
```

> Ollama must be accessible at `http://host.docker.internal:11434` (Mac/Windows)
> or set `OLLAMA_BASE_URL=http://172.17.0.1:11434` (Linux).

---

## API Examples

### Anonymise a clinical document

```bash
curl -s -X POST http://localhost:8000/api/v1/anonymise \
  -H 'Content-Type: application/json' \
  -d '{"text": "Patient Ramesh Kumar (Aadhaar: 1234-5678-9012) enrolled in trial.", "mode": "pseudonymise"}'
```

### Classify an SAE

```bash
curl -s -X POST http://localhost:8000/api/v1/classify-sae \
  -H 'Content-Type: application/json' \
  -d '{"case_narration": "Patient developed anaphylaxis post-infusion, hospitalised for 48 hours, recovered."}'
```

### Summarise an SAE narration

```bash
curl -s -X POST http://localhost:8000/api/v1/summarise \
  -H 'Content-Type: application/json' \
  -d '{"text": "...", "source_type": "sae_narration"}'
```

---

## Repository Structure

```
├── Cargo.toml
├── src/
│   ├── main.rs             axum app + routes
│   ├── config.rs           settings from .env
│   ├── error.rs            AppError + IntoResponse
│   ├── ollama.rs           typed Ollama HTTP client
│   ├── anonymisation/      PII/PHI detection + de-identification
│   ├── summarisation/      document summarisation
│   ├── completeness/       completeness checker + document comparator
│   ├── classification/     SAE severity classifier
│   └── inspection/         inspection report generator
├── data/
│   ├── schemas/            CDSCO SUGAM + SAE form schemas (JSON)
│   └── sample/             anonymised sample documents
└── docs/
    ├── architecture.md     system design
    └── model_card.md       model card (required by hackathon)
```

---

## Compliance

- **DPDP Act 2023** — Aadhaar/health data flagged, pseudonymised/anonymised before processing
- **NDHM Health Data Management Policy** — de-identification per standards
- **ICMR Ethical Guidelines** — no PII in logs; audit trail of entity types only
- **CDSCO / Schedule Y** — SAE severity + 15-day expedited reporting logic
- **Zero external API calls** — all inference on-device; data never leaves the CDSCO environment

---

## Team

**AdiOS Platform Private Limited**  
Founder / CTO: Malay Baral — malay@adiosplat.io — https://www.adiosplat.io
