# AdiOS Regulatory AI — CDSCO-IndiaAI Health Innovation Hackathon

> **Submitted by:** AdiOS Platform Private Limited (CIN: U58201TS2026PTC211867)  
> **DPIIT Recognised Startup** | Hyderabad, India  
> **Hackathon:** CDSCO-IndiaAI Health Innovation Acceleration Hackathon (MeitY / IndiaAI)

---

## Overview

A comprehensive, sovereign AI platform for CDSCO regulatory workflow automation — built to make drug and device approvals faster, consistent, and auditable without compromising data sovereignty or compliance.

The system integrates five core capabilities into a single REST API:

| # | Module | What it does |
|---|--------|--------------|
| 1 | **Anonymisation** | Hybrid rule-based + NLP PII/PHI detection with two-step de-identification (pseudonymisation → irreversible anonymisation) across structured and unstructured documents |
| 2 | **Document Summarisation** | Extracts and synthesises critical regulatory information from SUGAM checklists, SAE case narrations, and meeting transcripts/audio |
| 3 | **Completeness Assessment + Document Comparison** | Verifies mandatory fields in clinical applications and SAE reports; highlights substantive changes between document versions |
| 4 | **SAE Classification** | Classifies adverse events by severity (death / disability / hospitalisation / other), detects duplicates, and prioritises for reviewer assignment |
| 5 | **Inspection Report Generation** | Converts unstructured and handwritten site inspection observations into standardised CDSCO-template reports |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI REST Layer                      │
│              /anonymise  /summarise  /assess              │
│              /classify   /inspect    /compare             │
└───────────┬──────────────┬──────────────┬────────────────┘
            │              │              │
   ┌────────▼──────┐ ┌─────▼──────┐ ┌───▼────────────────┐
   │  Anonymisation│ │Summarisation│ │Completeness/Compare│
   │  Engine       │ │Engine       │ │Engine              │
   │  (Presidio +  │ │(Claude API +│ │(Rule engine +      │
   │   spaCy NER)  │ │ extractive) │ │ semantic diff)     │
   └───────────────┘ └─────────────┘ └────────────────────┘
            │              │              │
   ┌────────▼──────────────▼──────────────▼────────────────┐
   │              Core LLM + Config Layer                    │
   │         (Anthropic Claude Sonnet 4.6 / local)          │
   └────────────────────────────────────────────────────────┘
```

### Compliance

- **DPDP Act 2023** — data minimisation, purpose limitation, pseudonymisation before processing
- **NDHM Health Data Management Policy** — de-identification standards
- **ICMR Ethical Guidelines** — IRB-compatible anonymisation
- **CDSCO Standards** — SAE classification per Schedule Y, inspection report templates
- **IT Act 2000 / CERT-In** — secure token storage, audit trails

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (recommended)
- Anthropic API key (for summarisation and report generation)

### Run with Docker

```bash
git clone https://github.com/adios-public/cdsco-indiaai-regulatory-ai.git
cd cdsco-indiaai-regulatory-ai
cp .env.example .env          # add your ANTHROPIC_API_KEY
docker compose up --build
```

API available at `http://localhost:8000`  
Swagger UI at `http://localhost:8000/docs`

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env
uvicorn src.api.main:app --reload
```

---

## API Reference

### 1. Anonymisation

```http
POST /api/v1/anonymise
Content-Type: application/json

{
  "text": "Patient John Doe (DOB: 12/03/1985, Aadhaar: 1234-5678-9012) presented...",
  "mode": "pseudonymise",   // or "anonymise"
  "document_type": "clinical_trial"
}
```

### 2. Document Summarisation

```http
POST /api/v1/summarise
Content-Type: application/json

{
  "text": "...",
  "source_type": "sugam_checklist",  // sae_narration | meeting_transcript
  "output_format": "structured"
}
```

### 3. Completeness Assessment

```http
POST /api/v1/assess-completeness
Content-Type: application/json

{
  "document": { ... },
  "schema_type": "new_drug_application",  // sae_report | clinical_trial
  "flag_inconsistencies": true
}
```

### 4. Document Comparison

```http
POST /api/v1/compare
Content-Type: application/json

{
  "document_v1": "...",
  "document_v2": "...",
  "highlight_substantive": true
}
```

### 5. SAE Classification

```http
POST /api/v1/classify-sae
Content-Type: application/json

{
  "case_narration": "...",
  "check_duplicate": true
}
```

---

## Evaluation Metrics

| Task | Metric | Target |
|------|--------|--------|
| PII/PHI Detection | F1-score (entity-level) | ≥ 0.90 |
| Anonymisation | k-anonymity, l-diversity, t-closeness | k≥5 |
| Summarisation | ROUGE-1/2/L, BERT Score | ROUGE-1 ≥ 0.45 |
| Completeness | Macro-F1, MCC | ≥ 0.88 |
| SAE Classification | Macro-F1, Confusion Matrix | ≥ 0.87 |
| Document Comparison | Semantic similarity (cosine) | — |
| Latency | Time per document | < 5s |

---

## Repository Structure

```
├── src/
│   ├── api/              FastAPI routes and app entry point
│   ├── anonymisation/    PII/PHI detection + de-identification engine
│   ├── summarisation/    Document summarisation (SUGAM / SAE / transcripts)
│   ├── completeness/     Form completeness checker + document comparator
│   ├── classification/   SAE severity classifier + deduplication
│   ├── inspection/       Inspection report generator
│   └── core/             Config, LLM client, shared utilities
├── data/
│   ├── schemas/          CDSCO form schemas (SUGAM, SAE, NDA checklists)
│   └── sample/           Anonymised sample documents for testing
├── tests/                Pytest test suite
├── notebooks/            Jupyter notebooks — EDA and pipeline demo
└── docs/                 Architecture diagram, model card
```

---

## Team

**AdiOS Platform Private Limited**  
Founder / CTO: Malay Baral  
Email: malay@adiosplat.io  
Website: https://www.adiosplat.io

---

## Licence

Submitted for the CDSCO-IndiaAI Health Innovation Acceleration Hackathon.  
Per hackathon terms, IP of submitted models/methodologies is assigned to IndiaAI and CDSCO upon award.  
Code structure and AdiOS platform components remain proprietary to AdiOS Platform Pvt Ltd.
