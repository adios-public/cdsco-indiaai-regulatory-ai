# Solution Architecture

## AdiOS Regulatory AI — CDSCO-IndiaAI Hackathon

### System Overview

The solution is a modular, API-first platform built on Python 3.11 + FastAPI. Each of the five CDSCO problem areas maps to an independent module that can be used standalone or as part of an integrated pipeline.

```
Client (Reviewer / SUGAM Portal / MD Online)
          │
          ▼
   ┌──────────────────────────────────────────┐
   │            FastAPI REST Gateway                   │
   │    /anonymise /summarise /assess-completeness    │
   │    /compare   /classify-sae  /inspect            │
   └──────────────────────────────────────────┘
          │
   ┌─────┴───────────────────────────────────┐
   │                  Module Layer                       │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
   │  │ Anonymisation│ │ Summarisation│ │Completeness │  │
   │  │ (Presidio +  │ │ (Claude API) │ │ + Comparator│  │
   │  │  spaCy NER)  │ └─────────────┘ │ (Rules +    │  │
   │  └─────────────┘                 │  SBERT)      │  │
   │                               └─────────────┘  │
   │  ┌─────────────┐ ┌─────────────┐               │
   │  │SAE Classifier│ │  Inspection  │               │
   │  │ (Claude API) │ │   Report Gen │               │
   │  └─────────────┘ └─────────────┘               │
   └───────────────────────────────────────────┘
          │
   ┌─────┴───────────────────────────────────┐
   │          Core Layer                            │
   │   Claude Sonnet 4.6 (via Anthropic API)       │
   │   spaCy en_core_web_lg + Presidio Analyzer     │
   │   Sentence-Transformers (all-MiniLM-L6-v2)    │
   └─────────────────────────────────────────┘
```

### Module Decisions

| Module | Approach | Rationale |
|--------|----------|-----------|
| PII/PHI Detection | Microsoft Presidio + spaCy NER + custom Indian regex | Industry-standard library; extended with Aadhaar/PAN/patient-ID patterns specific to Indian regulatory context |
| Anonymisation | Two-step: SHA-256 HMAC pseudonymisation → generalisation/redaction | Matches DPDP Act 2023 de-identification requirements; reversible tokens only where needed |
| Summarisation | Claude Sonnet 4.6 with source-type-specific system prompts | Handles all three document types (checklist/SAE/transcript) with structured JSON output; best-in-class for regulatory language |
| Completeness | Rule engine against hardcoded CDSCO mandatory field lists | Deterministic, auditable, no LLM cost; schemas loaded from `data/schemas/` |
| Document Comparison | `difflib` line diff + Sentence-Transformers semantic similarity | Lexical diff catches every change; semantic layer identifies substantive vs. cosmetic edits |
| SAE Classification | Claude Sonnet 4.6 with Schedule Y-grounded prompt | Handles free-text narrations; rule layer maps severity → priority and expedited reporting flag |
| Inspection Report | Claude Sonnet 4.6 with CDSCO template prompt | Converts unstructured/handwritten observations to standard critical/major/minor taxonomy |

### Data Flow — Anonymisation Pipeline

```
Raw Document
    │
    ▼
[1] PIIDetector.detect()        ←  spaCy NER + Presidio + Indian regex patterns
    │
    ▼
[2] AnonymisationEngine.process()
    ├── mode=pseudonymise  →  replace with <TOK-PERS-A3F1B2C9> (SHA-256 HMAC token)
    └── mode=anonymise     →  replace with [INDIVIDUAL REDACTED] (irreversible)
    │
    ▼
[3] AnonymisationResponse
    ├── anonymised_text
    ├── entities_detected  (type, position, confidence)
    ├── token_map          (token → entity_type; original value never stored)
    ├── k_anonymity_estimate
    └── compliance_flags   (DPDP-2023, ICMR markers)
```

### Stage 2 Integration (CDSCO Premises)

For Stage 2, the system will be extended to:
- Integrate with the **SUGAM portal API** for live document ingestion
- Integrate with **MD Online** for device application workflows
- Deploy on the **MeitY-approved secure cloud environment**
- Replace Stage 1’s heuristic duplicate detection with **vector similarity search** against the CDSCO SAE case database
- Add **OCR pipeline** (Tesseract / Azure Vision) for handwritten inspection sheets
