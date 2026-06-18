# Model Card — AdiOS Regulatory AI

## Model Details

**Organisation:** AdiOS Platform Private Limited  
**Contact:** malay@adiosplat.io  
**Version:** 1.0.0 (Stage 1 Hackathon Submission)  
**Date:** June 2026  

## Intended Use

**Primary use:** Assist CDSCO reviewers in processing drug/device regulatory submissions by:
- Detecting and removing PII/PHI from clinical documents before review
- Summarising lengthy application documents, SAE narrations, and meeting transcripts
- Verifying completeness of mandatory submission fields
- Classifying SAE severity and prioritising reviewer queues
- Converting site inspection notes into standardised reports

**Intended users:** CDSCO regulatory reviewers, pharmacovigilance officers, site inspectors  
**Out-of-scope:** Clinical diagnosis, prescribing decisions, patient-facing use  

## Model Architecture

The system combines:
1. **Microsoft Presidio** (rule-based + NER) for PII/PHI detection — no ML model training required
2. **spaCy en_core_web_lg** — pretrained English NER for person/location/date entities
3. **Anthropic Claude Sonnet 4.6** — for summarisation, classification, and report generation tasks via API
4. **sentence-transformers/all-MiniLM-L6-v2** — for semantic document comparison

## Training Data

**Stage 1:** No fine-tuning performed. System uses:
- Pretrained spaCy `en_core_web_lg` (trained on OntoNotes 5)
- Presidio built-in recognisers (no training data)
- Claude Sonnet 4.6 (Anthropic pretrained foundation model)
- Sentence-Transformers pretrained checkpoint

**Stage 2:** Models will be refined/fine-tuned on CDSCO-provided anonymised/synthetic datasets.

## Privacy and Security

- **No raw PII is stored or logged.** Pseudonymisation tokens are one-way SHA-256 HMAC hashes.
- **Irreversible anonymisation** replaces entities with generalised labels; the original values cannot be reconstructed.
- **No training on submitted data.** The Claude API does not train on API inputs (per Anthropic’s data usage policy).
- **Audit trail:** Every API call logs entity types detected (not values) and compliance flags raised.
- **DPDP Act 2023 compliance:** Sensitive personal data (Aadhaar, health data) is flagged and de-identified before any downstream processing.

## Limitations

1. **Language:** English only in Stage 1. Indian language support (Hindi, Bengali, Tamil etc.) planned for Stage 2 via IndicBERT / IndicNLP.
2. **Handwriting:** Stage 1 assumes OCR-processed text input. Native handwriting recognition added in Stage 2.
3. **Duplicate detection:** Stage 1 uses a placeholder heuristic. Stage 2 uses vector similarity against the live CDSCO case database.
4. **Domain-specific fine-tuning:** The LLM is prompted but not fine-tuned on CDSCO-specific regulatory language. Fine-tuning on the CDSCO dataset in Stage 2 will improve precision.
5. **k-anonymity estimate:** The current k-anonymity computation is approximate. A formal privacy audit against the CDSCO dataset will be performed in Stage 2.

## Ethical Considerations

- The system is an **assistance tool**, not a decision-maker. All regulatory decisions remain with qualified CDSCO officers.
- Classification outputs include confidence scores and rationale to support human oversight.
- The system does not profile or score patients — it classifies events, not individuals.
- All outputs are auditable and traceable to source document sections.

## Evaluation Metrics

See `README.md` § Evaluation Metrics for the full table of benchmarks, targets, and datasets.

## Responsible AI Principles (IndiaAI)

| Principle | Implementation |
|-----------|----------------|
| Safety & Reliability | Retry logic, fallback handling, confidence thresholds |
| Privacy & Security | DPDP-compliant de-identification, no PII logging |
| Transparency | Rationale field in all classification outputs |
| Accountability | Human reviewer always in the decision loop |
| Non-discrimination | No patient profiling; event-level classification only |
| Fairness | No demographic features used in SAE classification |
