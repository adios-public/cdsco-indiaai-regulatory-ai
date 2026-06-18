//! PII / PHI detection and two-step de-identification.
//!
//! Step 1 — Pseudonymise: replace with reversible HMAC-SHA256 token.
//! Step 2 — Anonymise:    replace with irreversible generalised label.
//!
//! Indian-specific patterns (Aadhaar, PAN, phone) are detected by regex;
//! person names and locations are detected by the local LLM.
use axum::{extract::State, Json};
use hmac::{Hmac, Mac};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;

use crate::{error::AppError, AppState};
use crate::ollama::{extract_json_arr};

// ── Regex patterns ───────────────────────────────────────────────────────────────────

static PATTERNS: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| vec![
    ("AADHAAR",    Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap()),
    ("PAN",        Regex::new(r"\b[A-Z]{5}\d{4}[A-Z]\b").unwrap()),
    ("PHONE",      Regex::new(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b").unwrap()),
    ("EMAIL",      Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap()),
    ("PATIENT_ID", Regex::new(r"\b(?:PT|PAT|MR)-?\d{4,10}\b").unwrap()),
    ("SUBJECT_ID", Regex::new(r"\b(?:SUB|SUBJ|SID)-?\d{4,10}\b").unwrap()),
    ("DATE",       Regex::new(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b").unwrap()),
]);

static REDACT_LABELS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("AADHAAR",    "[AADHAAR REDACTED]");
    m.insert("PAN",        "[PAN REDACTED]");
    m.insert("PHONE",      "[PHONE REDACTED]");
    m.insert("EMAIL",      "[EMAIL REDACTED]");
    m.insert("PATIENT_ID", "[PATIENT-ID REDACTED]");
    m.insert("SUBJECT_ID", "[SUBJECT-ID REDACTED]");
    m.insert("DATE",       "[DATE REDACTED]");
    m.insert("PERSON",     "[INDIVIDUAL REDACTED]");
    m.insert("LOCATION",   "[LOCATION REDACTED]");
    m
});

// ── Request / Response types ────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AnonymisationRequest {
    pub text:          String,
    pub mode:          Mode,
    #[allow(dead_code)]
    pub document_type: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode { Pseudonymise, Anonymise }

impl Mode {
    fn as_str(&self) -> &'static str {
        match self { Mode::Pseudonymise => "pseudonymise", Mode::Anonymise => "anonymise" }
    }
}

#[derive(Serialize)]
pub struct DetectedEntity {
    pub text:        String,
    pub entity_type: String,
    pub start:       usize,
    pub end:         usize,
    pub score:       f32,
}

#[derive(Serialize)]
pub struct AnonymisationResponse {
    pub original_length:      usize,
    pub anonymised_text:      String,
    pub mode:                 &'static str,
    pub entities_detected:    Vec<DetectedEntity>,
    pub token_map:            HashMap<String, String>,
    pub k_anonymity_estimate: u32,
    pub compliance_flags:     Vec<String>,
}

// ── Handler ───────────────────────────────────────────────────────────────────────

pub async fn handle(
    State(state): State<AppState>,
    Json(req):    Json<AnonymisationRequest>,
) -> Result<Json<AnonymisationResponse>, AppError> {
    let text = &req.text;

    // 1. Regex detection
    let mut entities: Vec<DetectedEntity> = Vec::new();
    for (etype, re) in PATTERNS.iter() {
        for m in re.find_iter(text) {
            entities.push(DetectedEntity {
                text:        m.as_str().to_owned(),
                entity_type: etype.to_string(),
                start:       m.start(),
                end:         m.end(),
                score:       0.95,
            });
        }
    }

    // 2. LLM NER for PERSON / LOCATION
    let llm_ents = llm_ner(&state, text).await;
    entities.extend(llm_ents);

    // 3. Sort + deduplicate overlapping spans
    entities.sort_by_key(|e| e.start);
    entities = dedup_spans(entities);

    // 4. Apply replacements (right-to-left to preserve offsets)
    let mut result = text.clone();
    let mut token_map: HashMap<String, String> = HashMap::new();
    let mut by_end: Vec<&DetectedEntity> = entities.iter().collect();
    by_end.sort_by(|a, b| b.start.cmp(&a.start));

    let salt   = &state.settings.anonymisation_salt;
    let prefix = &state.settings.pseudo_token_prefix;

    for e in &by_end {
        if e.end > result.len() { continue; }
        let replacement = match req.mode {
            Mode::Pseudonymise => {
                let tok = pseudo_token(salt, prefix, &e.entity_type, &e.text);
                token_map.insert(tok.clone(), e.entity_type.clone());
                tok
            }
            Mode::Anonymise => {
                REDACT_LABELS.get(e.entity_type.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "[REDACTED]".to_owned())
            }
        };
        result.replace_range(e.start..e.end, &replacement);
    }

    // 5. Compliance flags
    let mut flags: Vec<String> = Vec::new();
    if entities.iter().any(|e| e.entity_type == "AADHAAR") {
        flags.push("DPDP-2023:S8-sensitive-personal-data".to_owned());
    }
    if entities.iter().any(|e| matches!(e.entity_type.as_str(), "PERSON" | "DATE")) {
        flags.push("ICMR:de-identification-applied".to_owned());
    }

    let redacted = result.matches("REDACTED").count() as u32;
    let k = 5_u32.max(100u32.saturating_sub(redacted * 10));

    Ok(Json(AnonymisationResponse {
        original_length:      text.len(),
        anonymised_text:      result,
        mode:                 req.mode.as_str(),
        entities_detected:    entities,
        token_map,
        k_anonymity_estimate: k,
        compliance_flags:     flags,
    }))
}

// ── Helpers ───────────────────────────────────────────────────────────────────────

async fn llm_ner(state: &AppState, text: &str) -> Vec<DetectedEntity> {
    let system = r#"Identify all PERSON names and LOCATION names in the clinical text below.
Return ONLY a JSON array: [{"text": "...", "type": "PERSON"}, {"text": "...", "type": "LOCATION"}]
No preamble. No markdown. Only valid JSON array."#;

    let Ok(raw) = state.ollama.chat(&state.settings.default_model, system, text, 512).await
    else { return Vec::new(); };

    let arr = extract_json_arr(&raw);
    let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(arr)
    else { return Vec::new(); };

    let mut out = Vec::new();
    for item in &items {
        let Some(name)  = item["text"].as_str() else { continue };
        let Some(etype) = item["type"].as_str() else { continue };
        let mut pos = 0;
        while let Some(off) = text[pos..].find(name) {
            let start = pos + off;
            let end   = start + name.len();
            out.push(DetectedEntity {
                text:        name.to_owned(),
                entity_type: etype.to_owned(),
                start, end, score: 0.85,
            });
            pos = end;
        }
    }
    out
}

fn pseudo_token(salt: &str, prefix: &str, etype: &str, text: &str) -> String {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(salt.as_bytes()).expect("HMAC init");
    mac.update(format!("{}:{}", etype, text).as_bytes());
    let digest = hex::encode(&mac.finalize().into_bytes()[..4]).to_uppercase();
    let short   = &etype[..4.min(etype.len())];
    format!("<{}-{}-{}>", prefix, short, digest)
}

fn dedup_spans(sorted: Vec<DetectedEntity>) -> Vec<DetectedEntity> {
    let mut out: Vec<DetectedEntity> = Vec::new();
    for e in sorted {
        if out.last().map_or(true, |last: &DetectedEntity| e.start >= last.end) {
            out.push(e);
        }
    }
    out
}
