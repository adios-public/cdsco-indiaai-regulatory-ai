//! SAE (Serious Adverse Event) classifier.
//!
//! Severity categories per Schedule Y of the Drugs and Cosmetics Act 1940.
//! Events classified as death or life-threatening require 15-day expedited
//! reporting to CDSCO.
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{error::AppError, AppState};
use crate::ollama::extract_json_obj;

const SYSTEM: &str = r#"You are a pharmacovigilance expert classifying Serious Adverse Events (SAEs)
for CDSCO regulatory review under Schedule Y of the Drugs and Cosmetics Act 1940.

Analyse the case narration and return a JSON object with these exact keys:
- "severity": one of "death" | "disability" | "hospitalisation" | "life_threatening" | "congenital_anomaly" | "other"
- "confidence": float 0.0-1.0
- "rationale": brief reasoning max 100 words
- "expedited_reporting_required": true if 15-day expedited reporting is required per Schedule Y

Return only valid JSON. No preamble. No markdown fences."#;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ClassificationRequest {
    pub case_narration:   String,
    // Reserved for Stage 2: vector search against CDSCO case DB
    #[allow(dead_code)]
    pub check_duplicate:  Option<bool>,
    #[allow(dead_code)]
    pub existing_case_ids: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct ClassificationResponse {
    pub severity:                    String,
    pub priority:                    &'static str,
    pub confidence:                  f32,
    pub is_duplicate:                bool,
    pub duplicate_case_ids:          Vec<String>,
    pub rationale:                   String,
    pub expedited_reporting_required: bool,
}

fn severity_to_priority(s: &str) -> &'static str {
    match s {
        "death" | "life_threatening"    => "critical",
        "disability" | "hospitalisation"
        | "congenital_anomaly"           => "high",
        _                               => "medium",
    }
}

// ── Handler ──────────────────────────────────────────────────────────────────

pub async fn handle(
    State(state): State<AppState>,
    Json(req):    Json<ClassificationRequest>,
) -> Result<Json<ClassificationResponse>, AppError> {
    let raw = state.ollama
        .chat(&state.settings.default_model, SYSTEM, &req.case_narration, 512)
        .await?;

    let json_str = extract_json_obj(&raw);
    let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|_| {
        serde_json::json!({
            "severity": "other",
            "confidence": 0.5,
            "rationale": "Classification uncertain — manual review required.",
            "expedited_reporting_required": false
        })
    });

    let severity   = parsed["severity"].as_str().unwrap_or("other").to_owned();
    let priority   = severity_to_priority(&severity);
    let confidence = parsed["confidence"].as_f64().unwrap_or(0.5) as f32;
    let expedited  = parsed["expedited_reporting_required"].as_bool().unwrap_or(false);
    let rationale  = parsed["rationale"].as_str().unwrap_or("").to_owned();

    Ok(Json(ClassificationResponse {
        severity,
        priority,
        confidence: (confidence * 1000.0).round() / 1000.0,
        is_duplicate: false,
        duplicate_case_ids: Vec::new(),
        rationale,
        expedited_reporting_required: expedited,
    }))
}
