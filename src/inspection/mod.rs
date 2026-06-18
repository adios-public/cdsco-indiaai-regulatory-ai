//! Converts unstructured / handwritten site inspection observations into
//! standardised CDSCO inspection report format (critical / major / minor).
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{error::AppError, AppState};
use crate::ollama::extract_json_obj;

const SYSTEM: &str = r#"You are a CDSCO inspection officer drafting a formal inspection report.
Convert the raw site inspection observations into a standardised CDSCO report structure.

Return a JSON object with these exact keys:
- "executive_summary": concise overview max 150 words
- "critical_observations": array of strings — immediate patient-safety risk
- "major_observations":    array of strings — significant non-compliance needing CAPA
- "minor_observations":    array of strings — minor deviations for improvement
- "recommendations":       array of strings — specific corrective/preventive actions

Classify per CDSCO Schedule M / GCP / GLP as applicable.
Return only valid JSON. No preamble. No markdown fences."#;

// ── Types ──────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InspectionRequest {
    pub observations_raw: String,
    pub site_name:        String,
    pub inspection_date:  String,
    pub inspector_name:   Option<String>,
    pub inspection_type:  Option<String>,
}

#[derive(Serialize)]
pub struct InspectionResponse {
    pub site_name:               String,
    pub inspection_date:         String,
    pub inspection_type:         String,
    pub executive_summary:       String,
    pub critical_observations:   Vec<String>,
    pub major_observations:      Vec<String>,
    pub minor_observations:      Vec<String>,
    pub recommendations:         Vec<String>,
    pub formatted_report:        String,
}

// ── Handler ───────────────────────────────────────────────────────────────────────

pub async fn handle(
    State(state): State<AppState>,
    Json(req):    Json<InspectionRequest>,
) -> Result<Json<InspectionResponse>, AppError> {
    let itype      = req.inspection_type.clone().unwrap_or_else(|| "GMP".to_owned());
    let inspector  = req.inspector_name.clone().unwrap_or_else(|| "[REDACTED]".to_owned());

    let user = format!(
        "Inspection Type: {itype}\nSite: {}\nDate: {}\n\nRaw Observations:\n{}",
        req.site_name, req.inspection_date, req.observations_raw
    );

    let raw = state.ollama
        .chat(&state.settings.powerful_model, SYSTEM, &user, 2048)
        .await?;

    let json_str = extract_json_obj(&raw);
    let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|_| {
        serde_json::json!({
            "executive_summary": &raw[..raw.len().min(500)],
            "critical_observations": [],
            "major_observations":    [],
            "minor_observations":    [],
            "recommendations":       ["Manual review required — parsing failed."]
        })
    });

    let critical = str_vec(&parsed["critical_observations"]);
    let major    = str_vec(&parsed["major_observations"]);
    let minor    = str_vec(&parsed["minor_observations"]);
    let recs     = str_vec(&parsed["recommendations"]);
    let exec     = parsed["executive_summary"].as_str().unwrap_or("").to_owned();

    let report = format!(
        "# CDSCO Inspection Report\n**Site:** {}  \n**Date:** {}  \n**Type:** {itype}  \n**Inspector:** {inspector}\n\n\
         ## Executive Summary\n{exec}\n\n{}{}{}{}",
        req.site_name, req.inspection_date,
        section("Critical Observations", &critical),
        section("Major Observations",    &major),
        section("Minor Observations",    &minor),
        section("Recommendations",       &recs),
    );

    Ok(Json(InspectionResponse {
        site_name:             req.site_name,
        inspection_date:       req.inspection_date,
        inspection_type:       itype,
        executive_summary:     exec,
        critical_observations: critical,
        major_observations:    major,
        minor_observations:    minor,
        recommendations:       recs,
        formatted_report:      report,
    }))
}

fn str_vec(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_owned())).collect())
        .unwrap_or_default()
}

fn section(title: &str, items: &[String]) -> String {
    if items.is_empty() {
        return format!("## {title}\nNil\n\n");
    }
    let body = items.iter().enumerate()
        .map(|(i, s)| format!("  {}. {s}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!("## {title}\n{body}\n\n")
}
