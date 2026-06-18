//! Document summarisation for three CDSCO source types:
//!  • SUGAM portal application checklists
//!  • SAE (Serious Adverse Event) case narrations
//!  • Meeting transcripts / audio transcripts
//!
//! Uses the powerful local model (qwen3.6) for long-form synthesis.
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::{error::AppError, AppState};
use crate::ollama::extract_json_obj;

// ── Types ──────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    SugamChecklist,
    SaeNarration,
    MeetingTranscript,
}

impl SourceType {
    fn as_str(&self) -> &'static str {
        match self {
            SourceType::SugamChecklist    => "sugam_checklist",
            SourceType::SaeNarration      => "sae_narration",
            SourceType::MeetingTranscript => "meeting_transcript",
        }
    }

    fn system_prompt(&self, max_words: u32) -> String {
        let base = match self {
            SourceType::SugamChecklist => format!(
                "You are a CDSCO regulatory affairs specialist.\n\
                 Analyse the SUGAM portal application checklist data provided.\n\
                 Return a JSON object with keys:\n\
                 - summary: concise prose overview (max {max_words} words)\n\
                 - key_decisions: list of regulatory decision points identified\n\
                 - action_items: list of outstanding items the reviewer must act on\n\
                 - flagged_concerns: list of data gaps, inconsistencies, or risk flags"
            ),
            SourceType::SaeNarration => format!(
                "You are a pharmacovigilance expert reviewing an SAE for CDSCO.\n\
                 Analyse the case narration provided.\n\
                 Return a JSON object with keys:\n\
                 - summary: case summary max {max_words} words (patient, event, timeline, causality, outcome)\n\
                 - key_decisions: causality assessments and regulatory signals\n\
                 - action_items: follow-up required (data, expedited reporting, label change)\n\
                 - flagged_concerns: missing data, inconsistencies, duplicate signals"
            ),
            SourceType::MeetingTranscript => format!(
                "You are a CDSCO regulatory meeting secretary.\n\
                 Analyse the meeting transcript provided.\n\
                 Return a JSON object with keys:\n\
                 - summary: meeting summary max {max_words} words\n\
                 - key_decisions: decisions made\n\
                 - action_items: action items (with implicit owners where mentioned)\n\
                 - flagged_concerns: open issues or unresolved points"
            ),
        };
        format!("{base}\nReturn only valid JSON. No preamble. No markdown fences.")
    }
}

#[derive(Deserialize)]
pub struct SummarisationRequest {
    pub text:             String,
    pub source_type:      SourceType,
    pub max_summary_words: Option<u32>,
}

#[derive(Serialize)]
pub struct SummarisationResponse {
    pub source_type:     &'static str,
    pub summary:         String,
    pub key_decisions:   Vec<String>,
    pub action_items:    Vec<String>,
    pub flagged_concerns: Vec<String>,
    pub word_count:      usize,
}

// ── Handler ───────────────────────────────────────────────────────────────────────

pub async fn handle(
    State(state): State<AppState>,
    Json(req):    Json<SummarisationRequest>,
) -> Result<Json<SummarisationResponse>, AppError> {
    let max_words = req.max_summary_words.unwrap_or(300);
    let system    = req.source_type.system_prompt(max_words);
    let kind      = req.source_type.as_str();

    // Use powerful model for long-form summarisation
    let raw = state.ollama
        .chat(&state.settings.powerful_model, &system, &req.text, 2048)
        .await?;

    let json_str = extract_json_obj(&raw);
    let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|_| {
        serde_json::json!({
            "summary": &raw[..raw.len().min(800)],
            "key_decisions": [],
            "action_items": [],
            "flagged_concerns": ["Model returned non-JSON; manual review required"]
        })
    });

    let summary = parsed["summary"].as_str().unwrap_or("").to_owned();
    let word_count = summary.split_whitespace().count();

    Ok(Json(SummarisationResponse {
        source_type: kind,
        word_count,
        summary,
        key_decisions:    str_vec(&parsed["key_decisions"]),
        action_items:     str_vec(&parsed["action_items"]),
        flagged_concerns: str_vec(&parsed["flagged_concerns"]),
    }))
}

fn str_vec(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_owned())).collect())
        .unwrap_or_default()
}
