//! Two capabilities:
//!  1. Completeness checker  — rule engine against CDSCO mandatory field schemas
//!  2. Document comparator   — line diff + nomic-embed-text semantic similarity
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};

use crate::{error::AppError, AppState};

// ── Shared field schema ───────────────────────────────────────────────────────────

struct FieldDef { name: &'static str, section: &'static str, mandatory: bool }

const NDA_FIELDS: &[FieldDef] = &[
    FieldDef { name: "applicant_name",    section: "Administrative", mandatory: true  },
    FieldDef { name: "drug_substance_name",section: "Drug",          mandatory: true  },
    FieldDef { name: "proposed_indication",section: "Clinical",      mandatory: true  },
    FieldDef { name: "dosage_form",        section: "Drug",          mandatory: true  },
    FieldDef { name: "route_of_administration", section: "Drug",     mandatory: true  },
    FieldDef { name: "manufacturing_site", section: "Manufacturing", mandatory: true  },
    FieldDef { name: "clinical_trial_data",section: "Clinical",      mandatory: true  },
    FieldDef { name: "safety_data",        section: "Safety",        mandatory: true  },
    FieldDef { name: "proposed_labelling", section: "Labelling",     mandatory: false },
];

const SAE_FIELDS: &[FieldDef] = &[
    FieldDef { name: "case_id",             section: "Administrative", mandatory: true  },
    FieldDef { name: "patient_age",         section: "Patient",       mandatory: true  },
    FieldDef { name: "patient_sex",         section: "Patient",       mandatory: true  },
    FieldDef { name: "suspect_drug",        section: "Drug",          mandatory: true  },
    FieldDef { name: "event_description",   section: "Event",         mandatory: true  },
    FieldDef { name: "event_onset_date",    section: "Event",         mandatory: true  },
    FieldDef { name: "outcome",             section: "Event",         mandatory: true  },
    FieldDef { name: "causality_assessment",section: "Assessment",    mandatory: true  },
    FieldDef { name: "reporter_name",       section: "Reporter",      mandatory: true  },
    FieldDef { name: "concomitant_medications",section:"Drug",        mandatory: false },
];

const CT_FIELDS: &[FieldDef] = &[
    FieldDef { name: "protocol_number",        section: "Administrative", mandatory: true },
    FieldDef { name: "sponsor_name",           section: "Administrative", mandatory: true },
    FieldDef { name: "investigational_product",section: "Product",        mandatory: true },
    FieldDef { name: "study_phase",            section: "Study Design",   mandatory: true },
    FieldDef { name: "primary_endpoint",       section: "Study Design",   mandatory: true },
    FieldDef { name: "sample_size",            section: "Study Design",   mandatory: true },
    FieldDef { name: "ethics_approval",        section: "Regulatory",     mandatory: true },
    FieldDef { name: "informed_consent_process",section:"Regulatory",     mandatory: true },
];

// ── Completeness types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaType { NewDrugApplication, ClinicalTrial, SaeReport }

#[derive(Deserialize)]
pub struct CompletenessRequest {
    pub document:             serde_json::Value,
    pub schema_type:          SchemaType,
    pub flag_inconsistencies: Option<bool>,
}

#[derive(Serialize)]
pub struct MissingField {
    pub field:    String,
    pub section:  String,
    pub severity: &'static str,
    pub reason:   String,
}

#[derive(Serialize)]
pub struct CompletenessResponse {
    pub schema_type:            &'static str,
    pub is_complete:            bool,
    pub completeness_score:     f32,
    pub missing_fields:         Vec<MissingField>,
    pub inconsistencies:        Vec<String>,
    pub review_recommendation:  String,
}

// ── Completeness handler ──────────────────────────────────────────────────────────

pub async fn handle_assess(
    State(_state): State<AppState>,
    Json(req):     Json<CompletenessRequest>,
) -> Result<Json<CompletenessResponse>, AppError> {
    let (fields, kind) = match req.schema_type {
        SchemaType::NewDrugApplication => (NDA_FIELDS, "new_drug_application"),
        SchemaType::SaeReport          => (SAE_FIELDS, "sae_report"),
        SchemaType::ClinicalTrial      => (CT_FIELDS,  "clinical_trial"),
    };

    let mut missing: Vec<MissingField> = Vec::new();
    for f in fields {
        let present = req.document.get(f.name)
            .map(|v| !v.is_null() && v.as_str().map(|s| !s.trim().is_empty()).unwrap_or(true))
            .unwrap_or(false);
        if !present {
            missing.push(MissingField {
                field:    f.name.to_owned(),
                section:  f.section.to_owned(),
                severity: if f.mandatory { "mandatory" } else { "recommended" },
                reason:   format!("'{}' is absent or empty", f.name),
            });
        }
    }

    let mandatory_total = fields.iter().filter(|f| f.mandatory).count();
    let mandatory_miss  = missing.iter().filter(|m| m.severity == "mandatory").count();
    let score = if mandatory_total == 0 { 1.0 }
                else { 1.0 - (mandatory_miss as f32 / mandatory_total as f32) };

    let inconsistencies = if req.flag_inconsistencies.unwrap_or(true) {
        check_inconsistencies(&req.document, kind)
    } else { Vec::new() };

    let recommendation = if score == 1.0 && inconsistencies.is_empty() {
        "ACCEPT for technical screening".to_owned()
    } else if score >= 0.8 {
        "QUERY applicant — recommended fields missing".to_owned()
    } else {
        "RETURN to applicant — mandatory fields incomplete".to_owned()
    };

    Ok(Json(CompletenessResponse {
        schema_type: kind,
        is_complete: score == 1.0 && inconsistencies.is_empty(),
        completeness_score: (score * 1000.0).round() / 1000.0,
        missing_fields: missing,
        inconsistencies,
        review_recommendation: recommendation,
    }))
}

fn check_inconsistencies(doc: &serde_json::Value, kind: &str) -> Vec<String> {
    let mut issues = Vec::new();
    if kind == "sae_report" {
        let onset  = doc["event_onset_date"].as_str().unwrap_or("");
        let report = doc["report_date"].as_str().unwrap_or("");
        if !onset.is_empty() && !report.is_empty() && onset > report {
            issues.push("event_onset_date is after report_date — chronology inconsistency".to_owned());
        }
    }
    issues
}

// ── Comparison types ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ComparisonRequest {
    pub document_v1:          String,
    pub document_v2:          String,
    pub highlight_substantive: Option<bool>,
}

#[derive(Serialize)]
pub struct DocumentChange {
    pub change_type:   &'static str,
    pub original:      String,
    pub revised:       String,
    pub is_substantive: bool,
    pub significance:  &'static str,
}

#[derive(Serialize)]
pub struct ComparisonResponse {
    pub total_changes:      usize,
    pub substantive_changes: usize,
    pub changes:            Vec<DocumentChange>,
    pub similarity_score:   f32,
    pub reviewer_summary:   String,
}

// ── Comparison handler ────────────────────────────────────────────────────────────

pub async fn handle_compare(
    State(state): State<AppState>,
    Json(req):    Json<ComparisonRequest>,
) -> Result<Json<ComparisonResponse>, AppError> {
    let diff = TextDiff::from_lines(&req.document_v1, &req.document_v2);

    let mut changes: Vec<DocumentChange> = Vec::new();
    for group in diff.grouped_ops(3) {
        for op in group {
            for change in diff.iter_changes(&op) {
                let (ctype, original, revised) = match change.tag() {
                    ChangeTag::Delete => ("deletion",  change.value().to_owned(), String::new()),
                    ChangeTag::Insert => ("addition",  String::new(), change.value().to_owned()),
                    ChangeTag::Equal  => continue,
                };
                let is_substantive = !original.trim().is_empty() || revised.len() > 20;
                let significance   = if revised.len() > 100 { "high" }
                                     else if is_substantive  { "medium" }
                                     else                    { "low" };
                changes.push(DocumentChange { change_type: ctype, original, revised, is_substantive, significance });
            }
        }
    }

    // Semantic similarity via nomic-embed-text
    let sim = semantic_sim(&state, &req.document_v1, &req.document_v2).await;

    let substantive = changes.iter().filter(|c| c.is_substantive).count();
    let summary = format!(
        "{} total changes ({} substantive). Similarity: {:.0}%. {}",
        changes.len(), substantive, sim * 100.0,
        if substantive > 0 { "Major revisions — reviewer attention required." }
        else { "Minor edits only." }
    );

    Ok(Json(ComparisonResponse {
        total_changes: changes.len(),
        substantive_changes: substantive,
        changes,
        similarity_score: (sim * 1000.0).round() / 1000.0,
        reviewer_summary: summary,
    }))
}

async fn semantic_sim(state: &AppState, a: &str, b: &str) -> f32 {
    let model = &state.settings.embed_model;
    let (ea, eb) = tokio::join!(
        state.ollama.embed(model, a),
        state.ollama.embed(model, b),
    );
    match (ea, eb) {
        (Ok(va), Ok(vb)) => cosine(&va, &vb),
        _ => 0.5, // fallback
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot:   f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { 0.0 } else { dot / (mag_a * mag_b) }
}
