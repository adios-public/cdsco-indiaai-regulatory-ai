//! Sovereign Model Orchestrator
//!
//! Routes every inference task to the cheapest capable model that satisfies
//! the task profile. No external API calls by default — all models run
//! locally via Ollama. Cloud models (Kimi API) are opt-in fallback only
//! and require explicit `allow_cloud: true` in the request.
//!
//! ## Tier ladder (cheapest → most capable)
//!
//! | Tier | Model      | Params | Best for |
//! |------|------------|--------|----------|
//! | T0   | sarvam     | 2B     | Ultra-fast classification, Indic scripts |
//! | T1   | ayurparam  | 2.9B   | Clinical/Ayurveda, Hindi/Sanskrit |
//! | T2   | gajendra   | 7B     | Bilingual generalist, Indian regulatory |
//! | T3   | glm-5.2    | ~9B    | Structured JSON, multi-step reasoning |
//! | T4   | deepseek-r1| 7B     | Logic-heavy, self-correcting |
//! | T5   | qwen3.6    | 36B    | Long-form summarisation, complex reports |
//! | T6   | kimi-k2.5  | ~9B    | Code + reasoning |
//! | CLOUD| Kimi API   | ∞      | Last resort — requires allow_cloud:true |

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::ollama::OllamaClient;

// ── Task Profile ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskDomain {
    Regulatory,
    Clinical,
    Indic,
    Structural,
    General,
    Code,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    Simple,
    Medium,
    Complex,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskProfile {
    pub domain:        TaskDomain,
    pub complexity:    Complexity,
    pub requires_json: bool,
    pub input_tokens:  usize,
    #[serde(default)]
    pub allow_cloud:   bool,
}

// ── Model registry ────────────────────────────────────────────────────────────

pub struct ModelEntry {
    pub tag:           &'static str,
    pub params_b:      f32,
    pub local:         bool,
    pub json_reliable: bool,
    pub indic_capable: bool,
    pub cost_per_1k:   f32,
}

pub const MODEL_REGISTRY: &[ModelEntry] = &[
    ModelEntry { tag: "sarvam:latest",    params_b: 2.0,  local: true,  json_reliable: false, indic_capable: true,  cost_per_1k: 0.0 },
    ModelEntry { tag: "ayurparam:latest", params_b: 2.9,  local: true,  json_reliable: false, indic_capable: true,  cost_per_1k: 0.0 },
    ModelEntry { tag: "gajendra:latest",  params_b: 7.0,  local: true,  json_reliable: false, indic_capable: true,  cost_per_1k: 0.0 },
    ModelEntry { tag: "glm-5.2:latest",   params_b: 9.0,  local: true,  json_reliable: true,  indic_capable: false, cost_per_1k: 0.0 },
    ModelEntry { tag: "deepseek-r1:7b",   params_b: 7.0,  local: true,  json_reliable: true,  indic_capable: false, cost_per_1k: 0.0 },
    ModelEntry { tag: "kimi-k2.5:latest", params_b: 9.0,  local: true,  json_reliable: true,  indic_capable: false, cost_per_1k: 0.0 },
    ModelEntry { tag: "qwen3.6:latest",   params_b: 36.0, local: true,  json_reliable: true,  indic_capable: false, cost_per_1k: 0.0 },
    ModelEntry { tag: "moonshot-v1-32k",  params_b: 999.0,local: false, json_reliable: true,  indic_capable: false, cost_per_1k: 0.012 },
];

// ── Router ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct RouteDecision {
    pub selected_model: String,
    pub tier:           &'static str,
    pub is_local:       bool,
    pub estimated_cost: f32,
    pub rationale:      String,
}

pub fn select_model(profile: &TaskProfile) -> RouteDecision {
    if profile.domain == TaskDomain::Structural {
        return RouteDecision {
            selected_model: "none".into(),
            tier: "T0-rule",
            is_local: true,
            estimated_cost: 0.0,
            rationale: "Structural task — rule engine only, no LLM needed".into(),
        };
    }

    let chosen = MODEL_REGISTRY.iter()
        .filter(|m| {
            if !m.local && !profile.allow_cloud { return false; }
            if profile.requires_json && !m.json_reliable { return false; }
            if profile.domain == TaskDomain::Indic && !m.indic_capable { return false; }
            if profile.complexity == Complexity::Complex && m.params_b < 7.0 { return false; }
            if profile.complexity == Complexity::Medium  && m.params_b < 2.9 { return false; }
            true
        })
        .min_by(|a, b| a.params_b.partial_cmp(&b.params_b).unwrap());

    match chosen {
        Some(m) => RouteDecision {
            selected_model: m.tag.to_owned(),
            tier: tier_label(m),
            is_local: m.local,
            estimated_cost: profile.input_tokens as f32 * m.cost_per_1k / 1000.0,
            rationale: format!(
                "{}B params | json_reliable={} | indic={} | cost/1k=${:.4}",
                m.params_b, m.json_reliable, m.indic_capable, m.cost_per_1k
            ),
        },
        None => RouteDecision {
            selected_model: "qwen3.6:latest".into(),
            tier: "T5-fallback",
            is_local: true,
            estimated_cost: 0.0,
            rationale: "No candidate matched — fallback to qwen3.6".into(),
        },
    }
}

fn tier_label(m: &ModelEntry) -> &'static str {
    match m.tag {
        "sarvam:latest"    => "T0-sarvam",
        "ayurparam:latest" => "T1-ayurparam",
        "gajendra:latest"  => "T2-gajendra",
        "glm-5.2:latest"   => "T3-glm",
        "deepseek-r1:7b"   => "T4-deepseek",
        "kimi-k2.5:latest" => "T5-kimi-local",
        "qwen3.6:latest"   => "T5-qwen",
        _                  => "T6-cloud",
    }
}

// ── Orchestrated inference with fallback ────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct InferenceResult {
    pub model_used:    String,
    pub tier:          String,
    pub content:       String,
    pub latency_ms:    u128,
    pub is_local:      bool,
    pub cost_usd:      f32,
    pub fallback_used: bool,
}

pub async fn run(
    ollama:     &OllamaClient,
    profile:    &TaskProfile,
    system:     &str,
    user:       &str,
    max_tokens: i32,
) -> anyhow::Result<InferenceResult> {
    let decision = select_model(profile);
    let fallback = "qwen3.6:latest";
    let start    = Instant::now();

    match ollama.chat(&decision.selected_model, system, user, max_tokens).await {
        Ok(content) => Ok(InferenceResult {
            model_used:    decision.selected_model,
            tier:          decision.tier.to_owned(),
            content,
            latency_ms:    start.elapsed().as_millis(),
            is_local:      decision.is_local,
            cost_usd:      decision.estimated_cost,
            fallback_used: false,
        }),
        Err(e) => {
            tracing::warn!("Model {} failed ({}), escalating to {}", decision.selected_model, e, fallback);
            let content = ollama.chat(fallback, system, user, max_tokens).await?;
            Ok(InferenceResult {
                model_used:    fallback.to_owned(),
                tier:          "T5-fallback".to_owned(),
                content,
                latency_ms:    start.elapsed().as_millis(),
                is_local:      true,
                cost_usd:      0.0,
                fallback_used: true,
            })
        }
    }
}

// ── HTTP handler for /api/v1/orchestrator/route ───────────────────────────────

use axum::{extract::State, Json};
use crate::AppState;
use crate::error::AppError;

#[derive(Deserialize)]
pub struct RouteRequest {
    pub profile: TaskProfile,
}

#[derive(Serialize)]
pub struct RouteResponse {
    pub decision:         RouteDecision,
    pub available_models: Vec<ModelSummary>,
}

#[derive(Serialize)]
pub struct ModelSummary {
    pub tag:         &'static str,
    pub params_b:    f32,
    pub local:       bool,
    pub cost_per_1k: f32,
}

pub async fn handle_route(
    State(_state): State<AppState>,
    Json(req):     Json<RouteRequest>,
) -> Result<Json<RouteResponse>, AppError> {
    let decision = select_model(&req.profile);
    let available = MODEL_REGISTRY.iter().map(|m| ModelSummary {
        tag:         m.tag,
        params_b:    m.params_b,
        local:       m.local,
        cost_per_1k: m.cost_per_1k,
    }).collect();
    Ok(Json(RouteResponse { decision, available_models: available }))
}
