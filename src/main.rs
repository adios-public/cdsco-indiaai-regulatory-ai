mod config;
mod error;
mod ollama;
mod anonymisation;
mod summarisation;
mod completeness;
mod classification;
mod inspection;
mod orchestrator;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
    Json,
};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, fmt};

pub use config::Settings;
pub use ollama::OllamaClient;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub ollama:   Arc<OllamaClient>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let settings = Arc::new(Settings::from_env());
    let ollama   = Arc::new(OllamaClient::new(settings.ollama_base_url.clone()));
    let state    = AppState { settings: settings.clone(), ollama };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/anonymise",                  post(anonymisation::handle))
        .route("/api/v1/summarise",                  post(summarisation::handle))
        .route("/api/v1/assess-completeness",        post(completeness::handle_assess))
        .route("/api/v1/compare",                    post(completeness::handle_compare))
        .route("/api/v1/classify-sae",               post(classification::handle))
        .route("/api/v1/generate-inspection-report", post(inspection::handle))
        .route("/api/v1/orchestrator/route",         post(orchestrator::handle_route))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("{}:{}", settings.api_host, settings.api_port);
    tracing::info!("AdiOS Regulatory AI listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "service": "adios-regulatory-ai" }))
}
