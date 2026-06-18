use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Inference error: {0}")]
    Inference(#[from] anyhow::Error),
    #[allow(dead_code)]
    #[error("Bad request: {0}")]
    BadRequest(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            AppError::Inference(e)   => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::BadRequest(m)  => (StatusCode::BAD_REQUEST,           m.clone()),
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}
