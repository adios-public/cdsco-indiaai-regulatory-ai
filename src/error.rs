use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Inference error: {0}")]
    Inference(#[from] anyhow::Error),
    #[error("Bad request: {0}")]
    BadRequest(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST,            self.to_string()),
            _                       => (StatusCode::INTERNAL_SERVER_ERROR,   self.to_string()),
        };
        (status, Json(serde_json::json!({"error": msg}))).into_response()
    }
}
