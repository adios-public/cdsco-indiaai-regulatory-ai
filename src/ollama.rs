//! Typed HTTP client for the local Ollama / AIKosh inference server.
//!
//! Uses the Ollama `/api/chat` endpoint for text generation
//! and `/api/embed` for sentence embeddings (nomic-embed-text).
use anyhow::Context;
use serde::{Deserialize, Serialize};

pub struct OllamaClient {
    client:   reqwest::Client,
    base_url: String,
}

// ── Chat ─────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatReq<'a> {
    model:    &'a str,
    messages: Vec<Msg<'a>>,
    stream:   bool,
    options:  Opts,
}

#[derive(Serialize)]
struct Msg<'a> { role: &'a str, content: &'a str }

#[derive(Serialize)]
struct Opts { num_predict: i32, temperature: f32 }

#[derive(Deserialize)]
struct ChatResp { message: MsgContent }

#[derive(Deserialize)]
struct MsgContent { content: String }

// ── Embed ─────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedReq<'a> { model: &'a str, input: &'a str }

#[derive(Deserialize)]
struct EmbedResp { embeddings: Vec<Vec<f32>> }

// ── Impl ──────────────────────────────────────────────────────────────────────

impl OllamaClient {
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .expect("reqwest client");
        Self { client, base_url }
    }

    /// Send a chat completion to the local Ollama model.
    pub async fn chat(
        &self,
        model:      &str,
        system:     &str,
        user:       &str,
        max_tokens: i32,
    ) -> anyhow::Result<String> {
        let body = ChatReq {
            model,
            messages: vec![
                Msg { role: "system", content: system },
                Msg { role: "user",   content: user },
            ],
            stream:  false,
            options: Opts { num_predict: max_tokens, temperature: 0.1 },
        };
        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send().await
            .context("Ollama /api/chat request")?;
        let data: ChatResp = resp.json().await.context("parse chat response")?;
        Ok(strip_think(&data.message.content))
    }

    /// Get a sentence embedding from nomic-embed-text (or any embed model).
    pub async fn embed(&self, model: &str, text: &str) -> anyhow::Result<Vec<f32>> {
        let body = EmbedReq { model, input: text };
        let resp = self.client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send().await
            .context("Ollama /api/embed request")?;
        let data: EmbedResp = resp.json().await.context("parse embed response")?;
        data.embeddings.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("no embeddings returned"))
    }
}

/// Remove `<think>…</think>` reasoning traces emitted by qwen3.6 / deepseek-r1.
fn strip_think(s: &str) -> String {
    let mut out = s.to_owned();
    loop {
        match (out.find("<think>"), out.find("</think>")) {
            (Some(a), Some(b)) if a < b => {
                out.replace_range(a..b + "</think>".len(), "");
            }
            _ => break,
        }
    }
    out.trim().to_owned()
}

/// Extract the first JSON object `{…}` from a string.
pub fn extract_json_obj(s: &str) -> &str {
    match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if a <= b => &s[a..=b],
        _ => "{}",
    }
}

/// Extract the first JSON array `[…]` from a string.
pub fn extract_json_arr(s: &str) -> &str {
    match (s.find('['), s.rfind(']')) {
        (Some(a), Some(b)) if a <= b => &s[a..=b],
        _ => "[]",
    }
}
