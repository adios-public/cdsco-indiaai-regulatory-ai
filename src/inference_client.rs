//! HTTP client for adios-inference gateway (port 8010).
//!
//! Mirrors the REST contract of `services/adios-inference/main.py` so this
//! module can be lifted directly into `plugins/ai/adios-inference` in the
//! AdiOS Platform without API changes.
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct InferenceClient {
    http:     reqwest::Client,
    base_url: String,
}

impl InferenceClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http:     reqwest::Client::builder()
                          .timeout(std::time::Duration::from_secs(180))
                          .build()
                          .expect("reqwest client"),
            base_url: base_url.trim_end_matches('/').to_owned(),
        }
    }

    /// Translate text between any two of the 22 Indian scheduled languages.
    /// source/target: BCP-47+script e.g. "eng_Latn", "hin_Deva", "tam_Taml"
    pub async fn translate(
        &self,
        text:        &str,
        source_lang: &str,
        target_lang: &str,
    ) -> anyhow::Result<String> {
        let resp: TranslateResponse = self.http
            .post(format!("{}/translate", self.base_url))
            .json(&TranslateRequest {
                text:        text.to_owned(),
                source_lang: source_lang.to_owned(),
                target_lang: target_lang.to_owned(),
                max_length:  512,
            })
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.translated)
    }

    /// IndicBERT sentence embedding (768-dim) or nomic-embed-text via Ollama.
    pub async fn embed(&self, text: &str, model: &str) -> anyhow::Result<Vec<f32>> {
        let resp: EmbedResponse = self.http
            .post(format!("{}/embed", self.base_url))
            .json(&EmbedRequest { text: text.to_owned(), model: model.to_owned() })
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.embedding)
    }

    /// Generate text via any Ollama model (proxied through adios-inference).
    pub async fn generate(
        &self,
        prompt:     &str,
        system:     Option<&str>,
        model:      &str,
        max_tokens: i32,
    ) -> anyhow::Result<String> {
        let resp: GenerateResponse = self.http
            .post(format!("{}/generate", self.base_url))
            .json(&GenerateRequest {
                prompt:      prompt.to_owned(),
                system:      system.map(|s| s.to_owned()),
                model:       model.to_owned(),
                max_tokens,
                temperature: 0.1,
            })
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.content)
    }

    /// Check if gateway is reachable.
    pub async fn is_healthy(&self) -> bool {
        self.http.get(format!("{}/health", self.base_url))
            .send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

// ── Wire types (mirror Python schemas exactly) ────────────────────────────────

#[derive(Serialize)]
struct TranslateRequest { text: String, source_lang: String, target_lang: String, max_length: u32 }
#[derive(Deserialize)]
struct TranslateResponse { translated: String }

#[derive(Serialize)]
struct EmbedRequest { text: String, model: String }
#[derive(Deserialize)]
struct EmbedResponse { embedding: Vec<f32> }

#[derive(Serialize)]
struct GenerateRequest { prompt: String, system: Option<String>, model: String, max_tokens: i32, temperature: f32 }
#[derive(Deserialize)]
struct GenerateResponse { content: String }
