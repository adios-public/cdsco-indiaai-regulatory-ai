#[derive(Debug, Clone)]
pub struct Settings {
    pub ollama_base_url:     String,
    pub default_model:       String,
    pub powerful_model:      String,
    pub embed_model:         String,
    pub api_host:            String,
    pub api_port:            u16,
    pub anonymisation_salt:  String,
    pub pseudo_token_prefix: String,
}

impl Settings {
    pub fn from_env() -> Self {
        Self {
            ollama_base_url:     env("OLLAMA_BASE_URL",     "http://localhost:11434"),
            default_model:       env("DEFAULT_MODEL",       "gajendra:latest"),
            powerful_model:      env("POWERFUL_MODEL",      "qwen3.6:latest"),
            embed_model:         env("EMBED_MODEL",         "nomic-embed-text:latest"),
            api_host:            env("API_HOST",            "0.0.0.0"),
            api_port:            std::env::var("API_PORT").ok()
                                     .and_then(|v| v.parse().ok())
                                     .unwrap_or(8000),
            anonymisation_salt:  env("ANONYMISATION_SALT",  "default-change-in-prod"),
            pseudo_token_prefix: env("PSEUDO_TOKEN_PREFIX", "TOK"),
        }
    }
}

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}
