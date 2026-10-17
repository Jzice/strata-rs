//! Chat-template rendering.
//!
//! The model's own Jinja template ships in the GGUF metadata
//! (`tokenizer.chat_template`); fine-tunes change it (Swift 1.5 differs from
//! Qwen3.8-Flash-Next's), so it is read per model rather than assumed. The
//! template is rendered with minijinja against the same context the
//! reference server passes.

use crate::error::{Error, Result};
use minijinja::{context, Environment, Value};
use serde_json::json;

/// One chat message.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// The model's chat template.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    source: String,
}

impl ChatTemplate {
    /// Builds a template from its Jinja source.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// Reads `tokenizer.chat_template` from the GGUF metadata. `None` when
    /// the file carries no template.
    pub fn from_gguf(file: &strata_gguf::GgufFile) -> Option<Self> {
        let source = file.get("tokenizer.chat_template").and_then(|v| v.as_str())?;
        if source.is_empty() {
            return None;
        }
        Some(Self {
            source: source.to_string(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Renders the prompt for a conversation. `add_generation_prompt` opens
    /// the assistant's turn (the trailing `assistant\n`).
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        let env = Environment::new();
        env.render_str(
            &self.source,
            context! {
                messages => Value::from(minijinja::value::Serde(&msgs)),
                add_generation_prompt => add_generation_prompt,
                bos_token => "",
                eos_token => "",
            },
        )
        .map_err(|e| Error::TemplateRender {
            message: e.to_string(),
        })
    }
}
