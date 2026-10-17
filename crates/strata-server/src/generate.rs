//! The bridge between the blocking engine and the async HTTP layer: render a
//! prompt, run the generation on a blocking task, and hand the chunks back as
//! an async stream.

use std::fmt::Write as _;
use std::sync::Arc;

use futures_util::stream::Stream;
use strata_core::generate::{eos_ids, generate, Chunk, FinishReason, GenerateParams, Stop};
use strata_core::sampler::SamplerParams;
use strata_tokenizer::{ChatTemplate, Message, Tokenizer};

use crate::error::Result;
use crate::state::{AppState, Loaded};

/// What a completion route asks for, before it is shaped into an API response.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Messages to render through the chat template, or `None` for raw text.
    pub messages: Vec<Message>,
    /// Raw text used when `messages` is empty (the `/v1/completions` path).
    pub raw: Option<String>,
    pub max_tokens: usize,
    pub sampler: SamplerParams,
    pub stop: Stop,
}

/// One event the stream yields.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub text: String,
    pub token: u32,
    pub is_stop: bool,
    pub finish: Option<FinishReason>,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub tokens_per_s: f64,
}

/// Renders the prompt: the chat template when the file has one, otherwise the
/// messages joined with a speaker line.
pub fn render_prompt(loaded: &Loaded, req: &CompletionRequest) -> String {
    if let Some(raw) = &req.raw {
        return raw.clone();
    }
    if let Some(t) = loaded
        .tokenizer
        .as_ref()
        .and_then(|_| ChatTemplate::from_gguf(loaded.engine.model.meta()))
    {
        if let Ok(s) = t.render(&req.messages, true) {
            return s;
        }
    }
    // No template (or a template this renderer cannot run): a plain transcript.
    let mut out = String::new();
    for m in &req.messages {
        let _ = writeln!(out, "{}: {}", m.role, m.content);
    }
    out.push_str("assistant:");
    out
}

/// Encodes `text` with the model's tokenizer, or falls back to a byte-ish id
/// sequence so a file without a vocabulary still serves.
pub fn encode(loaded: &Loaded, text: &str) -> Result<Vec<u32>> {
    if let Some(tok) = loaded.engine.tokenizer.as_ref() {
        let mut tok = tok.clone();
        return tok
            .encode(text, true)
            .map_err(|e| crate::error::Error::BadRequest {
                message: format!("tokenizing failed: {e}"),
            });
    }
    Ok(text.bytes().take(64).map(|b| u32::from(b) % 256).collect())
}

/// Runs the generation and streams its chunks.
///
/// The engine is a blocking, single-sequence engine, so the work happens on a
/// blocking task while the chunks travel over a channel.
pub async fn stream_completion(
    state: &AppState,
    req: CompletionRequest,
) -> Result<impl Stream<Item = StreamEvent> + Send + 'static> {
    let loaded = state.loaded().await?;
    let backend = state.backend();
    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    tokio::task::spawn_blocking(move || {
        // Held for the whole generation: the engine runs one sequence at a
        // time, and this task owns it until the last chunk is sent.
        let mut session = strata_core::Session::with_backend(Arc::clone(&loaded.engine), backend);
        let prompt = render_prompt(&loaded, &req);
        let ids = match encode(&loaded, &prompt) {
            Ok(ids) => ids,
            Err(e) => {
                tracing::error!("{e}");
                return;
            }
        };
        let params = GenerateParams {
            max_tokens: req.max_tokens,
            sampler: req.sampler.clone(),
            stop: req.stop.clone(),
        };
        let tokenizer = loaded.engine.tokenizer.clone();
        let stats = generate(
            &loaded.engine,
            &mut session,
            &ids,
            &params,
            tokenizer.as_ref(),
            |chunk: &Chunk| {
                let ev = StreamEvent {
                    text: chunk.text.clone(),
                    token: chunk.token,
                    is_stop: chunk.is_stop,
                    finish: None,
                    prompt_tokens: ids.len() as u32,
                    gen_tokens: 0,
                    tokens_per_s: 0.0,
                };
                tx.blocking_send(ev).is_ok()
            },
        );
        let _ = tx.blocking_send(StreamEvent {
            text: String::new(),
            token: 0,
            is_stop: true,
            finish: Some(stats.finish_reason),
            prompt_tokens: stats.prompt_tokens as u32,
            gen_tokens: stats.gen_tokens as u32,
            tokens_per_s: stats.tokens_per_s,
        });
    });

    Ok(futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|ev| (ev, rx))
    }))
}

/// The stop tokens a completion should honour.
pub fn stop_tokens(loaded: &Loaded, extra: &[String]) -> Stop {
    Stop {
        token_ids: eos_ids(loaded.engine.tokenizer.as_ref()),
        strings: extra.to_vec(),
    }
}

/// The sampler from an OpenAI-style request.
#[derive(Debug, Clone, Default)]
pub struct SamplerKnobs {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl SamplerKnobs {
    /// Fills the defaults the CLI uses.
    pub fn sampler(&self) -> SamplerParams {
        let d = SamplerParams::default();
        SamplerParams {
            temperature: self.temperature.unwrap_or(d.temperature),
            top_p: self.top_p.unwrap_or(d.top_p),
            top_k: self.top_k.unwrap_or(d.top_k),
            min_p: self.min_p.unwrap_or(d.min_p),
            penalty_present: self.presence_penalty.unwrap_or(0.0),
            penalty_freq: self.frequency_penalty.unwrap_or(0.0),
            penalty_repeat: self.repetition_penalty.unwrap_or(1.0),
            seed: self.seed.unwrap_or(d.seed),
            ..d
        }
    }
}

/// The finish reason, in OpenAI's spelling.
pub fn finish_reason(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "cancelled",
    }
}

/// Builds a chat template for the loaded model, if the file carries one.
pub fn template(loaded: &Loaded) -> Option<ChatTemplate> {
    ChatTemplate::from_gguf(loaded.engine.model.meta())
}

/// The tokenizer of the loaded model, when the file has one.
pub fn tokenizer(loaded: &Loaded) -> Option<Tokenizer> {
    loaded.engine.tokenizer.clone()
}
