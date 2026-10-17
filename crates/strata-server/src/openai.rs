//! The OpenAI-compatible surface: `/v1/models`, `/v1/chat/completions` and
//! `/v1/completions`, with SSE streaming.
//!
//! The shapes are the ones every OpenAI client expects: `choices[].message`,
//! `delta.content` on the stream, `usage` with prompt and completion counts,
//! and `finish_reason` of `stop` or `length`.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use strata_core::generate::FinishReason;
use strata_tokenizer::Message;

use crate::error::Error;
use crate::generate::{finish_reason, stop_tokens, stream_completion, CompletionRequest, SamplerKnobs};
use crate::state::AppState;

/// One message of a chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Present on assistant turns that called a tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A chat completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    /// A string or a list of strings.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Not an OpenAI field, but every local server has it.
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

/// A text completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequestWire {
    #[serde(default)]
    pub model: Option<String>,
    /// A string or a list of token-ish strings, as OpenAI allows.
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

/// One SSE chunk of a chat completion stream.
#[derive(Debug, Serialize)]
struct ChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A request id: a timestamp plus a process-wide counter. Clients only need
/// it to be unique per response, so no UUID crate is involved.
fn id(prefix: &str) -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = now();
    format!("{prefix}-{t}-{n:04x}")
}

/// `stop` arrives as a string or a list of strings.
fn stop_list(v: Option<&serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => {
            a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()
        }
        _ => Vec::new(),
    }
}

/// `/v1/chat/completions`
pub async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let started = std::time::Instant::now();
    if req.messages.is_empty() {
        return Error::BadRequest {
            message: "messages must not be empty".to_string(),
        }
        .into_response();
    }
    let loaded = match state.loaded().await {
        Ok(l) => l,
        Err(e) => return e.into_response(),
    };
    if let Some(m) = &req.model {
        if *m != loaded.name && !m.is_empty() && *m != "strata" {
            return Error::ModelNotFound { model: m.clone() }.into_response();
        }
    }

    let knobs = SamplerKnobs {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: req.repetition_penalty,
        seed: req.seed,
    };
    let max_tokens = req.max_tokens.unwrap_or(512);
    let stops = stop_list(req.stop.as_ref());
    let request = CompletionRequest {
        messages: req
            .messages
            .iter()
            .map(|m| Message {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
        raw: None,
        max_tokens,
        sampler: knobs.sampler(),
        stop: stop_tokens(&loaded, &stops),
    };
    let model_name = loaded.name.clone();
    let stream = req.stream;
    let chat_id = id("chatcmpl");

    let events = match stream_completion(&state, request).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };

    if stream {
        let model = model_name.clone();
        let cid = chat_id.clone();
        let sse = Sse::new(
            events
                .map(move |ev| {
                    let chunk = ChatChunk {
                        id: cid.clone(),
                        object: "chat.completion.chunk",
                        created: now(),
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: if ev.text.is_empty() {
                                    None
                                } else {
                                    Some(ev.text.clone())
                                },
                            },
                            finish_reason: ev.finish.map(finish_reason),
                        }],
                    };
                    Ok::<Event, std::convert::Infallible>(
                        Event::default()
                            .json_data(chunk)
                            .unwrap_or_else(|_| Event::default()),
                    )
                })
                .chain(futures_util::stream::once(async {
                    Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
                })),
        )
        .keep_alive(KeepAlive::default());
        state
            .count(|c| {
                c.chat_completions += 1;
                c.streamed += 1;
            })
            .await;
        return sse.into_response();
    }

    // Non-streaming: collect every chunk into one answer.
    let mut text = String::new();
    let mut prompt_tokens = 0u32;
    let mut gen_tokens = 0u32;
    let mut reason = FinishReason::Length;
    let mut events = Box::pin(events);
    while let Some(ev) = events.next().await {
        text.push_str(&ev.text);
        prompt_tokens = ev.prompt_tokens;
        if ev.gen_tokens > 0 {
            gen_tokens = ev.gen_tokens;
        }
        if let Some(f) = ev.finish {
            reason = f;
        }
    }
    state.count(|c| c.chat_completions += 1).await;
    let body = ChatResponse {
        id: chat_id,
        object: "chat.completion",
        created: now(),
        model: model_name.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
                name: None,
            },
            finish_reason: finish_reason(reason).to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: gen_tokens,
            total_tokens: prompt_tokens + gen_tokens,
        },
    };
    let status = 200;
    let _ = state
        .record(crate::state::RequestRecord {
            id: 0,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            status,
            ms: started.elapsed().as_secs_f64() * 1000.0,
            prompt_tokens,
            gen_tokens,
            at_ms: state.uptime_ms(),
            streamed: false,
            model: Some(model_name),
        })
        .await;
    Json(body).into_response()
}

/// `/v1/completions`
pub async fn completions(State(state): State<AppState>, Json(req): Json<CompletionRequestWire>) -> Response {
    let loaded = match state.loaded().await {
        Ok(l) => l,
        Err(e) => return e.into_response(),
    };
    let raw = match &req.prompt {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(""),
        _ => {
            return Error::BadRequest {
                message: "prompt must be a string or a list of strings".to_string(),
            }
            .into_response()
        }
    };
    let knobs = SamplerKnobs {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        ..SamplerKnobs::default()
    };
    let stops = stop_list(req.stop.as_ref());
    let request = CompletionRequest {
        messages: Vec::new(),
        raw: Some(raw),
        max_tokens: req.max_tokens.unwrap_or(256),
        sampler: knobs.sampler(),
        stop: stop_tokens(&loaded, &stops),
    };
    let model_name = loaded.name.clone();
    let events = match stream_completion(&state, request).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };
    let mut text = String::new();
    let mut prompt_tokens = 0u32;
    let mut gen_tokens = 0u32;
    let mut reason = FinishReason::Length;
    let mut events = Box::pin(events);
    while let Some(ev) = events.next().await {
        text.push_str(&ev.text);
        prompt_tokens = ev.prompt_tokens;
        if ev.gen_tokens > 0 {
            gen_tokens = ev.gen_tokens;
        }
        if let Some(f) = ev.finish {
            reason = f;
        }
    }
    let body = CompletionResponse {
        id: id("cmpl"),
        object: "text_completion",
        created: now(),
        model: model_name,
        choices: vec![TextChoice {
            index: 0,
            text,
            finish_reason: finish_reason(reason).to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: gen_tokens,
            total_tokens: prompt_tokens + gen_tokens,
        },
    };
    state.count(|c| c.completions += 1).await;
    Json(body).into_response()
}

/// `/v1/models`
pub async fn models(State(state): State<AppState>) -> Response {
    let loaded = state.loaded_opt().await;
    let data = match &loaded {
        Some(l) => vec![serde_json::json!({
            "id": l.name,
            "object": "model",
            "created": now(),
            "owned_by": "strata",
            "strata": {
                "family": l.family,
                "architecture": l.architecture,
                "context_length": l.n_ctx,
                "backend": l.backend_name,
            },
        })],
        None => Vec::new(),
    };
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}
