//! The Anthropic-compatible surface: `/v1/messages`.
//!
//! Anthropic's shape differs from OpenAI's in the details that matter: the
//! content is a list of blocks, the answer is `content[].text`, the stream
//! sends typed events (`content_block_delta`, `message_delta`, ...), and the
//! token counts are named `input_tokens` / `output_tokens`.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use strata_core::generate::FinishReason;
use strata_tokenizer::Message;

use crate::error::Error;
use crate::generate::{stop_tokens, stream_completion, CompletionRequest, SamplerKnobs};
use crate::state::AppState;

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default = "default_max")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
}

fn default_max() -> u32 {
    1024
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// A string or a list of content blocks.
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlock {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Flattens Anthropic's content shapes into plain text.
fn text_of(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn system_text(v: Option<&serde_json::Value>) -> Option<String> {
    match v {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(blocks)) => {
            let joined = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

/// `stop_reason`, in Anthropic's spelling.
fn stop_reason(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Length => "max_tokens",
        FinishReason::Stop => "end_turn",
        FinishReason::Cancelled => "cancelled",
    }
}

/// `/v1/messages`
pub async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    let loaded = match state.loaded().await {
        Ok(l) => l,
        Err(e) => return e.into_response(),
    };
    if req.messages.is_empty() {
        return Error::BadRequest {
            message: "messages must not be empty".to_string(),
        }
        .into_response();
    }

    let mut msgs: Vec<Message> = Vec::new();
    if let Some(sys) = system_text(req.system.as_ref()) {
        msgs.push(Message {
            role: "system".to_string(),
            content: sys,
        });
    }
    for m in &req.messages {
        msgs.push(Message {
            role: m.role.clone(),
            content: text_of(&m.content),
        });
    }

    let knobs = SamplerKnobs {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        ..SamplerKnobs::default()
    };
    let request = CompletionRequest {
        messages: msgs,
        raw: None,
        max_tokens: req.max_tokens as usize,
        sampler: knobs.sampler(),
        stop: stop_tokens(&loaded, &req.stop_sequences),
    };
    let model_name = loaded.name.clone();
    let stream = req.stream;
    let events = match stream_completion(&state, request).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };
    state.count(|c| c.messages += 1).await;

    if stream {
        let model = model_name;
        let sse = Sse::new(
            events
                .enumerate()
                .map(move |(i, ev)| {
                    // Anthropic opens with message_start, then one delta per
                    // chunk, then message_stop.
                    let mut out: Vec<Result<Event, std::convert::Infallible>> = Vec::new();
                    if i == 0 {
                        out.push(Ok(event(
                            "message_start",
                            serde_json::json!({
                                "type": "message_start",
                                "message": {
                                    "id": "msg_strata",
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model,
                                    "content": [],
                                    "usage": {"input_tokens": ev.prompt_tokens, "output_tokens": 0},
                                }
                            }),
                        )));
                        out.push(Ok(event(
                            "content_block_start",
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            }),
                        )));
                    }
                    if !ev.text.is_empty() {
                        out.push(Ok(event(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": ev.text},
                            }),
                        )));
                    }
                    if let Some(f) = ev.finish {
                        out.push(Ok(event(
                            "content_block_stop",
                            serde_json::json!({"type": "content_block_stop", "index": 0}),
                        )));
                        out.push(Ok(event(
                            "message_delta",
                            serde_json::json!({
                                "type": "message_delta",
                                "delta": {"stop_reason": stop_reason(f)},
                                "usage": {"output_tokens": ev.gen_tokens},
                            }),
                        )));
                        out.push(Ok(event(
                            "message_stop",
                            serde_json::json!({"type": "message_stop"}),
                        )));
                    }
                    futures_util::stream::iter(out)
                })
                .flatten(),
        )
        .keep_alive(KeepAlive::default());
        return sse.into_response();
    }

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
    Json(MessagesResponse {
        id: "msg_strata".to_string(),
        r#type: "message",
        role: "assistant",
        model: model_name,
        content: vec![ContentBlock { r#type: "text", text }],
        stop_reason: stop_reason(reason).to_string(),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: prompt_tokens,
            output_tokens: gen_tokens,
        },
    })
    .into_response()
}

/// An SSE event, or the empty event if the body cannot be serialized.
fn event(name: &'static str, data: serde_json::Value) -> Event {
    Event::default().event(name).json_data(data).unwrap_or_default()
}
