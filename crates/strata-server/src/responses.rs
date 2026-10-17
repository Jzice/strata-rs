//! The Responses API (`/v1/responses`): a single call that takes an `input`
//! (a string or a list of message blocks) and returns a `message` output with
//! a status, plus a stream of `response.output_text.delta` events.

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

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// A string, or a list of `{role, content}` blocks.
    pub input: serde_json::Value,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub instructions: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<OutputItem>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
pub struct OutputItem {
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<OutputContent>,
}

#[derive(Debug, Serialize)]
pub struct OutputContent {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Turns the `input` field into chat messages.
fn messages_of(input: &serde_json::Value, instructions: Option<&String>) -> Vec<Message> {
    let mut out = Vec::new();
    if let Some(i) = instructions {
        out.push(Message {
            role: "system".to_string(),
            content: i.clone(),
        });
    }
    match input {
        serde_json::Value::String(s) => out.push(Message {
            role: "user".to_string(),
            content: s.clone(),
        }),
        serde_json::Value::Array(items) => {
            for it in items {
                let role = it
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = match it.get("content") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(blocks)) => blocks
                        .iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                out.push(Message { role, content });
            }
        }
        _ => {}
    }
    out
}

/// `/v1/responses`
pub async fn responses(State(state): State<AppState>, Json(req): Json<ResponsesRequest>) -> Response {
    let loaded = match state.loaded().await {
        Ok(l) => l,
        Err(e) => return e.into_response(),
    };
    let msgs = messages_of(&req.input, req.instructions.as_ref());
    if msgs.is_empty() {
        return Error::BadRequest {
            message: "input must be a string or a list of message blocks".to_string(),
        }
        .into_response();
    }
    let knobs = SamplerKnobs {
        temperature: req.temperature,
        top_p: req.top_p,
        ..SamplerKnobs::default()
    };
    let request = CompletionRequest {
        messages: msgs,
        raw: None,
        max_tokens: req.max_output_tokens.unwrap_or(512) as usize,
        sampler: knobs.sampler(),
        stop: stop_tokens(&loaded, &[]),
    };
    let model_name = loaded.name.clone();
    let stream = req.stream;
    let events = match stream_completion(&state, request).await {
        Ok(s) => s,
        Err(e) => return e.into_response(),
    };
    state.count(|c| c.responses += 1).await;

    if stream {
        let model = model_name;
        let sse = Sse::new(
            events
                .map(move |ev| {
                    let mut out: Vec<Result<Event, std::convert::Infallible>> = Vec::new();
                    if !ev.text.is_empty() {
                        out.push(Ok(ev_json(
                            "response.output_text.delta",
                            serde_json::json!({
                                "type": "response.output_text.delta",
                                "delta": ev.text,
                            }),
                        )));
                    }
                    if let Some(f) = ev.finish {
                        out.push(Ok(ev_json(
                            "response.completed",
                            serde_json::json!({
                                "type": "response.completed",
                                "response": {
                                    "id": "resp_strata",
                                    "object": "response",
                                    "model": model,
                                    "status": if finish_reason(f) == "length" {
                                        "incomplete"
                                    } else {
                                        "completed"
                                    },
                                    "usage": {
                                        "input_tokens": ev.prompt_tokens,
                                        "output_tokens": ev.gen_tokens,
                                        "total_tokens": ev.prompt_tokens + ev.gen_tokens,
                                    },
                                }
                            }),
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
    Json(ResponsesResponse {
        id: "resp_strata".to_string(),
        object: "response",
        created_at: now(),
        model: model_name,
        status: if reason == FinishReason::Length {
            "incomplete"
        } else {
            "completed"
        },
        output: vec![OutputItem {
            r#type: "message",
            role: "assistant",
            content: vec![OutputContent {
                r#type: "output_text",
                text,
            }],
        }],
        usage: ResponsesUsage {
            input_tokens: prompt_tokens,
            output_tokens: gen_tokens,
            total_tokens: prompt_tokens + gen_tokens,
        },
    })
    .into_response()
}

/// An SSE event, or the empty event if the body cannot be serialized.
fn ev_json(name: &'static str, data: serde_json::Value) -> Event {
    Event::default().event(name).json_data(data).unwrap_or_default()
}
