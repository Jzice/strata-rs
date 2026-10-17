//! The Model Context Protocol surface.
//!
//! MCP is JSON-RPC 2.0: `initialize`, `tools/list` and `tools/call`. Two
//! tools are exposed — `strata_status` and `strata_complete` — so an agent
//! can ask the server what is loaded and get an answer from the model.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::error::Error;
use crate::generate::{stop_tokens, stream_completion, CompletionRequest, SamplerKnobs};
use crate::state::AppState;

const PROTOCOL: &str = "2025-06-18";

/// A JSON-RPC request.
#[derive(Debug, serde::Deserialize)]
pub struct Rpc {
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// `/mcp` — the whole protocol on one endpoint.
pub async fn mcp(State(state): State<AppState>, Json(rpc): Json<Rpc>) -> Response {
    let id = rpc.id.clone().unwrap_or(Value::Null);
    let result = match rpc.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {"listChanged": false}},
            "serverInfo": {"name": "strata", "version": env!("CARGO_PKG_VERSION")},
        })),
        "tools/list" => Ok(json!({"tools": tools()})),
        "tools/call" => call(&state, &rpc.params).await,
        "notifications/initialized" | "ping" => Ok(json!({})),
        other => Err(Error::BadRequest {
            message: format!("unknown method {other}"),
        }),
    };
    match result {
        Ok(r) => Json(json!({"jsonrpc": "2.0", "id": id, "result": r})).into_response(),
        Err(e) => Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32602, "message": e.to_string()},
        }))
        .into_response(),
    }
}

fn tools() -> Vec<Value> {
    vec![
        json!({
            "name": "strata_status",
            "description": "What the Strata server has loaded: the model, its geometry and the backend.",
            "inputSchema": {"type": "object", "properties": {}},
        }),
        json!({
            "name": "strata_complete",
            "description": "Ask the loaded model a question and get its answer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt to complete."},
                    "max_tokens": {"type": "integer", "description": "How many tokens to generate."},
                    "temperature": {"type": "number", "description": "Sampling temperature."},
                },
                "required": ["prompt"],
            },
        }),
    ]
}

async fn call(state: &AppState, params: &Value) -> Result<Value, Error> {
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| Error::BadRequest {
            message: "tools/call needs a tool name".to_string(),
        })?;
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    match name {
        "strata_status" => {
            let loaded = state.loaded_opt().await;
            let c = state.counters().await;
            let text = match &loaded {
                Some(l) => format!(
                    "model {} ({}, {} layers, {} of {} experts, context {}) on {}, {} requests served",
                    l.name,
                    l.family,
                    l.engine.geometry.n_layers,
                    l.engine.geometry.n_expert_used,
                    l.engine.geometry.n_expert,
                    l.n_ctx,
                    l.backend_name,
                    c.requests
                ),
                None => "no model is loaded".to_string(),
            };
            Ok(json!({"content": [{"type": "text", "text": text}], "isError": false}))
        }
        "strata_complete" => {
            let prompt = args
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| Error::BadRequest {
                    message: "strata_complete needs a prompt".to_string(),
                })?;
            let loaded = state.loaded().await?;
            let knobs = SamplerKnobs {
                temperature: args
                    .get("temperature")
                    .and_then(serde_json::Value::as_f64)
                    .map(|v| v as f32),
                ..SamplerKnobs::default()
            };
            let request = CompletionRequest {
                messages: Vec::new(),
                raw: Some(prompt.to_string()),
                max_tokens: args
                    .get("max_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(256) as usize,
                sampler: knobs.sampler(),
                stop: stop_tokens(&loaded, &[]),
            };
            let mut text = String::new();
            let mut events = Box::pin(stream_completion(state, request).await?);
            use futures_util::StreamExt;
            while let Some(ev) = events.next().await {
                text.push_str(&ev.text);
            }
            Ok(json!({"content": [{"type": "text", "text": text}], "isError": false}))
        }
        other => Err(Error::BadRequest {
            message: format!("unknown tool {other}"),
        }),
    }
}
