//! Monitoring: the request log, the monitor summary and Prometheus metrics.

use std::fmt::Write as _;

use axum::extract::{Query, Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct Limit {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

/// Records every request: the monitor's log and the error counter both read
/// this, so a route handler does not have to remember to log itself. Token
/// counts come from the counters the handlers bump.
pub async fn record_mw(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let started = std::time::Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16();
    let _ = state
        .record(crate::state::RequestRecord {
            id: 0,
            method,
            path,
            status,
            ms: started.elapsed().as_secs_f64() * 1000.0,
            prompt_tokens: 0,
            gen_tokens: 0,
            at_ms: state.uptime_ms(),
            streamed: false,
            model: None,
        })
        .await;
    response
}

/// `/api/requests` — the most recent requests, newest first.
pub async fn requests(State(state): State<AppState>, Query(q): Query<Limit>) -> Response {
    let limit = q.limit.clamp(1, 200);
    Json(serde_json::json!({"requests": state.requests(limit).await})).into_response()
}

/// `/api-monitor` — the counters the web dashboard shows.
pub async fn monitor(State(state): State<AppState>) -> Response {
    let c = state.counters().await;
    let recent = state.requests(20).await;
    let gen_s = if c.gen_ms > 0.0 { c.gen_ms / 1000.0 } else { 0.0 };
    Json(serde_json::json!({
        "uptime_ms": state.uptime_ms(),
        "backend": state.backend().name(),
        "requests": c.requests,
        "errors": c.errors,
        "chat_completions": c.chat_completions,
        "completions": c.completions,
        "messages": c.messages,
        "responses": c.responses,
        "streamed": c.streamed,
        "prompt_tokens": c.prompt_tokens,
        "gen_tokens": c.gen_tokens,
        "gen_seconds": gen_s,
        "gen_tokens_per_s": if gen_s > 0.0 { c.gen_tokens as f64 / gen_s } else { 0.0 },
        "recent": recent,
    }))
    .into_response()
}

/// `/metrics` — the Prometheus text exposition format.
pub async fn metrics(State(state): State<AppState>) -> Response {
    let c = state.counters().await;
    let backend = state.backend().name();
    let loaded = state.loaded_opt().await;
    let mut out = String::new();
    let mut line = |name: &str, help: &str, kind: &str, value: f64, labels: &str| {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        if labels.is_empty() {
            let _ = writeln!(out, "{name} {value}");
        } else {
            let _ = writeln!(out, "{name}{{{labels}}} {value}");
        }
    };
    line(
        "strata_uptime_seconds",
        "Seconds since the server started.",
        "gauge",
        state.uptime_ms() as f64 / 1000.0,
        "",
    );
    line(
        "strata_requests_total",
        "Requests served.",
        "counter",
        c.requests as f64,
        "",
    );
    line(
        "strata_errors_total",
        "Requests that answered with a 4xx or 5xx.",
        "counter",
        c.errors as f64,
        "",
    );
    line(
        "strata_chat_completions_total",
        "Chat completions served.",
        "counter",
        c.chat_completions as f64,
        "",
    );
    line(
        "strata_completions_total",
        "Text completions served.",
        "counter",
        c.completions as f64,
        "",
    );
    line(
        "strata_messages_total",
        "Anthropic messages served.",
        "counter",
        c.messages as f64,
        "",
    );
    line(
        "strata_responses_total",
        "Responses API calls served.",
        "counter",
        c.responses as f64,
        "",
    );
    line(
        "strata_prompt_tokens_total",
        "Prompt tokens processed.",
        "counter",
        c.prompt_tokens as f64,
        "",
    );
    line(
        "strata_generated_tokens_total",
        "Tokens generated.",
        "counter",
        c.gen_tokens as f64,
        "",
    );
    line(
        "strata_model_loaded",
        "1 when a model is loaded.",
        "gauge",
        if loaded.is_some() { 1.0 } else { 0.0 },
        "",
    );
    if let Some(l) = &loaded {
        line(
            "strata_model_bytes",
            "Bytes mapped by the model file.",
            "gauge",
            l.mapped_bytes as f64,
            &format!("model=\"{}\"", l.name),
        );
        line(
            "strata_model_layers",
            "Layers in the loaded model.",
            "gauge",
            l.engine.geometry.n_layers as f64,
            &format!("model=\"{}\"", l.name),
        );
    }
    line(
        "strata_backend_info",
        "The backend in use.",
        "gauge",
        1.0,
        &format!("backend=\"{backend}\""),
    );
    Response::builder()
        .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        .body(axum::body::Body::from(out))
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}
