//! The lifecycle routes: health, status, load and unload.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::Error;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct LoadRequest {
    /// A model file path (or the first shard of one).
    pub model: Option<String>,
    /// A path is also accepted under `path`.
    pub path: Option<String>,
    #[serde(default)]
    pub n_ctx: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub loaded: bool,
    pub model: Option<String>,
    pub backend: String,
    pub uptime_ms: u64,
    pub version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Status {
    pub loaded: bool,
    pub model: Option<ModelStatus>,
    pub backend: String,
    pub bind: String,
    pub port: u16,
    pub uptime_ms: u64,
    pub requests: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelStatus {
    pub name: String,
    pub path: String,
    pub family: String,
    pub architecture: String,
    pub layers: i64,
    pub n_embd: i64,
    pub experts: i64,
    pub experts_used: i64,
    pub context_length: u32,
    pub vocabulary: i64,
    pub quant: String,
    pub mapped_bytes: u64,
    pub loaded_ms: u64,
}

/// `/health`
pub async fn health(State(state): State<AppState>) -> Response {
    let loaded = state.loaded_opt().await;
    Json(Health {
        status: "ok",
        loaded: loaded.is_some(),
        model: loaded.as_ref().map(|l| l.name.clone()),
        backend: state.backend().name().to_string(),
        uptime_ms: state.uptime_ms(),
        version: env!("CARGO_PKG_VERSION"),
    })
    .into_response()
}

/// `/v1/status`
pub async fn status(State(state): State<AppState>) -> Response {
    let loaded = state.loaded_opt().await;
    let counters = state.counters().await;
    let cfg = state.config();
    let model = loaded.as_ref().map(|l| {
        let g = &l.engine.geometry;
        ModelStatus {
            name: l.name.clone(),
            path: l.path.display().to_string(),
            family: l.family.clone(),
            architecture: l.architecture.clone(),
            layers: g.n_layers,
            n_embd: g.n_embd,
            experts: g.n_expert,
            experts_used: g.n_expert_used,
            context_length: l.n_ctx,
            vocabulary: g.n_vocab,
            quant: quant_of(l),
            mapped_bytes: l.mapped_bytes,
            loaded_ms: l.loaded_at.elapsed().as_millis() as u64,
        }
    });
    Json(Status {
        loaded: loaded.is_some(),
        model,
        backend: state.backend().name().to_string(),
        bind: cfg.bind.clone(),
        port: cfg.port,
        uptime_ms: state.uptime_ms(),
        requests: counters.requests,
        errors: counters.errors,
        prompt_tokens: counters.prompt_tokens,
        gen_tokens: counters.gen_tokens,
        version: env!("CARGO_PKG_VERSION"),
    })
    .into_response()
}

/// The dominant quantization of the loaded file, from its tensor types.
fn quant_of(l: &crate::state::Loaded) -> String {
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for shard in l.engine.model.shards() {
        for t in shard.tensors() {
            if let Ok(q) = strata_gguf::quant::QuantType::from_id(t.ty) {
                *counts.entry(q.name()).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map_or_else(|| "unknown".to_string(), |(name, _)| name.to_string())
}

/// `/v1/load`
pub async fn load(State(state): State<AppState>, Json(req): Json<LoadRequest>) -> Response {
    let path = if let Some(p) = req.model.or(req.path) {
        Some(PathBuf::from(p))
    } else {
        state.config().model.clone()
    };
    let Some(path) = path else {
        return Error::BadRequest {
            message: "load needs a model path".to_string(),
        }
        .into_response();
    };
    let n_ctx = req.n_ctx.unwrap_or(state.config().n_ctx);
    match state.load(path, n_ctx).await {
        Ok(l) => Json(serde_json::json!({
            "status": "loaded",
            "model": l.name,
            "path": l.path.display().to_string(),
            "family": l.family,
            "context_length": l.n_ctx,
            "backend": l.backend_name,
        }))
        .into_response(),
        Err(e) => e.into_response(),
    }
}

/// `/v1/unload`
pub async fn unload(State(state): State<AppState>) -> Response {
    match state.unload().await {
        Ok(()) => Json(serde_json::json!({"status": "unloaded"})).into_response(),
        Err(e) => e.into_response(),
    }
}
