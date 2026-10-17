//! # strata-server
//!
//! The HTTP front end: an OpenAI- and Anthropic-compatible API
//! (`/v1/chat/completions`, `/v1/messages`, `/v1/completions`, `/v1/responses`)
//! with SSE streaming, model lifecycle routes, request monitoring, Prometheus
//! metrics and the MCP tool surface. Binds `127.0.0.1` by default; exposing it
//! beyond localhost requires an API key.
//!
//! ```text
//! GET   /health              liveness and what is loaded
//! GET   /v1/status           model, backend, counters
//! GET   /v1/models           OpenAI model list
//! POST  /v1/load             load a GGUF file
//! POST  /v1/unload           drop it
//! POST  /v1/chat/completions OpenAI chat (SSE with stream: true)
//! POST  /v1/completions      OpenAI text completion
//! POST  /v1/messages         Anthropic messages (SSE with stream: true)
//! POST  /v1/responses        Responses API (SSE with stream: true)
//! GET   /api/requests        the recent request log
//! GET   /api-monitor         counters for the dashboard
//! GET   /metrics             Prometheus exposition
//! POST  /mcp                 Model Context Protocol (JSON-RPC 2.0)
//! ```

pub mod admin;
pub mod anthropic;
pub mod error;
pub mod generate;
pub mod mcp;
pub mod monitor;
pub mod openai;
pub mod responses;
pub mod state;

pub use error::{Error, Result};
pub use state::{AppState, ServerConfig};

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

impl Error {
    /// The HTTP status this error answers with.
    pub fn status(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode;
        match self {
            Error::NotLoaded => StatusCode::SERVICE_UNAVAILABLE,
            Error::AlreadyLoaded => StatusCode::CONFLICT,
            Error::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Error::BadBody { .. } | Error::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Error::Unauthorized => StatusCode::UNAUTHORIZED,
            Error::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The body, in OpenAI's error shape so clients can read it.
    pub fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": self.kind(),
                "code": self.status().as_u16(),
            }
        })
    }

    fn kind(&self) -> &'static str {
        match self {
            Error::NotLoaded => "model_not_loaded",
            Error::AlreadyLoaded => "already_loaded",
            Error::ModelNotFound { .. } => "model_not_found",
            Error::BadBody { .. } => "invalid_request_body",
            Error::BadRequest { .. } => "invalid_request_error",
            Error::Unauthorized => "unauthorized",
            Error::Engine(_) => "engine_error",
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = self.status();
        tracing::warn!("{status}: {self}");
        (status, axum::Json(self.body())).into_response()
    }
}

/// The route table. When a web directory is configured, its built assets are
/// served from `/` behind the API routes.
pub fn router(state: AppState) -> Router {
    let web = state.config().web_dir.clone();
    let api = Router::new()
        .route("/health", get(admin::health))
        .route("/api/health", get(admin::health))
        .route("/v1/status", get(admin::status))
        .route("/v1/models", get(openai::models))
        .route("/v1/load", post(admin::load))
        .route("/v1/unload", post(admin::unload))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/responses", post(responses::responses))
        .route("/api/requests", get(monitor::requests))
        .route("/api-monitor", get(monitor::monitor))
        .route("/metrics", get(monitor::metrics))
        .route("/mcp", post(mcp::mcp))
        .with_state(state.clone());

    let app = Router::new()
        .merge(api)
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth))
        .layer(axum::middleware::from_fn_with_state(state, monitor::record_mw))
        .layer(TraceLayer::new_for_http());

    match web {
        Some(dir) if dir.join("index.html").exists() => app.fallback_service(
            tower_http::services::ServeDir::new(&dir)
                .fallback(tower_http::services::ServeFile::new(dir.join("index.html"))),
        ),
        _ => app,
    }
}

/// Bearer-token auth. Skipped on loopback when no key is configured, and for
/// the health and metrics endpoints.
async fn auth(State(state): axum::extract::State<AppState>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let open = path == "/health" || path == "/api/health" || path == "/metrics";
    let Some(key) = state.api_key() else {
        return next.run(req).await;
    };
    if open {
        return next.run(req).await;
    }
    let ok = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|v| v == key);
    if ok {
        return next.run(req).await;
    }
    Error::Unauthorized.into_response()
}

/// Builds the state and binds the listener.
pub async fn serve(
    config: ServerConfig,
    backend: Arc<dyn strata_core::backend::ComputeBackend>,
) -> Result<()> {
    let addr = format!("{}:{}", config.bind, config.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| Error::BadRequest {
            message: format!("binding {addr} failed: {e}"),
        })?;
    tracing::info!("strata listening on http://{addr}");
    run(listener, config, backend).await
}

/// Serves on an already-bound listener (a test binds port 0 and asks for the
/// address).
pub async fn run(
    listener: tokio::net::TcpListener,
    config: ServerConfig,
    backend: Arc<dyn strata_core::backend::ComputeBackend>,
) -> Result<()> {
    let state = AppState::new(config.clone(), backend);
    if let Some(path) = &config.model {
        state.load(path, config.n_ctx).await?;
    }
    let app = router(state);
    axum::serve(listener, app).await.map_err(|e| Error::BadRequest {
        message: format!("serving failed: {e}"),
    })
}
