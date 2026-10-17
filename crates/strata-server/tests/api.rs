//! HTTP integration tests: the server really binds, loads the tiny model and
//! answers every endpoint.
//!
//! The model is the synthetic fixture from `strata-cli/tests/data`, so these
//! tests need no 35 GB download and run on any machine.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};
use strata_core::backend::CpuBackend;
use strata_server::{AppState, ServerConfig};

fn model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../strata-cli/tests/data/tiny-qwen4exp.gguf")
}

/// Boots a server on an ephemeral port with the fixture loaded.
async fn spawn() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let config = ServerConfig {
        bind: "127.0.0.1".to_string(),
        port: addr.port(),
        api_key: None,
        model: Some(model()),
        n_ctx: 512,
        gpu_weights_mib: 64,
        web_dir: None,
    };
    let handle = tokio::spawn(async move {
        let _ = strata_server::run(listener, config, Arc::new(CpuBackend::new())).await;
    });
    // Wait until the model is loaded (the route answers 200 with loaded=true).
    for _ in 0..200 {
        if let Ok(r) = reqwest::get(format!("http://{addr}/v1/status")).await {
            if r.status().is_success() {
                let v: Value = r.json().await.unwrap_or(json!({}));
                if v["loaded"] == json!(true) {
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    (addr, handle)
}

fn url(addr: &SocketAddr, path: &str) -> String {
    format!("http://{addr}{path}")
}

#[tokio::test]
async fn health_and_status_report_the_loaded_model() {
    let (addr, _h) = spawn().await;
    let health: Value = reqwest::get(url(&addr, "/health"))
        .await
        .expect("health")
        .json()
        .await
        .expect("json");
    assert_eq!(health["status"], json!("ok"));
    assert_eq!(health["loaded"], json!(true));
    assert!(health["model"].as_str().unwrap_or("").contains("tiny-qwen4exp"));

    let status: Value = reqwest::get(url(&addr, "/v1/status"))
        .await
        .expect("status")
        .json()
        .await
        .expect("json");
    assert_eq!(status["loaded"], json!(true));
    assert_eq!(status["model"]["family"], json!("Qwen3.8-Flash-Next"));
    assert_eq!(status["backend"], json!("cpu"));
    assert!(status["model"]["layers"].as_i64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn models_lists_the_loaded_model() {
    let (addr, _h) = spawn().await;
    let v: Value = reqwest::get(url(&addr, "/v1/models"))
        .await
        .expect("models")
        .json()
        .await
        .expect("json");
    assert_eq!(v["object"], json!("list"));
    assert_eq!(v["data"].as_array().expect("array").len(), 1);
    assert!(v["data"][0]["id"].as_str().unwrap().contains("tiny-qwen4exp"));
}

#[tokio::test]
async fn chat_completion_answers_in_openai_shape() {
    let (addr, _h) = spawn().await;
    let model = "tiny-qwen4exp";
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("post");
    assert!(r.status().is_success(), "{}", r.status());
    let v: Value = r.json().await.expect("json");
    assert_eq!(v["object"], json!("chat.completion"));
    assert_eq!(v["choices"][0]["message"]["role"], json!("assistant"));
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(v["choices"][0]["finish_reason"].is_string());
}

#[tokio::test]
async fn chat_completion_streams_sse() {
    let (addr, _h) = spawn().await;
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "tiny-qwen4exp",
            "messages": [{"role": "user", "content": "count to four"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": true,
        }))
        .send()
        .await
        .expect("post");
    assert!(r.status().is_success());
    let body = r.text().await.expect("body");
    assert!(body.contains("data: "), "SSE frames: {body}");
    assert!(body.contains("[DONE]"), "the stream terminates: {body}");
    assert!(body.contains("chat.completion.chunk"), "chunk objects: {body}");
}

#[tokio::test]
async fn anthropic_messages_answers_in_anthropic_shape() {
    let (addr, _h) = spawn().await;
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/messages"))
        .json(&json!({
            "model": "tiny-qwen4exp",
            "max_tokens": 4,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }))
        .send()
        .await
        .expect("post");
    assert!(r.status().is_success(), "{}", r.status());
    let v: Value = r.json().await.expect("json");
    assert_eq!(v["type"], json!("message"));
    assert_eq!(v["role"], json!("assistant"));
    assert_eq!(v["content"][0]["type"], json!("text"));
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn responses_api_answers() {
    let (addr, _h) = spawn().await;
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/responses"))
        .json(&json!({
            "model": "tiny-qwen4exp",
            "input": "say something short",
            "max_output_tokens": 4,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("post");
    assert!(r.status().is_success(), "{}", r.status());
    let v: Value = r.json().await.expect("json");
    assert_eq!(v["object"], json!("response"));
    assert_eq!(v["output"][0]["type"], json!("message"));
    assert_eq!(v["output"][0]["content"][0]["type"], json!("output_text"));
}

#[tokio::test]
async fn completions_endpoint_answers() {
    let (addr, _h) = spawn().await;
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/completions"))
        .json(&json!({
            "model": "tiny-qwen4exp",
            "prompt": "once upon a time",
            "max_tokens": 3,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("post");
    assert!(r.status().is_success(), "{}", r.status());
    let v: Value = r.json().await.expect("json");
    assert_eq!(v["object"], json!("text_completion"));
    assert!(v["choices"][0]["text"].is_string());
}

#[tokio::test]
async fn monitor_metrics_and_mcp_are_served() {
    let (addr, _h) = spawn().await;
    // Produce one request so the log has something in it.
    let _ = reqwest::Client::new()
        .post(url(&addr, "/v1/completions"))
        .json(&json!({"prompt": "hi", "max_tokens": 2}))
        .send()
        .await;

    let monitor: Value = reqwest::get(url(&addr, "/api-monitor"))
        .await
        .expect("monitor")
        .json()
        .await
        .expect("json");
    assert!(monitor["uptime_ms"].as_u64().unwrap() > 0);
    assert_eq!(monitor["backend"], json!("cpu"));

    let requests: Value = reqwest::get(url(&addr, "/api/requests?limit=10"))
        .await
        .expect("requests")
        .json()
        .await
        .expect("json");
    assert!(!requests["requests"].as_array().expect("array").is_empty());

    let metrics = reqwest::get(url(&addr, "/metrics"))
        .await
        .expect("metrics")
        .text()
        .await
        .expect("text");
    assert!(metrics.contains("strata_requests_total"), "{metrics}");
    assert!(metrics.contains("strata_model_loaded 1"), "{metrics}");

    let mcp = reqwest::Client::new()
        .post(url(&addr, "/mcp"))
        .json(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}))
        .send()
        .await
        .expect("mcp");
    let v: Value = mcp.json().await.expect("json");
    let tools = v["result"]["tools"].as_array().expect("tools").clone();
    assert!(tools.iter().any(|t| t["name"] == json!("strata_status")));
    assert!(tools.iter().any(|t| t["name"] == json!("strata_complete")));
}

#[tokio::test]
async fn unload_then_requests_are_rejected() {
    let (addr, _h) = spawn().await;
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/unload"))
        .json(&json!({}))
        .send()
        .await
        .expect("unload");
    assert!(r.status().is_success());
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("post");
    assert_eq!(r.status(), 503, "no model means 503");
    // And it can be loaded again.
    let r = reqwest::Client::new()
        .post(url(&addr, "/v1/load"))
        .json(&json!({"model": model().to_string_lossy(), "n_ctx": 256}))
        .send()
        .await
        .expect("load");
    assert!(r.status().is_success());
}

#[tokio::test]
async fn api_key_is_required_when_configured() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let config = ServerConfig {
        bind: "127.0.0.1".to_string(),
        port: addr.port(),
        api_key: Some("secret".to_string()),
        model: None,
        n_ctx: 256,
        gpu_weights_mib: 64,
        web_dir: None,
    };
    let _h = tokio::spawn(async move {
        let _ = strata_server::run(listener, config, Arc::new(CpuBackend::new())).await;
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let denied = reqwest::get(url(&addr, "/v1/status")).await.expect("status");
    assert_eq!(denied.status(), 401, "no key means 401");
    // Health stays open so a load balancer can probe it.
    let health = reqwest::get(url(&addr, "/health")).await.expect("health");
    assert!(health.status().is_success());

    let allowed = reqwest::Client::new()
        .get(url(&addr, "/v1/status"))
        .bearer_auth("secret")
        .send()
        .await
        .expect("status");
    assert!(allowed.status().is_success());
}

#[tokio::test]
async fn state_loads_and_unloads_directly() {
    let state = AppState::new(
        ServerConfig {
            model: None,
            ..ServerConfig::default()
        },
        Arc::new(CpuBackend::new()),
    );
    assert!(state.loaded_opt().await.is_none());
    let loaded = state.load(model(), 512).await.expect("load");
    assert_eq!(loaded.family, "Qwen3.8-Flash-Next");
    assert!(state.loaded_opt().await.is_some());
    state.unload().await.expect("unload");
    assert!(state.loaded_opt().await.is_none());
}
