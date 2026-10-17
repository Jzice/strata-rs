//! Shared server state: the engine handle, the loaded-model registry, the
//! request accounting the monitor and Prometheus routes read, and the
//! generation queue.
//!
//! One engine runs one session at a time, so generation is serialised behind
//! `busy`: concurrent requests queue rather than interleaving their tokens.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use strata_core::backend::ComputeBackend;
use strata_core::{Engine, Session};
use strata_tokenizer::Tokenizer;

use crate::error::{Error, Result};

/// The most recent requests the monitor keeps.
const REQUEST_LOG: usize = 200;

/// Where the server came from and how it is configured.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// The host it binds; shown by `/v1/status`.
    pub bind: String,
    pub port: u16,
    /// `None` means no key is required (the default, loopback-only bind).
    pub api_key: Option<String>,
    /// A model path loaded at startup, if any.
    pub model: Option<PathBuf>,
    pub n_ctx: u32,
    /// Resident weight budget for the GPU backend, in MiB.
    pub gpu_weights_mib: u64,
    /// Where the web app's built assets live, if they are present.
    pub web_dir: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            model: None,
            n_ctx: 32768,
            gpu_weights_mib: 4096,
            web_dir: None,
        }
    }
}

/// A loaded model and the facts the status routes report.
#[derive(Debug)]
pub struct Loaded {
    pub name: String,
    pub path: PathBuf,
    pub engine: Arc<Engine>,
    pub tokenizer: Option<Tokenizer>,
    pub family: String,
    pub architecture: String,
    pub n_ctx: u32,
    pub backend_name: String,
    pub loaded_at: Instant,
    /// Bytes mapped by the GGUF file.
    pub mapped_bytes: u64,
}

/// One request the monitor saw.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestRecord {
    pub id: u64,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub ms: f64,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub at_ms: u64,
    pub streamed: bool,
    pub model: Option<String>,
}

/// The counters Prometheus reports.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counters {
    pub requests: u64,
    pub completions: u64,
    pub chat_completions: u64,
    pub messages: u64,
    pub responses: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub gen_ms: f64,
    pub streamed: u64,
    pub load_count: u64,
    pub unload_count: u64,
}

/// State shared by every route handler.
#[derive(Debug, Clone)]
pub struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    loaded: tokio::sync::RwLock<Option<Arc<Loaded>>>,
    /// Serialises generation: the engine runs one sequence at a time.
    busy: tokio::sync::Mutex<()>,
    backend: tokio::sync::RwLock<Arc<dyn ComputeBackend>>,
    requests: tokio::sync::RwLock<VecDeque<RequestRecord>>,
    counters: tokio::sync::RwLock<Counters>,
    config: ServerConfig,
    started: Instant,
    next_id: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("config", &self.config)
            .field("started", &self.started)
            .finish_non_exhaustive()
    }
}

impl AppState {
    pub fn new(config: ServerConfig, backend: Arc<dyn ComputeBackend>) -> Self {
        Self {
            inner: Arc::new(Inner {
                loaded: tokio::sync::RwLock::new(None),
                busy: tokio::sync::Mutex::new(()),
                backend: tokio::sync::RwLock::new(backend),
                requests: tokio::sync::RwLock::new(VecDeque::with_capacity(REQUEST_LOG)),
                counters: tokio::sync::RwLock::new(Counters::default()),
                config,
                started: Instant::now(),
                next_id: std::sync::atomic::AtomicU64::new(1),
            }),
        }
    }

    pub fn config(&self) -> &ServerConfig {
        &self.inner.config
    }

    pub fn api_key(&self) -> Option<&str> {
        self.inner.config.api_key.as_deref()
    }

    pub fn uptime_ms(&self) -> u64 {
        self.inner.started.elapsed().as_millis() as u64
    }

    pub fn backend(&self) -> Arc<dyn ComputeBackend> {
        // A read lock that cannot block: the backend is only replaced when the
        // operator explicitly swaps it, and a busy lock falls back to the CPU
        // reference for this call.
        if let Ok(g) = self.inner.backend.try_read() {
            Arc::clone(&g)
        } else {
            Arc::new(strata_core::backend::CpuBackend::new())
        }
    }

    /// The loaded model, or `NotLoaded`.
    pub async fn loaded(&self) -> Result<Arc<Loaded>> {
        self.inner.loaded.read().await.clone().ok_or(Error::NotLoaded)
    }

    pub async fn loaded_opt(&self) -> Option<Arc<Loaded>> {
        self.inner.loaded.read().await.clone()
    }

    /// Held for the duration of one generation.
    pub async fn generation_lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.inner.busy.lock().await
    }

    /// Loads a model file, replacing any loaded one.
    pub async fn load(&self, path: impl AsRef<Path>, n_ctx: u32) -> Result<Arc<Loaded>> {
        let path = path.as_ref().to_path_buf();
        let be = self.backend();
        let handle = tokio::task::spawn_blocking(move || -> Result<Arc<Loaded>> {
            let engine = Engine::load(&path)?;
            let tokenizer = Tokenizer::from_gguf(engine.model.meta()).ok();
            let name = match path.file_stem() {
                Some(s) => s.to_string_lossy().to_string(),
                None => "model".to_string(),
            };
            Ok(Arc::new(Loaded {
                name,
                path: path.clone(),
                family: engine.family.title().to_string(),
                architecture: engine.family.arch().to_string(),
                n_ctx,
                backend_name: be.name().to_string(),
                mapped_bytes: engine.model.mapped_bytes(),
                tokenizer,
                engine,
                loaded_at: Instant::now(),
            }))
        });
        // The join error first (the task itself), then whatever loading said.
        let loaded = handle.await.map_err(|e| Error::BadRequest {
            message: format!("loading task failed: {e}"),
        })??;

        *self.inner.loaded.write().await = Some(Arc::clone(&loaded));
        let mut c = self.inner.counters.write().await;
        c.load_count += 1;
        tracing::info!(
            "loaded {} ({} layers, {:.2} GiB mapped)",
            loaded.name,
            loaded.engine.geometry.n_layers,
            loaded.mapped_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Ok(loaded)
    }

    /// Drops the engine and every session with it.
    pub async fn unload(&self) -> Result<()> {
        let had = self.inner.loaded.write().await.take();
        if had.is_none() {
            return Err(Error::NotLoaded);
        }
        self.inner.counters.write().await.unload_count += 1;
        tracing::info!("unloaded the model");
        Ok(())
    }

    /// A fresh session for `loaded`, on this server's backend.
    pub fn session(&self, loaded: &Arc<Loaded>) -> Session {
        Session::with_backend(Arc::clone(&loaded.engine), self.backend())
    }

    pub async fn record(&self, mut rec: RequestRecord) -> u64 {
        rec.id = self
            .inner
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = rec.id;
        let mut c = self.inner.counters.write().await;
        c.requests += 1;
        c.prompt_tokens += u64::from(rec.prompt_tokens);
        c.gen_tokens += u64::from(rec.gen_tokens);
        if rec.streamed {
            c.streamed += 1;
        }
        if rec.status >= 400 {
            c.errors += 1;
        }
        drop(c);
        let mut log = self.inner.requests.write().await;
        if log.len() == REQUEST_LOG {
            log.pop_front();
        }
        log.push_back(rec);
        id
    }

    pub async fn requests(&self, limit: usize) -> Vec<RequestRecord> {
        self.inner
            .requests
            .read()
            .await
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub async fn counters(&self) -> Counters {
        *self.inner.counters.read().await
    }

    /// Bumps a named counter (used by the route handlers).
    pub async fn count(&self, f: impl FnOnce(&mut Counters)) {
        f(&mut *self.inner.counters.write().await);
    }
}
