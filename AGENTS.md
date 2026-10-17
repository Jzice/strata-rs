# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this is

`strata-rs` is a Rust replica of the local LLM inference engine
[Strata](https://github.com/Niko1221/Strata). It runs Qwen3.8-Flash-Next
(`qwen4exp`), Qwen3.6-35B-A3B (`qwen35moe`), and Qwen3.5-9B (`qwen35`, a dense
GDN + gated-attention hybrid) on a gaming PC and provides an
OpenAI/Anthropic-compatible HTTP server, a web chat app and a one-click
installer. The upstream project is the reference implementation: numeric and
API parity against it is a hard requirement, and `docs/` holds the porting
notes and the parity report.

## Repository layout

```
crates/strata-gguf          GGUF reader, metadata, model geometry, ggml dequant
crates/strata-tokenizer     BPE tokenizer from GGUF metadata + chat templates
crates/strata-core          Engine: families, layers, sampling, sessions, ComputeBackend
crates/strata-backend-cpu   CPU backend (rayon)
crates/strata-backend-wgpu  wgpu backend (WGSL), CPU fallback
crates/strata-server        axum HTTP API (OpenAI/Anthropic/Responses, SSE, monitor)
crates/strata-cli           chat / benchmark / serve binary
crates/strata-installer     installer binary (detect, download, config, service)
tools/                      Python codegen (extract_ggml_tables.py)
web/                        React + Tailwind chat SPA (built by Vite, served by strata-server)
installer/                  Windows entry scripts + Dockerfile
```

The workspace is feature-complete across all planned phases; see the phase
table in `README.md` for the status. Model files (`*.gguf`) are not in this
repository — the installer downloads them.

## Commands

```sh
cargo build                  # CPU-only; no GPU needed
cargo test
cargo clippy --all-targets   # must be warning-free
cargo fmt --check
```

The wgpu backend is opt-in via the `strata-backend-wgpu/gpu` feature. Everything
must build, test and pass clippy without it, since default development targets
the CPU backend.

## Code conventions

- Rust edition 2021, MSRV 1.85. All dependency versions and lints are pinned
  once in the workspace `Cargo.toml`; crates inherit them with
  `version.workspace = true` / `[lints] workspace = true`. Do not add new
  external dependencies without adding them to `[workspace.dependencies]` first.
- Formatting: `rustfmt.toml` (max_width 110, Unix newlines). Run `cargo fmt`
  before committing.
- Clippy `all` + `pedantic` are warn-level; do not introduce new warnings.
  `#[allow(...)]` needs a comment explaining why (see the relaxed lints in
  `Cargo.toml`).
- Error handling: each crate defines its own `Error` enum (thiserror) plus
  `pub type Result<T> = std::result::Result<T, Error>` in `src/error.rs`, and
  re-exports `Error`/`Result` from `src/lib.rs`. Binaries use `anyhow`.
- Logging: `tracing`, never `println!` except in the CLI binary itself.
- Every public crate, module and item carries doc comments; crates start with a
  `//!` module header. Keep comments factual — explain *why*, not *what*.
- Public types must derive `Debug` (workspace lint `missing_debug_implementations`).
- Generated code must not be hand-edited: `crates/strata-gguf/src/tables.rs` is
  produced by `tools/extract_ggml_tables.py` from ggml headers. Regenerate it
  with the script instead.

## Architecture rules

- `strata-core` defines the `ComputeBackend` trait; `strata-backend-cpu` and
  `strata-backend-wgpu` are its only implementations. Engine code must never
  reference a concrete backend or backend-only types (wgpu, rayon) — only the
  trait. Backends are cheap-to-clone, `Arc`-shared, own no weights, and must be
  usable from multiple request threads (`Send + Sync`).
- GGUF files are memory-mapped; never assume tensors are resident or aligned
  beyond the file's declared alignment. Dequantization must be exact against the
  ggml reference implementation (bit-exact, not approximate).
- The HTTP API surface must stay wire-compatible with the OpenAI and Anthropic
  APIs (including SSE streaming shape), because external clients target them.
