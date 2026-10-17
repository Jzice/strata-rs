# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`qwen35` model family (Qwen3.5-9B dense GDN + gated-attention hybrid).**
  The architecture is the same GDN + QSA layer split used by `qwen4exp`, but
  with a plain pre-norm residual (no hyper-connections), a dense SwiGLU FFN
  (`ffn_gate` / `ffn_up` / `ffn_down` instead of the MoE), and no MTP block.
  It reuses the existing `GdnLayer` / `QsaLayer` implementations; the FFN half
  of `forward_qwen35_dense` replaces the MoE path with a dense MLP
  (`SiLU(gate) * up`, then `down`).
- `ModelFamily::Qwen35` with arch string `qwen35`, added `read_geometry_qwen35`
  and `check_architecture_qwen35` (geometry is read from the `qwen35.` metadata
  prefix; no expert / MTP / hyper-connection fields are pinned).

### Changed

- Session scratch (`deq`) is now sized to cover the dense FFN down-projection
  width (`n_ff`) in addition to `n_embd` / `ssm_value_dim`, since the dense
  FFN's `ffn_down` reads a row of width `n_ff` during dequantization.

## [Initial implementation]

### Added

- `strata-gguf`: GGUF reader, metadata access, model geometry inference, and
  exact ggml dequantization (produced `tables.rs` via `tools/extract_ggml_tables.py`).
- `strata-tokenizer`: BPE tokenizer built from GGUF metadata with chat templates.
- `strata-core` + `strata-backend-cpu`: the CPU inference engine with the
  `qwen4exp` (GDN + QSA + MoE + hyper-connections) and `qwen35moe` (dense
  attention + MTP) families, sampling, sessions, and the `ComputeBackend` trait.
- `strata-backend-wgpu`: opt-in WGSL backend (Vulkan-first) behind the
  `strata-backend-wgpu/gpu` feature, with a CPU fallback.
- `strata-server`: axum HTTP API — OpenAI / Anthropic / Responses compatibility,
  SSE streaming, a monitor endpoint, and MCP support.
- `strata-cli`: `chat` / `benchmark` / `serve` subcommands, including the
  `--web-dir` flag that serves the built SPA.
- `web/`: a React + Tailwind chat SPA (chat, models/status, monitor, settings)
  built with Vite and served by `strata-server`.
- `installer/`: a Windows PowerShell one-click installer (hardware detection,
  resumable download, SHA-256 verification, launcher, self-update) plus a
  Dockerfile.

### Validation

- `cargo build --workspace`, `cargo test`, and `cargo clippy --all-targets`
  (including `--all-features`) are warning-free.
- The served SPA, client-route fallback, static assets, `/health`, `/v1/status`,
  and the `/v1/chat/completions` SSE streaming path were verified end-to-end.
