# strata-rs

A Rust replica of [Strata](https://github.com/Niko1221/Strata): a local LLM
inference engine that runs Qwen3.8-Flash-Next (`qwen4exp`), Qwen3.6-35B-A3B
(`qwen35moe`), and Qwen3.5-9B (`qwen35`, a dense GDN + gated-attention hybrid)
on a gaming PC, with an OpenAI/Anthropic-compatible HTTP server, a web chat app
and a one-click installer.

## Status

All planned phases are implemented and the workspace builds warning-free. See
[CHANGELOG.md](CHANGELOG.md) for the per-change history.

| Phase | Crate | Status |
| --- | --- | --- |
| 1 | `strata-gguf` (reader, metadata, quantization) | done |
| 2 | `strata-tokenizer` (BPE, chat template) | done |
| 3 | `strata-core` + `strata-backend-cpu` (CPU engine) | done |
| 4 | `strata-cli` (chat, benchmark) + golden-output validation | done |
| 5 | `strata-backend-wgpu` (WGSL kernels, Vulkan-first) | done (opt-in) |
| 6 | `strata-server` (OpenAI/Anthropic API, SSE, monitor, MCP) | done |
| 7 | `web/` (React chat app) | done |
| 8 | `strata-installer` + `installer/` scripts | done |

## Building

```sh
cargo build                 # CPU-only engine (no GPU needed)
cargo build --features ...  # per-crate features, e.g. strata-backend-wgpu/gpu
cargo test
cargo clippy --all-targets
cargo fmt --check
```

The wgpu backend is opt-in (`strata-backend-wgpu/gpu`); without it everything
runs on the CPU backend, so development and tests need no graphics card.

## Layout

```
crates/strata-gguf          GGUF reader, metadata, model geometry, ggml dequant
crates/strata-tokenizer     BPE tokenizer from GGUF metadata + chat templates
crates/strata-core          Engine: families, layers, sampling, sessions, ComputeBackend
crates/strata-backend-cpu   CPU backend (rayon)
crates/strata-backend-wgpu  wgpu backend (WGSL), CPU fallback
crates/strata-server        axum HTTP API (OpenAI/Anthropic/Responses, SSE, monitor)
crates/strata-cli           chat / benchmark / serve binary
crates/strata-installer     installer binary (detect, download, config, service)
web/                        React + Tailwind chat SPA (built by Vite, served by strata-server)
installer/                  Windows entry scripts + Dockerfile
```

Model files (`*.gguf`) are not part of this repository; the installer downloads
them. `docs/` holds the porting notes and the numeric/API parity report against
the upstream engine.

## License

MIT OR Apache-2.0.
