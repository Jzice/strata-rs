//! CPU / GPU parity: the same prompt, the same greedy answer.
//!
//! The GPU backend runs the matmul path on the device and leaves the
//! elementwise kernels on the host, so the two paths must still agree token
//! for token. Skipped when no adapter is available, and compiled only with
//! the `gpu` feature.

#![cfg(feature = "gpu")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use strata_core::backend::{ComputeBackend, CpuBackend};
use strata_core::generate::{generate, GenerateParams, Stop};
use strata_core::sampler::SamplerParams;
use strata_core::{Engine, Session};

fn model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny-qwen4exp.gguf")
}

/// Greedy: no RNG, so the two backends are directly comparable.
fn answer(be: Arc<dyn ComputeBackend>) -> Vec<u32> {
    let engine = Engine::load(model()).expect("load");
    let prompt = [1u32, 2, 3, 4, 5];
    let params = GenerateParams {
        max_tokens: 6,
        sampler: SamplerParams {
            temperature: 0.0,
            ..SamplerParams::default()
        },
        stop: Stop::default(),
    };
    let mut session = Session::with_backend(Arc::clone(&engine), be);
    let mut tokens = Vec::new();
    generate(
        &engine,
        &mut session,
        &prompt,
        &params,
        None::<&strata_tokenizer::Tokenizer>,
        |chunk| {
            tokens.push(chunk.token);
            true
        },
    );
    tokens
}

#[test]
fn gpu_and_cpu_give_the_same_greedy_answer() {
    let cpu = answer(Arc::new(CpuBackend::new()));
    let gpu = match strata_backend_wgpu::WgpuBackend::probe_with_budget(256 << 20) {
        Ok(b) => answer(Arc::new(b)),
        Err(e) => {
            eprintln!("skipping the parity test: {e}");
            return;
        }
    };
    assert_eq!(cpu, gpu, "the GPU backend must match the CPU reference");
    assert!(!cpu.is_empty());
}
