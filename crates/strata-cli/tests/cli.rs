//! CLI integration tests, and the golden-output regression check.
//!
//! The fixtures are tiny synthetic models checked in under `tests/data`
//! (regenerate them with `STRATA_WRITE_FIXTURES=1 cargo test -p strata-core
//! write_cli_fixtures`). `golden-qwen4exp.json` records this port's own
//! numerics for one prompt: every layer kind runs, so a change that moves a
//! number is caught here even before a real model file is available.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use strata_core::generate::{generate, GenerateParams, Stop};
use strata_core::sampler::SamplerParams;
use strata_core::{Engine, Session};
use strata_tokenizer::Tokenizer;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn strata(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_strata"))
        .args(args)
        .output()
        .expect("run strata")
}

fn text(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn info_prints_the_geometry_of_both_families() {
    let out = strata(&["info", "--model", &data("tiny-qwen4exp.gguf").to_string_lossy()]);
    assert!(out.status.success(), "{}", text(&out));
    let s = text(&out);
    assert!(s.contains("Qwen3.8-Flash-Next"), "{s}");
    assert!(s.contains("layers       4"), "{s}");
    assert!(s.contains("hyper-conn   2 streams"), "{s}");
    assert!(s.contains("indexer"), "{s}");

    let out = strata(&["info", "--model", &data("tiny-qwen35moe.gguf").to_string_lossy()]);
    assert!(out.status.success(), "{}", text(&out));
    let s = text(&out);
    assert!(s.contains("Qwen3.6"), "{s}");
    assert!(s.contains("MTP blocks   1"), "{s}");
}

#[test]
fn run_completes_and_reports_stats() {
    let out = strata(&[
        "run",
        "--model",
        &data("tiny-qwen4exp.gguf").to_string_lossy(),
        "--prompt",
        "the capital of France is",
        "--max-tokens",
        "8",
        "--temperature",
        "0",
    ]);
    assert!(out.status.success(), "{}", text(&out));
    let s = text(&out);
    assert!(s.contains("8 generated"), "{s}");
    assert!(s.contains("tok/s"), "{s}");
    // --quiet prints only the answer.
    let quiet = strata(&[
        "run",
        "--model",
        &data("tiny-qwen4exp.gguf").to_string_lossy(),
        "--prompt",
        "hi",
        "--max-tokens",
        "4",
        "--quiet",
    ]);
    assert!(!text(&quiet).contains("tok/s"), "{}", text(&quiet));
}

#[test]
fn bench_reports_prompt_and_decode_speed() {
    let out = strata(&[
        "bench",
        "--model",
        &data("tiny-qwen35moe.gguf").to_string_lossy(),
        "--prompt-tokens",
        "8",
        "--gen-tokens",
        "4",
        "--runs",
        "1",
    ]);
    assert!(out.status.success(), "{}", text(&out));
    let s = text(&out);
    assert!(s.contains("prompt"), "{s}");
    assert!(s.contains("decode"), "{s}");
    assert!(s.contains("best:"), "{s}");
}

/// The recorded numerics of one prompt through the tiny qwen4exp model.
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct Golden {
    /// The prompt's token ids.
    prompt: Vec<u32>,
    /// The first eight logits of the last prompt token, rounded to 1e-6.
    logits: Vec<f32>,
    /// The first eight greedy tokens.
    tokens: Vec<u32>,
}

fn measure() -> Golden {
    let engine = Engine::load(data("tiny-qwen4exp.gguf")).expect("load");
    let mut tokenizer = Tokenizer::from_gguf(engine.model.meta()).expect("tokenizer");
    let prompt = tokenizer
        .encode("the capital of France is", true)
        .expect("encode");
    let mut session = Session::new(Arc::clone(&engine));
    // One forward per prompt token; the last one's logits start the answer.
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = engine.forward(&mut session, t);
    }
    let head: Vec<f32> = logits.iter().take(8).map(|v| (v * 1e6).round() / 1e6).collect();

    let mut session = Session::new(Arc::clone(&engine));
    let params = GenerateParams {
        max_tokens: 8,
        sampler: SamplerParams {
            temperature: 0.0, // greedy: no RNG in the golden
            ..SamplerParams::default()
        },
        stop: Stop::default(),
    };
    let mut tokens = Vec::new();
    generate(
        &engine,
        &mut session,
        &prompt,
        &params,
        None::<&Tokenizer>,
        |chunk| {
            tokens.push(chunk.token);
            true
        },
    );
    Golden {
        prompt,
        logits: head,
        tokens,
    }
}

#[test]
fn golden_output_matches_the_record() {
    let path = data("golden-qwen4exp.json");
    let got = measure();
    if std::env::var("STRATA_WRITE_GOLDEN").is_ok() {
        std::fs::write(&path, serde_json::to_string_pretty(&got).expect("serialize")).expect("write golden");
        return;
    }
    let want: Golden =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("golden file")).expect("parse");
    assert_eq!(want.prompt, got.prompt, "the prompt tokenization changed");
    assert_eq!(want.logits, got.logits, "the logits changed");
    assert_eq!(want.tokens, got.tokens, "the greedy answer changed");
}
