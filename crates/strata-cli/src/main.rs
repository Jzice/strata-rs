//! `strata` command line: model info, one-shot completion, interactive chat
//! and a benchmark.

use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use strata_core::backend::{ComputeBackend, CpuBackend};
use strata_core::generate::{eos_ids, generate, FinishReason, GenerateParams, GenerateStats, Stop};
use strata_core::sampler::SamplerParams;
use strata_core::{Engine, Session};
use strata_tokenizer::Tokenizer;

/// Which backend runs the matmuls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// The GPU if one is available, the CPU otherwise.
    #[default]
    Auto,
    /// The CPU reference path.
    Cpu,
    /// The wgpu device path (Vulkan, then D3D12, then Metal).
    Gpu,
}

/// Picks the backend, with `auto` falling back to the CPU when no adapter is
/// found (or when this build has no GPU support compiled in).
fn pick_backend(choice: BackendChoice, budget_mib: u64) -> Arc<dyn ComputeBackend> {
    match choice {
        BackendChoice::Cpu => {
            eprintln!("backend: cpu (the reference path)");
            Arc::new(CpuBackend::new())
        }
        BackendChoice::Auto | BackendChoice::Gpu => {
            if let Some(b) = gpu_backend(budget_mib) {
                b
            } else {
                if choice == BackendChoice::Gpu {
                    eprintln!("backend: no GPU adapter, falling back to cpu");
                }
                Arc::new(CpuBackend::new())
            }
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu_backend(budget_mib: u64) -> Option<Arc<dyn ComputeBackend>> {
    match strata_backend_wgpu::WgpuBackend::probe_with_budget(budget_mib << 20) {
        Ok(b) => {
            eprintln!(
                "backend: {} on {} ({} MiB of resident weights)",
                b.name(),
                b.adapter(),
                budget_mib
            );
            Some(Arc::new(b))
        }
        Err(e) => {
            eprintln!("backend: gpu unavailable ({e})");
            None
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn gpu_backend(_budget_mib: u64) -> Option<Arc<dyn ComputeBackend>> {
    eprintln!("backend: this build has no GPU support (rebuild with --features gpu)");
    None
}

#[derive(Parser)]
#[command(
    name = "strata",
    version,
    about = "Strata: run a large MoE model on your own PC"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print a model file's geometry and quantization summary.
    Info {
        /// Path to a GGUF model file or its first shard.
        #[arg(long)]
        model: PathBuf,
    },
    /// One prompt, one answer.
    Run {
        #[arg(long)]
        model: PathBuf,
        /// The prompt text.
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: i32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 0.05)]
        min_p: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Print only the answer (no banner, no stats).
        #[arg(long)]
        quiet: bool,
        /// Where the matmuls run.
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendChoice,
        /// Resident weight budget for the GPU backend.
        #[arg(long, default_value_t = 4096)]
        gpu_weights_mib: u64,
    },
    /// Interactive chat: one message per line, `exit` to leave.
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: i32,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Where the matmuls run.
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendChoice,
        /// Resident weight budget for the GPU backend.
        #[arg(long, default_value_t = 4096)]
        gpu_weights_mib: u64,
    },
    /// Run the HTTP server (OpenAI- and Anthropic-compatible).
    Serve {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Required when binding outside loopback.
        #[arg(long)]
        api_key: Option<String>,
        /// Prompt + answer budget per request.
        #[arg(long, default_value_t = 32768)]
        n_ctx: u32,
        /// Where the matmuls run.
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendChoice,
        /// Resident weight budget for the GPU backend.
        #[arg(long, default_value_t = 4096)]
        gpu_weights_mib: u64,
        /// The web app's built assets, if present.
        #[arg(long)]
        web_dir: Option<PathBuf>,
    },
    /// Measure prefill and decode speed.
    Bench {
        #[arg(long)]
        model: PathBuf,
        /// Prompt tokens, synthesized from the vocabulary.
        #[arg(long, default_value_t = 64)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 32)]
        gen_tokens: usize,
        #[arg(long, default_value_t = 2)]
        runs: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Where the matmuls run.
        #[arg(long, value_enum, default_value = "auto")]
        backend: BackendChoice,
        /// Resident weight budget for the GPU backend.
        #[arg(long, default_value_t = 4096)]
        gpu_weights_mib: u64,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let cli = Cli::parse();
    let code = match run(cli) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Serve {
            model,
            host,
            port,
            api_key,
            n_ctx,
            backend,
            gpu_weights_mib,
            web_dir,
        } => {
            // Binding outside loopback without a key is refused: the server
            // has no other authentication.
            if api_key.is_none() && host != "127.0.0.1" && host != "localhost" && host != "[::1]" {
                return Err("--api-key is required when binding outside loopback".into());
            }
            let config = strata_server::ServerConfig {
                bind: host,
                port,
                api_key,
                model,
                n_ctx,
                gpu_weights_mib,
                web_dir,
            };
            let be = pick_backend(backend, gpu_weights_mib);
            tokio::runtime::Runtime::new()?.block_on(strata_server::serve(config, be))?;
        }
        Command::Info { model } => {
            let engine = Engine::load(&model)?;
            print_info(&engine);
        }
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            quiet,
            backend,
            gpu_weights_mib,
        } => {
            let engine = Engine::load(&model)?;
            let mut tokenizer = Tokenizer::from_gguf(engine.model.meta()).ok();
            let prompt_ids = match &mut tokenizer {
                Some(t) => t.encode(&prompt, true)?,
                None => vec![1],
            };
            let params = GenerateParams {
                max_tokens,
                sampler: SamplerParams {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    seed,
                    ..SamplerParams::default()
                },
                stop: Stop {
                    token_ids: eos_ids(tokenizer.as_ref()),
                    strings: Vec::new(),
                },
            };
            let be = pick_backend(backend, gpu_weights_mib);
            let mut session = Session::with_backend(Arc::clone(&engine), be);
            let stats = generate(
                &engine,
                &mut session,
                &prompt_ids,
                &params,
                tokenizer.as_ref(),
                |chunk| {
                    if !chunk.is_stop {
                        print!("{}", chunk.text);
                        std::io::stdout().flush().ok();
                    }
                    true
                },
            );
            println!();
            if !quiet {
                print_stats(&stats);
            }
        }
        Command::Chat {
            model,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            backend,
            gpu_weights_mib,
        } => {
            let engine = Engine::load(&model)?;
            let mut tokenizer = Tokenizer::from_gguf(engine.model.meta()).ok();
            let eos = eos_ids(tokenizer.as_ref());
            let be = pick_backend(backend, gpu_weights_mib);
            let mut session = Session::with_backend(Arc::clone(&engine), Arc::clone(&be));
            println!(
                "strata chat — {} ({} layers, {} experts of {})",
                engine.family.title(),
                engine.geometry.n_layers,
                engine.geometry.n_expert_used,
                engine.geometry.n_expert
            );
            println!("type a message, or 'exit' to leave, or 'reset' to start a new chat.");
            loop {
                print!("\nyou> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    break;
                }
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if line == "exit" || line == "quit" {
                    break;
                }
                if line == "reset" {
                    session.reset();
                    println!("(new chat)");
                    continue;
                }
                let ids = match &mut tokenizer {
                    Some(t) => t.encode(line, true)?,
                    None => vec![1],
                };
                let params = GenerateParams {
                    max_tokens,
                    sampler: SamplerParams {
                        temperature,
                        top_k,
                        top_p,
                        seed,
                        ..SamplerParams::default()
                    },
                    stop: Stop {
                        token_ids: eos.clone(),
                        strings: Vec::new(),
                    },
                };
                print!("strata> ");
                std::io::stdout().flush()?;
                let stats = generate(
                    &engine,
                    &mut session,
                    &ids,
                    &params,
                    tokenizer.as_ref(),
                    |chunk| {
                        if !chunk.is_stop {
                            print!("{}", chunk.text);
                            std::io::stdout().flush().ok();
                        }
                        true
                    },
                );
                println!();
                eprintln!("[{:.1} tok/s]", stats.tokens_per_s);
            }
        }
        Command::Bench {
            model,
            prompt_tokens,
            gen_tokens,
            runs,
            seed,
            backend,
            gpu_weights_mib,
        } => {
            let engine = Engine::load(&model)?;
            let be = pick_backend(backend, gpu_weights_mib);
            let vocab = engine.geometry.n_vocab.max(2) as u32;
            let prompt: Vec<u32> = (0..prompt_tokens)
                .map(|i| 1 + ((i as u32 * 7 + 3) % (vocab - 1)))
                .collect();
            println!(
                "bench: {} prompt tokens, {} generated, {} run(s)",
                prompt.len(),
                gen_tokens,
                runs
            );
            let mut best_prompt = 0.0f64;
            let mut best_gen = 0.0f64;
            for r in 0..runs {
                let mut session = Session::with_backend(Arc::clone(&engine), Arc::clone(&be));
                let params = GenerateParams {
                    max_tokens: gen_tokens,
                    sampler: SamplerParams {
                        temperature: 0.0, // greedy: the run measures compute
                        seed: seed + r as u64,
                        ..SamplerParams::default()
                    },
                    stop: Stop::default(),
                };
                let stats = generate(
                    &engine,
                    &mut session,
                    &prompt,
                    &params,
                    None::<&Tokenizer>,
                    |_| true,
                );
                let p = if stats.prefill_ms > 0.0 {
                    stats.prompt_tokens as f64 / (stats.prefill_ms / 1000.0)
                } else {
                    0.0
                };
                println!(
                    "run {}: prompt {:.1} tok/s ({} tokens in {:.0} ms), decode {:.1} tok/s ({} tokens in {:.0} ms)",
                    r + 1,
                    p,
                    stats.prompt_tokens,
                    stats.prefill_ms,
                    stats.tokens_per_s,
                    stats.gen_tokens,
                    stats.decode_ms
                );
                best_prompt = best_prompt.max(p);
                best_gen = best_gen.max(stats.tokens_per_s);
            }
            println!("best: prompt {best_prompt:.1} tok/s, decode {best_gen:.1} tok/s");
        }
    }
    Ok(())
}

fn print_info(engine: &Engine) {
    let g = &engine.geometry;
    println!("family       {}", engine.family.title());
    println!("architecture {}", engine.family.arch());
    println!(
        "layers       {} ({} GDN, {} attention, every {})",
        g.n_layers,
        g.n_gdn_layers(),
        g.n_qsa_layers(),
        g.qsa_interval
    );
    println!("embedding    {}", g.n_embd);
    println!("vocabulary   {}", g.n_vocab);
    println!(
        "attention    {} heads, {} kv heads, head_dim {}, rope {}",
        g.n_head, g.n_head_kv, g.head_dim, g.n_rot
    );
    if g.has_indexer() {
        println!(
            "indexer      {} heads, key_dim {}, block {}, top {}",
            g.idx_q_heads, g.idx_key_dim, g.idx_block, g.idx_top_k
        );
    }
    println!(
        "GDN          state {}, {} key / {} value heads, conv {}, value_dim {}",
        g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads, g.ssm_d_conv, g.ssm_value_dim
    );
    println!(
        "MoE          {} experts, top {}, ff {} (shared {})",
        g.n_expert, g.n_expert_used, g.n_ff, g.n_ff_shexp
    );
    if g.has_hyper_connections() {
        println!("hyper-conn   {} streams, low rank {}", g.hc, g.hc_lr);
    }
    println!("MTP blocks   {}", g.n_mtp_layers);
    println!("tensors      {}", engine.model.tensor_count());
    println!("shards       {}", engine.model.n_shards());
    println!(
        "bytes        {:.2} GiB of mapped weights",
        engine.model.mapped_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

fn print_stats(s: &GenerateStats) {
    let reason = match s.finish_reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "cancelled",
    };
    println!(
        "[{} prompt tokens in {:.0} ms, {} generated in {:.0} ms, {:.1} tok/s, {reason}]",
        s.prompt_tokens, s.prefill_ms, s.gen_tokens, s.decode_ms, s.tokens_per_s
    );
}
