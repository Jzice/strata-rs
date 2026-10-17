//! A tiny synthetic-model builder plus end-to-end engine tests for both
//! families.
//!
//! The engine is geometry-driven, so a small model exercises every layer
//! kind (GDN, sparse attention with the indexer, dense attention), the
//! hyper-connection stack, the MoE router and the head — without the 35 GB
//! of weights the real files carry.

#![allow(clippy::needless_range_loop)]
use half::bf16;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use strata_core::engine::Engine;
use strata_core::sampler::{sample, SamplerParams};
use strata_core::session::Session;
use strata_gguf::quant::QuantType;

/// A minimal GGUF writer: metadata (all value types), 2-D and 3-D tensors.
struct GgufWriter {
    buf: Vec<u8>,
    n_tensors: u64,
    n_kv: u64,
    kv: Vec<u8>,
    infos: Vec<u8>,
    data: Vec<u8>,
    offset: u64,
}

#[derive(Clone)]
enum Val {
    U32(u32),
    F32(f32),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
}

impl GgufWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            n_tensors: 0,
            n_kv: 0,
            kv: Vec::new(),
            infos: Vec::new(),
            data: Vec::new(),
            offset: 0,
        }
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn kv(&mut self, key: &str, v: &Val) {
        self.n_kv += 1;
        let mut b = Vec::new();
        Self::str_into(&mut b, key);
        match v {
            Val::U32(x) => {
                b.extend_from_slice(&4u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            Val::F32(x) => {
                b.extend_from_slice(&6u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            Val::Str(s) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                Self::str_into(&mut b, s);
            }
            Val::ArrStr(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for s in items {
                    Self::str_into(&mut b, s);
                }
            }
            Val::ArrI32(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&5u32.to_le_bytes()); // element type: i32
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for x in items {
                    b.extend_from_slice(&(*x as u32).to_le_bytes());
                }
            }
        }
        self.kv.extend_from_slice(&b);
    }

    fn str_into(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn tensor(&mut self, name: &str, shape: &[u64], ty: u32, bytes: &[u8]) {
        self.n_tensors += 1;
        Self::str_into(&mut self.infos, name);
        self.infos.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            self.infos.extend_from_slice(&d.to_le_bytes());
        }
        self.infos.extend_from_slice(&ty.to_le_bytes());
        self.infos.extend_from_slice(&self.offset.to_le_bytes());
        self.offset += bytes.len() as u64;
        self.data.extend_from_slice(bytes);
    }

    /// A tensor of pseudo-random f32 values (deterministic LCG).
    fn tensor_f32(&mut self, name: &str, shape: &[u64], seed: &mut u32) {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // small values in [-0.5, 0.5)
            let v = ((*seed >> 8) as f32 / u16::MAX as f32) - 0.5;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.tensor(name, shape, QuantType::F32.id(), &bytes);
    }

    /// A matrix of `rows x cols` f32 values. GGUF stores dims `[cols, rows]`
    /// (dim 0 is the fast axis), so this is the mathematical orientation.
    fn mat(&mut self, name: &str, rows: usize, cols: usize, seed: &mut u32) {
        self.tensor_f32(name, &[cols as u64, rows as u64], seed);
    }

    /// A matrix of `rows x cols` bf16 values.
    fn mat_bf16(&mut self, name: &str, rows: usize, cols: usize, seed: &mut u32) {
        self.tensor_bf16(name, &[cols as u64, rows as u64], seed);
    }

    /// A tensor of pseudo-random bf16 values.
    fn tensor_bf16(&mut self, name: &str, shape: &[u64], seed: &mut u32) {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let mut bytes = Vec::with_capacity(n * 2);
        for _ in 0..n {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((*seed >> 8) as f32 / u16::MAX as f32) - 0.5;
            bytes.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
        }
        self.tensor(name, shape, QuantType::Bf16.id(), &bytes);
    }

    /// A 1-D f32 tensor.
    fn vec_f32(&mut self, name: &str, n: usize, fill: impl Fn(usize) -> f32) {
        let mut bytes = Vec::with_capacity(n * 4);
        for i in 0..n {
            bytes.extend_from_slice(&fill(i).to_le_bytes());
        }
        self.tensor(name, &[n as u64], QuantType::F32.id(), &bytes);
    }

    fn finish(mut self) -> Vec<u8> {
        self.buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        self.u32(3);
        self.u64(self.n_tensors);
        self.u64(self.n_kv);
        self.buf.extend_from_slice(&self.kv);
        self.buf.extend_from_slice(&self.infos);
        while self.buf.len() % 32 != 0 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(&self.data);
        self.buf
    }
}

/// Tiny qwen4exp geometry: n_embd=32, 4 layers (layer 3 = QSA), hc=2,
/// 4 experts top-2, indexer on.
fn build_qwen4exp() -> Vec<u8> {
    let n = 32usize;
    let hc = 2usize;
    let hc_lr = 4usize;
    let n_expert = 4usize;
    let n_ff = 8usize;
    let s = 4usize; // ssm_state_size
    let hk = 2usize; // ssm_k_heads
    let hv = 4usize; // ssm_v_heads
    let value_dim = s * hv; // 16
    let channels = 2 * s * hk + value_dim; // 32
    let d_conv = 2usize;
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let head_dim = 8usize;
    let idx_heads = 2usize;
    let idx_dim = 8usize;
    let n_layers = 4usize;
    let mut seed = 12345u32;

    let mut w = GgufWriter::new();
    w.kv("general.architecture", &Val::Str("qwen4exp".into()));
    w.kv("qwen4exp.block_count", &Val::U32(n_layers as u32));
    w.kv("qwen4exp.embedding_length", &Val::U32(n as u32));
    w.kv("qwen4exp.expert_count", &Val::U32(n_expert as u32));
    w.kv("qwen4exp.expert_used_count", &Val::U32(2));
    w.kv("qwen4exp.attention.head_count", &Val::U32(n_head as u32));
    w.kv("qwen4exp.attention.head_count_kv", &Val::U32(n_head_kv as u32));
    w.kv("qwen4exp.full_attention_interval", &Val::U32(4));
    w.kv("qwen4exp.ssm.state_size", &Val::U32(s as u32));
    w.kv("qwen4exp.ssm.group_count", &Val::U32(hk as u32));
    w.kv("qwen4exp.ssm.time_step_rank", &Val::U32(hv as u32));
    w.kv("qwen4exp.ssm.conv_kernel", &Val::U32(d_conv as u32));
    w.kv("qwen4exp.ssm.inner_size", &Val::U32(value_dim as u32));
    w.kv("qwen4exp.attention.key_length", &Val::U32(head_dim as u32));
    w.kv("qwen4exp.attention.value_length", &Val::U32(head_dim as u32));
    w.kv("qwen4exp.rope.dimension_count", &Val::U32(4));
    w.kv(
        "qwen4exp.attention.indexer.head_count",
        &Val::U32(idx_heads as u32),
    );
    w.kv("qwen4exp.attention.indexer.key_length", &Val::U32(idx_dim as u32));
    w.kv("qwen4exp.attention.indexer.top_k", &Val::U32(6));
    w.kv("qwen4exp.hyper_connection.count", &Val::U32(hc as u32));
    w.kv("qwen4exp.hyper_connection.low_rank", &Val::U32(hc_lr as u32));
    w.kv("qwen4exp.expert_feed_forward_length", &Val::U32(n_ff as u32));
    w.kv(
        "qwen4exp.expert_shared_feed_forward_length",
        &Val::U32(n_ff as u32),
    );
    w.kv("qwen4exp.attention.layer_norm_rms_epsilon", &Val::F32(1e-6));
    let (tokens, types) = byte_vocab();
    let n_vocab = tokens.len() as u64;
    w.kv("tokenizer.ggml.tokens", &Val::ArrStr(tokens));
    w.kv("tokenizer.ggml.merges", &Val::ArrStr(vec![]));
    w.kv("tokenizer.ggml.model", &Val::Str("gpt2".into()));
    w.kv("tokenizer.ggml.token_type", &Val::ArrI32(types));

    for l in 0..n_layers {
        let p = format!("blk.{l}.");
        if l % 4 == 3 {
            // QSA layer.
            w.mat(&format!("{p}attn_q.weight"), n_head * 2 * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_k.weight"), n_head_kv * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_v.weight"), n_head_kv * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_output.weight"), n, n_head * head_dim, &mut seed);
            w.mat(
                &format!("{p}indexer.q_proj.weight"),
                idx_heads * idx_dim,
                n,
                &mut seed,
            );
            w.mat(&format!("{p}indexer.k_proj.weight"), idx_dim, n, &mut seed);
            w.vec_f32(&format!("{p}attn_q_norm.weight"), head_dim, |i| {
                1.0 + i as f32 * 0.01
            });
            w.vec_f32(&format!("{p}attn_k_norm.weight"), head_dim, |i| {
                1.0 + i as f32 * 0.01
            });
            w.vec_f32(&format!("{p}indexer.q_norm.weight"), idx_dim, |i| {
                1.0 + i as f32 * 0.01
            });
            w.vec_f32(&format!("{p}indexer.k_norm.weight"), idx_dim, |i| {
                1.0 + i as f32 * 0.01
            });
        } else {
            // GDN layer.
            w.mat(&format!("{p}attn_qkv.weight"), channels, n, &mut seed);
            w.mat(&format!("{p}attn_gate.weight"), value_dim, n, &mut seed);
            w.mat(&format!("{p}ssm_out.weight"), n, value_dim, &mut seed);
            // The conv taps: one channel per row, d_conv columns.
            w.tensor_f32(
                &format!("{p}ssm_conv1d.weight"),
                &[d_conv as u64, channels as u64],
                &mut seed,
            );
            w.mat(&format!("{p}ssm_alpha.weight"), hv, n, &mut seed);
            w.mat(&format!("{p}ssm_beta.weight"), hv, n, &mut seed);
            w.vec_f32(&format!("{p}ssm_a"), hv, |_| -0.5);
            w.vec_f32(&format!("{p}ssm_dt.bias"), hv, |_| 0.1);
            w.vec_f32(&format!("{p}ssm_norm.weight"), s, |i| 1.0 + i as f32 * 0.01);
        }
        // MoE.
        w.mat(&format!("{p}ffn_gate_inp.weight"), n_expert, n, &mut seed);
        w.tensor_f32(
            &format!("{p}ffn_gate_exps.weight"),
            &[n_ff as u64, n as u64, n_expert as u64],
            &mut seed,
        );
        w.tensor_f32(
            &format!("{p}ffn_up_exps.weight"),
            &[n_ff as u64, n as u64, n_expert as u64],
            &mut seed,
        );
        w.tensor_f32(
            &format!("{p}ffn_down_exps.weight"),
            &[n as u64, n_ff as u64, n_expert as u64],
            &mut seed,
        );
        w.mat(&format!("{p}ffn_gate_shexp.weight"), n_ff, n, &mut seed);
        w.mat(&format!("{p}ffn_up_shexp.weight"), n_ff, n, &mut seed);
        w.mat(&format!("{p}ffn_down_shexp.weight"), n, n_ff, &mut seed);
        w.tensor_f32(
            &format!("{p}ffn_gate_inp_shexp.weight"),
            &[n as u64, 1],
            &mut seed,
        );
        // Hyper-connections.
        for prefix in ["hc_attn_", "hc_ffn_"] {
            w.vec_f32(&format!("{p}{prefix}norm.weight"), hc * n, |i| {
                1.0 + (i % 7) as f32 * 0.01
            });
            w.mat_bf16(&format!("{p}{prefix}down.weight"), hc_lr, hc * n, &mut seed);
            w.mat_bf16(&format!("{p}{prefix}up.weight"), hc * n, hc_lr, &mut seed);
            w.mat_bf16(&format!("{p}{prefix}inject.weight"), hc, hc * n, &mut seed);
        }
    }
    // Globals.
    w.tensor_f32("token_embd.weight", &[n as u64, n_vocab], &mut seed);
    w.tensor_f32("output.weight", &[n as u64, n_vocab], &mut seed);
    w.vec_f32("output_hc_norm.weight", hc * n, |i| 1.0 + (i % 5) as f32 * 0.01);
    w.mat_bf16("output_hc_down.weight", hc_lr, hc * n, &mut seed);
    w.mat_bf16("output_hc_up.weight", hc * n, hc_lr, &mut seed);
    w.finish()
}

/// Tiny qwen35moe geometry: plain residual, dense attention, SiLU GDN.
fn build_qwen35moe() -> Vec<u8> {
    let n = 32usize;
    let n_expert = 4usize;
    let n_ff = 8usize;
    let s = 4usize;
    let hk = 2usize;
    let hv = 4usize;
    let value_dim = s * hv;
    let channels = 2 * s * hk + value_dim;
    let d_conv = 2usize;
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let head_dim = 8usize;
    let n_layers = 4usize;
    let mut seed = 999u32;

    let mut w = GgufWriter::new();
    w.kv("general.architecture", &Val::Str("qwen35moe".into()));
    w.kv("qwen35moe.block_count", &Val::U32((n_layers + 1) as u32)); // +1 for the MTP block
    w.kv("qwen35moe.nextn_predict_layers", &Val::U32(1));
    w.kv("qwen35moe.embedding_length", &Val::U32(n as u32));
    w.kv("qwen35moe.expert_count", &Val::U32(n_expert as u32));
    w.kv("qwen35moe.expert_used_count", &Val::U32(2));
    w.kv("qwen35moe.attention.head_count", &Val::U32(n_head as u32));
    w.kv("qwen35moe.attention.head_count_kv", &Val::U32(n_head_kv as u32));
    w.kv("qwen35moe.attention.key_length", &Val::U32(head_dim as u32));
    w.kv("qwen35moe.attention.value_length", &Val::U32(head_dim as u32));
    w.kv("qwen35moe.full_attention_interval", &Val::U32(4));
    w.kv("qwen35moe.ssm.state_size", &Val::U32(s as u32));
    w.kv("qwen35moe.ssm.group_count", &Val::U32(hk as u32));
    w.kv("qwen35moe.ssm.time_step_rank", &Val::U32(hv as u32));
    w.kv("qwen35moe.ssm.conv_kernel", &Val::U32(d_conv as u32));
    w.kv("qwen35moe.ssm.inner_size", &Val::U32(value_dim as u32));
    w.kv("qwen35moe.expert_feed_forward_length", &Val::U32(n_ff as u32));
    w.kv(
        "qwen35moe.expert_shared_feed_forward_length",
        &Val::U32(n_ff as u32),
    );
    w.kv("qwen35moe.attention.layer_norm_rms_epsilon", &Val::F32(1e-6));
    let (tokens, types) = byte_vocab();
    let n_vocab = tokens.len() as u64;
    w.kv("tokenizer.ggml.tokens", &Val::ArrStr(tokens));
    w.kv("tokenizer.ggml.merges", &Val::ArrStr(vec![]));
    w.kv("tokenizer.ggml.model", &Val::Str("gpt2".into()));
    w.kv("tokenizer.ggml.token_type", &Val::ArrI32(types));

    for l in 0..n_layers {
        let p = format!("blk.{l}.");
        if l % 4 == 3 {
            w.mat(&format!("{p}attn_q.weight"), n_head * 2 * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_k.weight"), n_head_kv * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_v.weight"), n_head_kv * head_dim, n, &mut seed);
            w.mat(&format!("{p}attn_output.weight"), n, n_head * head_dim, &mut seed);
            w.vec_f32(&format!("{p}attn_q_norm.weight"), head_dim, |i| {
                1.0 + i as f32 * 0.01
            });
            w.vec_f32(&format!("{p}attn_k_norm.weight"), head_dim, |i| {
                1.0 + i as f32 * 0.01
            });
        } else {
            w.mat(&format!("{p}attn_qkv.weight"), channels, n, &mut seed);
            w.mat(&format!("{p}attn_gate.weight"), value_dim, n, &mut seed);
            w.mat(&format!("{p}ssm_out.weight"), n, value_dim, &mut seed);
            w.tensor_f32(
                &format!("{p}ssm_conv1d.weight"),
                &[d_conv as u64, channels as u64],
                &mut seed,
            );
            w.mat(&format!("{p}ssm_alpha.weight"), hv, n, &mut seed);
            w.mat(&format!("{p}ssm_beta.weight"), hv, n, &mut seed);
            w.vec_f32(&format!("{p}ssm_a"), hv, |_| -0.5);
            w.vec_f32(&format!("{p}ssm_dt.bias"), hv, |_| 0.1);
            w.vec_f32(&format!("{p}ssm_norm.weight"), s, |i| 1.0 + i as f32 * 0.01);
        }
        w.mat(&format!("{p}ffn_gate_inp.weight"), n_expert, n, &mut seed);
        w.tensor_f32(
            &format!("{p}ffn_gate_exps.weight"),
            &[n_ff as u64, n as u64, n_expert as u64],
            &mut seed,
        );
        w.tensor_f32(
            &format!("{p}ffn_up_exps.weight"),
            &[n_ff as u64, n as u64, n_expert as u64],
            &mut seed,
        );
        w.tensor_f32(
            &format!("{p}ffn_down_exps.weight"),
            &[n as u64, n_ff as u64, n_expert as u64],
            &mut seed,
        );
        w.mat(&format!("{p}ffn_gate_shexp.weight"), n_ff, n, &mut seed);
        w.mat(&format!("{p}ffn_up_shexp.weight"), n_ff, n, &mut seed);
        w.mat(&format!("{p}ffn_down_shexp.weight"), n, n_ff, &mut seed);
        w.tensor_f32(
            &format!("{p}ffn_gate_inp_shexp.weight"),
            &[n as u64, 1],
            &mut seed,
        );
        w.vec_f32(&format!("{p}attn_norm.weight"), n, |i| 1.0 + i as f32 * 0.01);
        w.vec_f32(&format!("{p}post_attention_norm.weight"), n, |i| {
            1.0 + i as f32 * 0.01
        });
    }
    w.tensor_f32("token_embd.weight", &[n as u64, n_vocab], &mut seed);
    w.tensor_f32("output.weight", &[n as u64, n_vocab], &mut seed);
    w.vec_f32("output_norm.weight", n, |i| 1.0 + i as f32 * 0.01);
    w.finish()
}

/// Writes the CLI's fixture files. Opt-in: the files are checked in, and this
/// only runs when `STRATA_WRITE_FIXTURES=1` is set, so a normal `cargo test`
/// The fixture vocabulary: the 256 byte-level characters (so any UTF-8 text
/// encodes) plus one EOS special token.
fn byte_vocab() -> (Vec<String>, Vec<i32>) {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|b| strata_tokenizer::byte_level::byte_to_char(b).to_string())
        .collect();
    tokens.push("<|endoftext|>".to_string());
    let mut types = vec![1i32; 256];
    types.push(3); // the EOS token is a control token
    (tokens, types)
}

#[test]
fn write_cli_fixtures() {
    if std::env::var("STRATA_WRITE_FIXTURES").is_err() {
        return;
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-cli/tests/data");
    std::fs::create_dir_all(&dir).expect("fixture dir");
    std::fs::write(dir.join("tiny-qwen4exp.gguf"), build_qwen4exp()).expect("write qwen4exp");
    std::fs::write(dir.join("tiny-qwen35moe.gguf"), build_qwen35moe()).expect("write qwen35moe");
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("strata-rs-tiny-{name}.gguf"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(bytes).unwrap();
    p
}

fn load(name: &str, bytes: &[u8]) -> Arc<Engine> {
    let path = write_temp(name, bytes);
    let engine = Engine::load(&path).expect("engine loads");
    std::fs::remove_file(&path).ok();
    engine
}

#[test]
fn qwen4exp_tiny_end_to_end() {
    let engine = load("qwen4exp", &build_qwen4exp());
    // The session is shared by the assertions below.
    assert_eq!(engine.geometry.n_layers, 4);
    assert_eq!(engine.geometry.n_vocab, 257, "256 byte tokens + EOS");
    assert!(engine.geometry.has_hyper_connections());

    let mut session = Session::new(Arc::clone(&engine));
    // A few tokens through the network.
    let mut logits = engine.forward(&mut session, 1);
    assert_eq!(logits.len(), engine.geometry.n_vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite");
    let first = logits.clone();

    // Determinism: the same token at the same position gives the same logits.
    let mut session2 = Session::new(Arc::clone(&engine));
    let again = engine.forward(&mut session2, 1);
    assert_eq!(first, again, "the forward pass must be deterministic");

    // A different token gives different logits.
    logits = engine.forward(&mut session, 2);
    assert_ne!(first, logits, "different tokens must give different logits");

    // The sampler picks a token.
    session.tokens.extend([1, 2]);
    let pick = sample(&mut logits, &session.tokens, &SamplerParams::default(), 0);
    assert!(
        pick < engine.geometry.n_vocab as u32,
        "the sampler must pick a vocabulary token"
    );
}

#[test]
fn qwen35moe_tiny_end_to_end() {
    let engine = load("qwen35moe", &build_qwen35moe());
    assert_eq!(engine.geometry.n_layers, 4);
    assert_eq!(engine.geometry.n_mtp_layers, 1);
    assert!(!engine.geometry.has_hyper_connections());
    assert!(!engine.geometry.has_indexer(), "qwen35moe attention is dense");

    let mut session = Session::new(Arc::clone(&engine));
    let logits = engine.forward(&mut session, 3);
    assert_eq!(logits.len(), engine.geometry.n_vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite");

    let mut session2 = Session::new(Arc::clone(&engine));
    let again = engine.forward(&mut session2, 3);
    assert_eq!(logits, again, "the forward pass must be deterministic");
}

#[test]
fn generation_is_reproducible_and_bounded() {
    let engine = load("qwen4exp-gen", &build_qwen4exp());
    let params = SamplerParams {
        temperature: 1.0,
        seed: 7,
        ..SamplerParams::default()
    };
    let run = |engine: &Arc<Engine>| -> Vec<u32> {
        let mut s = Session::new(Arc::clone(engine));
        let mut out = Vec::new();
        let mut token = 1u32;
        for _ in 0..12 {
            let mut logits = engine.forward(&mut s, token);
            let next = sample(&mut logits, &s.tokens, &params, out.len() as u64);
            out.push(next);
            s.tokens.push(next as i32);
            token = next;
        }
        out
    };
    let a = run(&engine);
    let b = run(&engine);
    assert_eq!(a, b, "a seeded run must be reproducible");
    assert!(
        a.iter().all(|&t| t < engine.geometry.n_vocab as u32),
        "every token is in vocabulary"
    );
}
