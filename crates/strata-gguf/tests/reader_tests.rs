//! Reader tests: a synthesized GGUF is written to a temp file and parsed back,
//! covering metadata value types, tensor directory, alignment, the model
//! geometry reader and the split-shard path resolution.

// Exact float comparison is the point of the known-answer tests.
#![allow(clippy::float_cmp)]

use half::f16;
use std::io::Write;
use std::path::{Path, PathBuf};
use strata_gguf::family::ModelFamily;
use strata_gguf::geometry::{check_architecture, read_geometry, ModelGeometry};
use strata_gguf::metadata::MetaType;
use strata_gguf::quant::QuantType;
use strata_gguf::reader::{split_paths, GgufFile, GgufModel};

/// A minimal GGUF writer for the tests.
struct GgufWriter {
    buf: Vec<u8>,
    n_tensors: u64,
    n_kv: u64,
    kv_bytes: Vec<u8>,
    tensor_bytes: Vec<u8>,
    tensor_infos: Vec<u8>,
    data_offset: u64,
}

impl GgufWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            n_tensors: 0,
            n_kv: 0,
            kv_bytes: Vec::new(),
            tensor_bytes: Vec::new(),
            tensor_infos: Vec::new(),
            data_offset: 0,
        }
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn kv(&mut self, key: &str, ty: MetaType, v: &MetaValueDef) {
        self.n_kv += 1;
        let mut b = Vec::new();
        Self::push_string(&mut b, key);
        let mut vb = Vec::new();
        Self::push_value(&mut vb, ty, v);
        b.extend_from_slice(&vb);
        self.kv_bytes.extend_from_slice(&b);
    }

    fn push_string(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn push_value(b: &mut Vec<u8>, ty: MetaType, v: &MetaValueDef) {
        b.extend_from_slice(&(ty as u32).to_le_bytes());
        Self::push_payload(b, ty, v);
    }

    /// The value's bytes only (no type tag; array items carry no tag).
    fn push_payload(b: &mut Vec<u8>, ty: MetaType, v: &MetaValueDef) {
        match (ty, v) {
            (MetaType::U32, MetaValueDef::U32(x)) => b.extend_from_slice(&x.to_le_bytes()),
            (MetaType::I32, MetaValueDef::I32(x)) => b.extend_from_slice(&(*x as u32).to_le_bytes()),
            (MetaType::F32, MetaValueDef::F32(x)) => b.extend_from_slice(&x.to_le_bytes()),
            (MetaType::String, MetaValueDef::Str(s)) => Self::push_string(b, s),
            (MetaType::Array, MetaValueDef::Array(elem, items)) => {
                b.extend_from_slice(&(*elem as u32).to_le_bytes());
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    Self::push_payload(b, *elem, item);
                }
            }
            _ => panic!("mismatched value type"),
        }
    }

    fn tensor(&mut self, name: &str, shape: &[u64], ty: u32, data: &[u8]) {
        self.n_tensors += 1;
        Self::push_string(&mut self.tensor_infos, name);
        self.tensor_infos
            .extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            self.tensor_infos.extend_from_slice(&d.to_le_bytes());
        }
        self.tensor_infos.extend_from_slice(&ty.to_le_bytes());
        self.tensor_infos
            .extend_from_slice(&self.data_offset.to_le_bytes());
        self.data_offset += data.len() as u64;
        self.tensor_bytes.extend_from_slice(data);
    }

    fn finish(mut self) -> Vec<u8> {
        self.buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        self.u32(3);
        self.u64(self.n_tensors);
        self.u64(self.n_kv);
        self.buf.extend_from_slice(&self.kv_bytes);
        self.buf.extend_from_slice(&self.tensor_infos);
        while self.buf.len() % 32 != 0 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(&self.tensor_bytes);
        self.buf
    }
}

#[derive(Clone)]
enum MetaValueDef {
    U32(u32),
    I32(i32),
    F32(f32),
    Str(String),
    Array(MetaType, Vec<MetaValueDef>),
}

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("strata-rs-test-{name}.gguf"));
    p
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = temp_path(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(bytes).unwrap();
    p
}

fn qwen4exp_metadata(w: &mut GgufWriter) {
    w.kv(
        "general.architecture",
        MetaType::String,
        &MetaValueDef::Str("qwen4exp".into()),
    );
    w.kv("general.alignment", MetaType::U32, &MetaValueDef::U32(32));
    w.kv("qwen4exp.block_count", MetaType::U32, &MetaValueDef::U32(48));
    w.kv(
        "qwen4exp.embedding_length",
        MetaType::U32,
        &MetaValueDef::U32(2560),
    );
    w.kv("qwen4exp.expert_count", MetaType::U32, &MetaValueDef::U32(512));
    w.kv(
        "qwen4exp.expert_used_count",
        MetaType::U32,
        &MetaValueDef::U32(10),
    );
    w.kv(
        "qwen4exp.attention.head_count",
        MetaType::U32,
        &MetaValueDef::U32(24),
    );
    w.kv(
        "qwen4exp.attention.head_count_kv",
        MetaType::U32,
        &MetaValueDef::U32(2),
    );
    w.kv(
        "qwen4exp.attention.layer_norm_rms_epsilon",
        MetaType::F32,
        &MetaValueDef::F32(1e-6),
    );
    w.kv(
        "qwen4exp.attention.compress_ratios",
        MetaType::Array,
        &MetaValueDef::Array(
            MetaType::I32,
            (0..48)
                .map(|l| MetaValueDef::I32(if l % 4 == 3 { 4 } else { 0 }))
                .collect(),
        ),
    );
    w.kv(
        "tokenizer.ggml.tokens",
        MetaType::Array,
        &MetaValueDef::Array(
            MetaType::String,
            vec![
                MetaValueDef::Str("<|endoftext|>".into()),
                MetaValueDef::Str("hello".into()),
                MetaValueDef::Str(" world".into()),
            ],
        ),
    );
    w.kv(
        "tokenizer.ggml.model",
        MetaType::String,
        &MetaValueDef::Str("gpt2".into()),
    );
}

#[test]
fn reads_back_metadata_and_tensors() {
    let mut w = GgufWriter::new();
    qwen4exp_metadata(&mut w);
    let mut data = vec![0u8; 34];
    data[..2].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
    data[2] = 5;
    w.tensor("blk.0.attn_q.weight", &[32], QuantType::Q8_0.id(), &data);
    let bytes = w.finish();
    let path = write_temp("read-back", &bytes);

    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.version(), 3);
    assert_eq!(f.get("general.architecture").unwrap().as_str(), Some("qwen4exp"));
    assert_eq!(f.get("qwen4exp.block_count").unwrap().as_int(), Some(48));
    assert_eq!(f.get("tokenizer.ggml.tokens").unwrap().count, 3);
    let eps = f.get("qwen4exp.attention.layer_norm_rms_epsilon").unwrap();
    let eps = eps.as_float().unwrap();
    assert!((eps - 1e-6).abs() < 1e-12);

    let t = f.find("blk.0.attn_q.weight").unwrap();
    assert_eq!(t.shape, vec![32]);
    assert_eq!(t.ty, QuantType::Q8_0.id());
    let td = f.tensor_data(t).unwrap();
    assert_eq!(td[2], 5);

    let mut out = [0.0f32; 32];
    strata_gguf::dequant::dequantize_row(QuantType::Q8_0, &td[..34], &mut out).unwrap();
    assert_eq!(out[0], 10.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn geometry_reads_from_metadata() {
    let mut w = GgufWriter::new();
    qwen4exp_metadata(&mut w);
    let bytes = w.finish();
    let path = write_temp("geometry", &bytes);

    let f = GgufFile::open(&path).unwrap();
    let mut g = ModelGeometry::default();
    let err = read_geometry(&f, &mut g);
    assert!(err.is_empty(), "qwen4exp metadata must read cleanly: {err}");
    assert_eq!(g.family, ModelFamily::Qwen4Exp);
    assert_eq!(g.n_layers, 48);
    assert_eq!(g.n_embd, 2560);
    assert_eq!(g.n_vocab, 3);
    assert_eq!(g.idx_block, 4);
    assert!((g.rms_eps - 1e-6).abs() < 1e-12);
    assert!(g.is_qsa_layer(3) && !g.is_qsa_layer(2));
    assert_eq!(g.n_qsa_layers(), 12);
    assert_eq!(g.n_gdn_layers(), 36);
    assert_eq!(check_architecture(&f), "");

    std::fs::remove_file(&path).ok();
}

#[test]
fn refuses_other_architectures_by_name() {
    let mut w = GgufWriter::new();
    w.kv(
        "general.architecture",
        MetaType::String,
        &MetaValueDef::Str("llama".into()),
    );
    let bytes = w.finish();
    let path = write_temp("other-arch", &bytes);

    let f = GgufFile::open(&path).unwrap();
    let mut g = ModelGeometry::default();
    let err = read_geometry(&f, &mut g);
    assert!(err.contains("llama"), "the error names the architecture: {err}");

    std::fs::remove_file(&path).ok();
}

#[test]
fn bad_magic_is_refused() {
    let bytes = b"NOTGGUF and more bytes................";
    let path = write_temp("bad-magic", bytes);
    let err = GgufFile::open(&path).expect_err("must refuse");
    assert!(matches!(err, strata_gguf::Error::BadMagic { .. }));
    std::fs::remove_file(&path).ok();
}

#[test]
fn split_paths_resolves_siblings() {
    let single = Path::new("D:/models/qwen4exp-q2_0.gguf");
    assert_eq!(split_paths(single).unwrap(), vec![single.to_path_buf()]);

    let missing = Path::new("D:/nope/model-00001-of-00002.gguf");
    let err = split_paths(missing).expect_err("a missing shard is an error");
    assert!(err.to_string().contains("00002"), "{err}");
}

#[test]
fn multi_shard_model_reads_tensors_across_shards() {
    let mut w1 = GgufWriter::new();
    qwen4exp_metadata(&mut w1);
    w1.kv("split.count", MetaType::U32, &MetaValueDef::U32(2));
    w1.kv("split.no", MetaType::U32, &MetaValueDef::U32(0));
    w1.kv("split.tensors.count", MetaType::U32, &MetaValueDef::U32(2));
    w1.tensor("token_embd.weight", &[32], QuantType::F32.id(), &[0u8; 128]);

    let mut w2 = GgufWriter::new();
    w2.kv("split.count", MetaType::U32, &MetaValueDef::U32(2));
    w2.kv("split.no", MetaType::U32, &MetaValueDef::U32(1));
    w2.kv("split.tensors.count", MetaType::U32, &MetaValueDef::U32(2));
    w2.tensor("output.weight", &[32], QuantType::F32.id(), &[1u8; 128]);

    let base = std::env::temp_dir();
    let stem = base.join("strata-rs-test-model");
    let p1 = PathBuf::from(format!("{}-00001-of-00002.gguf", stem.display()));
    let p2 = PathBuf::from(format!("{}-00002-of-00002.gguf", stem.display()));
    std::fs::write(&p1, w1.finish()).unwrap();
    std::fs::write(&p2, w2.finish()).unwrap();

    let model = GgufModel::open(&p2).unwrap();
    assert_eq!(model.n_shards(), 2);
    assert_eq!(
        model.meta().get("general.architecture").unwrap().as_str(),
        Some("qwen4exp")
    );
    assert!(model.find("token_embd.weight").is_some());
    assert!(model.find("output.weight").is_some());
    let data = model.tensor_data("output.weight").unwrap().unwrap();
    assert_eq!(data[0], 1);

    std::fs::remove_file(&p1).ok();
    std::fs::remove_file(&p2).ok();
}
