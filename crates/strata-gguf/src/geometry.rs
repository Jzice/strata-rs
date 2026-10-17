//! The model geometry: the shape a model file declares, read from its own
//! metadata (llama.cpp's key names under the architecture prefix).
//!
//! Defaults are Qwen3.8-Flash-Next's (`qwen4exp`); the engine validates a
//! file against this struct instead of compiling the numbers into kernels.

use crate::error::{Error, Result};
use crate::family::ModelFamily;
use crate::metadata::MetaValue;
use crate::reader::GgufFile;

/// Every field is a dimension the layer implementations depend on.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelGeometry {
    pub family: ModelFamily,
    pub n_embd: i64,       // embedding_length
    pub n_layers: i64,     // block_count (the trunk; MTP blocks counted apart)
    pub n_vocab: i64,      // the length of tokenizer.ggml.tokens
    pub rms_eps: f64,      // attention.layer_norm_rms_epsilon
    pub qsa_interval: i64, // full_attention_interval

    // GDN (the linear-attention layers)
    pub ssm_state_size: i64, // ssm.state_size
    pub ssm_k_heads: i64,    // ssm.group_count
    pub ssm_v_heads: i64,    // ssm.time_step_rank
    pub ssm_d_conv: i64,     // ssm.conv_kernel
    pub ssm_conv_channels: i64,
    pub ssm_value_dim: i64, // ssm.inner_size

    // QSA (the full-attention layers)
    pub n_head: i64,      // attention.head_count
    pub n_head_kv: i64,   // attention.head_count_kv
    pub head_dim: i64,    // attention.key_length (= attention.value_length)
    pub n_rot: i64,       // rope.dimension_count
    pub idx_q_heads: i64, // attention.indexer.head_count (0: dense attention)
    pub idx_key_dim: i64, // attention.indexer.key_length
    pub idx_block: i64,   // the indexer's pooling block (compress ratio)
    pub idx_top_k: i64,   // attention.indexer.top_k

    // gated residual, on every layer
    pub hc: i64,    // hyper_connection.count (0: plain pre-norm residual)
    pub hc_lr: i64, // hyper_connection.low_rank

    // MoE, on every layer
    pub n_expert: i64,      // expert_count
    pub n_expert_used: i64, // expert_used_count
    pub n_ff: i64,          // expert_feed_forward_length
    pub n_ff_shexp: i64,    // the shared expert's FFN width

    // appended for the model families
    pub n_mtp_layers: i64, // MTP blocks inside the file (qwen35moe: blk.40)
}

impl Default for ModelGeometry {
    fn default() -> Self {
        Self::qwen4exp_defaults()
    }
}

impl ModelGeometry {
    /// Qwen3.8-Flash-Next's shape: the values the reference kernels are
    /// compiled for.
    pub fn qwen4exp_defaults() -> Self {
        Self {
            family: ModelFamily::Qwen4Exp,
            n_embd: 2560,
            n_layers: 48,
            n_vocab: 248_320,
            rms_eps: 1e-6,
            qsa_interval: 4,

            ssm_state_size: 128,
            ssm_k_heads: 16,
            ssm_v_heads: 48,
            ssm_d_conv: 4,
            ssm_conv_channels: 10_240,
            ssm_value_dim: 6144,

            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            n_rot: 64,
            idx_q_heads: 4,
            idx_key_dim: 128,
            idx_block: 4,
            idx_top_k: 2048,

            hc: 4,
            hc_lr: 320,

            n_expert: 512,
            n_expert_used: 10,
            n_ff: 640,
            n_ff_shexp: 640,

            n_mtp_layers: 0,
        }
    }

    pub fn has_indexer(&self) -> bool {
        self.idx_q_heads > 0
    }

    pub fn has_hyper_connections(&self) -> bool {
        self.hc > 0
    }

    pub fn hc_dim(&self) -> i64 {
        self.hc * self.n_embd
    }

    /// `layer % qsa_interval == qsa_interval - 1` is full attention.
    pub fn is_qsa_layer(&self, layer: i64) -> bool {
        layer % self.qsa_interval == self.qsa_interval - 1
    }

    pub fn n_qsa_layers(&self) -> i64 {
        self.n_layers / self.qsa_interval
    }

    pub fn n_gdn_layers(&self) -> i64 {
        self.n_layers - self.n_qsa_layers()
    }
}

/// The integer metadata keys of the `qwen4exp` geometry, with their fields.
struct IntKey {
    key: &'static str,
    field: fn(&mut ModelGeometry) -> &mut i64,
    required: bool,
}

macro_rules! field {
    ($f:ident) => {
        |g: &mut ModelGeometry| &mut g.$f
    };
}

fn qwen4exp_keys() -> Vec<IntKey> {
    vec![
        IntKey {
            key: "block_count",
            field: field!(n_layers),
            required: true,
        },
        IntKey {
            key: "embedding_length",
            field: field!(n_embd),
            required: true,
        },
        IntKey {
            key: "expert_count",
            field: field!(n_expert),
            required: true,
        },
        IntKey {
            key: "expert_used_count",
            field: field!(n_expert_used),
            required: true,
        },
        IntKey {
            key: "attention.head_count",
            field: field!(n_head),
            required: true,
        },
        IntKey {
            key: "attention.head_count_kv",
            field: field!(n_head_kv),
            required: true,
        },
        IntKey {
            key: "full_attention_interval",
            field: field!(qsa_interval),
            required: false,
        },
        IntKey {
            key: "ssm.state_size",
            field: field!(ssm_state_size),
            required: false,
        },
        IntKey {
            key: "ssm.group_count",
            field: field!(ssm_k_heads),
            required: false,
        },
        IntKey {
            key: "ssm.time_step_rank",
            field: field!(ssm_v_heads),
            required: false,
        },
        IntKey {
            key: "ssm.conv_kernel",
            field: field!(ssm_d_conv),
            required: false,
        },
        IntKey {
            key: "ssm.inner_size",
            field: field!(ssm_value_dim),
            required: false,
        },
        IntKey {
            key: "attention.key_length",
            field: field!(head_dim),
            required: false,
        },
        IntKey {
            key: "rope.dimension_count",
            field: field!(n_rot),
            required: false,
        },
        IntKey {
            key: "attention.indexer.head_count",
            field: field!(idx_q_heads),
            required: false,
        },
        IntKey {
            key: "attention.indexer.key_length",
            field: field!(idx_key_dim),
            required: false,
        },
        IntKey {
            key: "attention.indexer.top_k",
            field: field!(idx_top_k),
            required: false,
        },
        IntKey {
            key: "hyper_connection.count",
            field: field!(hc),
            required: false,
        },
        IntKey {
            key: "hyper_connection.low_rank",
            field: field!(hc_lr),
            required: false,
        },
        IntKey {
            key: "expert_feed_forward_length",
            field: field!(n_ff),
            required: false,
        },
    ]
}

/// Fills `g` from the metadata shard `f`. Returns the first problem, or `""`.
/// Another architecture is refused by name first: its keys carry another
/// prefix, and "missing qwen4exp.block_count" would hide what is wrong.
pub fn read_geometry(f: &GgufFile, g: &mut ModelGeometry) -> String {
    *g = ModelGeometry::qwen4exp_defaults();
    let arch = match f.get("general.architecture") {
        Some(v) => v.as_str().unwrap_or_default().to_string(),
        None => return "missing general.architecture".to_string(),
    };
    if arch != "qwen4exp" {
        return format!("architecture is '{arch}', this engine requires 'qwen4exp'");
    }
    g.family = ModelFamily::from_arch(&arch);
    let p = "qwen4exp.";

    let read_int = |key: &str, out: &mut i64, required: bool| -> Result<bool> {
        match f.get(&format!("{p}{key}")) {
            None => {
                if required {
                    return Err(Error::MissingMetadata {
                        key: format!("{p}{key}"),
                    });
                }
                Ok(true)
            }
            Some(v) => {
                let n = v.as_int().ok_or_else(|| Error::WrongMetadataType {
                    key: format!("{p}{key}"),
                })?;
                if n <= 0 {
                    return Err(Error::UnsupportedGeometry {
                        message: format!("{p}{key} = {n}, it must be positive"),
                    });
                }
                *out = n;
                Ok(true)
            }
        }
    };

    for k in qwen4exp_keys() {
        let field = (k.field)(g);
        match read_int(k.key, field, k.required) {
            Ok(true) => {}
            Ok(false) => {}
            Err(e) => return e.to_string(),
        }
    }

    let mut value_length = g.head_dim;
    let mut ff_shexp = g.n_ff;
    if let Err(e) = read_int("attention.value_length", &mut value_length, false) {
        return e.to_string();
    }
    if let Err(e) = read_int("expert_shared_feed_forward_length", &mut ff_shexp, false) {
        return e.to_string();
    }
    if value_length != g.head_dim {
        return format!(
            "{p}attention.value_length = {value_length} and attention.key_length = {}: the attention kernels use one head width for both",
            g.head_dim
        );
    }
    if ff_shexp != g.n_ff {
        return format!(
            "{p}expert_shared_feed_forward_length = {ff_shexp} and expert_feed_forward_length = {}: the shared expert is read with the routed experts' width",
            g.n_ff
        );
    }
    g.n_ff_shexp = ff_shexp;

    if let Some(v) = f.get(&format!("{p}attention.layer_norm_rms_epsilon")) {
        match v.as_float() {
            Some(x) if x > 0.0 => g.rms_eps = x,
            _ => return format!("{p}attention.layer_norm_rms_epsilon must be a positive number"),
        }
    }
    if let Some(v) = f.get("tokenizer.ggml.tokens") {
        if v.ty != crate::metadata::MetaType::Array || v.count == 0 {
            return "tokenizer.ggml.tokens is not a list of tokens".to_string();
        }
        g.n_vocab = v.count as i64;
    }

    if g.n_head % g.n_head_kv != 0 {
        return format!(
            "{p}attention.head_count = {} is not a multiple of head_count_kv = {}",
            g.n_head, g.n_head_kv
        );
    }
    // GDN's value heads are as wide as its key heads (the state is square)
    if g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads {
        return format!(
            "{p}ssm.inner_size = {}, but ssm.state_size x ssm.time_step_rank = {}: the GDN kernels take square heads",
            g.ssm_value_dim,
            g.ssm_state_size * g.ssm_v_heads
        );
    }
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    if g.n_expert_used > g.n_expert || g.n_expert_used > 64 {
        return format!(
            "{p}expert_used_count = {}, the router takes 1..{} of {} experts",
            g.n_expert_used,
            g.n_expert.min(64),
            g.n_expert
        );
    }
    if g.n_rot > g.head_dim || g.n_rot > g.idx_key_dim || g.n_rot % 2 != 0 {
        return format!(
            "{p}rope.dimension_count = {} is not an even slice of the head widths",
            g.n_rot
        );
    }

    // The layer split, twice over: the interval, and a ratio per layer that is
    // non-zero exactly on the full-attention layers.
    if let Some(v) = f.get(&format!("{p}attention.compress_ratios")) {
        if v.ty != crate::metadata::MetaType::Array
            || v.count != g.n_layers as u64
            || v.items.len() != v.count as usize
        {
            return format!(
                "{p}attention.compress_ratios must list one entry for each of the {} layers",
                g.n_layers
            );
        }
        let mut block = 0i64;
        for l in 0..g.n_layers {
            let r = v.items[l as usize].as_int().unwrap_or(-1);
            if (r != 0) != g.is_qsa_layer(l) || r < 0 || (block != 0 && r != 0 && r != block) {
                return format!(
                    "{p}attention.compress_ratios[{l}] = {r}, but full_attention_interval = {} makes layer {l} {}",
                    g.qsa_interval,
                    if g.is_qsa_layer(l) {
                        "full attention"
                    } else {
                        "linear attention"
                    }
                );
            }
            if r != 0 {
                block = r;
            }
        }
        if block != 0 {
            g.idx_block = block;
        }
    }
    String::new()
}

/// The `qwen35moe` geometry from its metadata. Every key the engine reads
/// must be present (`check_architecture_qwen35moe` guards that); this fills
/// the struct from them. Plain pre-norm residual (`hc = 0`), dense attention
/// (no indexer), one MTP block after the trunk.
pub fn read_geometry_qwen35moe(f: &GgufFile, g: &mut ModelGeometry) -> std::result::Result<(), String> {
    let read = |key: &str| -> std::result::Result<i64, String> {
        f.get(key)
            .and_then(crate::metadata::MetaValue::as_int)
            .ok_or_else(|| format!("missing or non-numeric {key}"))
    };
    let p = "qwen35moe.";
    *g = ModelGeometry::qwen4exp_defaults();
    g.family = ModelFamily::Qwen35Moe;
    g.hc = 0;
    g.hc_lr = 0;
    g.idx_q_heads = 0;
    g.idx_key_dim = 0;
    g.idx_block = 0;
    g.idx_top_k = 0;

    let block_count = read(&format!("{p}block_count"))?;
    let nextn = read(&format!("{p}nextn_predict_layers"))?;
    g.n_layers = block_count - nextn;
    g.n_mtp_layers = nextn;
    g.n_embd = read(&format!("{p}embedding_length"))?;
    g.n_head = read(&format!("{p}attention.head_count"))?;
    g.n_head_kv = read(&format!("{p}attention.head_count_kv"))?;
    g.head_dim = read(&format!("{p}attention.key_length"))?;
    let value_length = read(&format!("{p}attention.value_length"))?;
    if value_length != g.head_dim {
        return Err(format!(
            "{p}attention.value_length = {value_length} and attention.key_length = {}: the attention kernels use one head width for both",
            g.head_dim
        ));
    }
    g.n_expert = read(&format!("{p}expert_count"))?;
    g.n_expert_used = read(&format!("{p}expert_used_count"))?;
    g.n_ff = read(&format!("{p}expert_feed_forward_length"))?;
    g.n_ff_shexp = read(&format!("{p}expert_shared_feed_forward_length"))?;
    g.ssm_d_conv = read(&format!("{p}ssm.conv_kernel"))?;
    g.ssm_state_size = read(&format!("{p}ssm.state_size"))?;
    g.ssm_k_heads = read(&format!("{p}ssm.group_count"))?;
    g.ssm_v_heads = read(&format!("{p}ssm.time_step_rank"))?;
    g.ssm_value_dim = read(&format!("{p}ssm.inner_size"))?;
    g.qsa_interval = read(&format!("{p}full_attention_interval"))?;
    if let Some(v) = f.get(&format!("{p}attention.layer_norm_rms_epsilon")) {
        match v.as_float() {
            Some(x) if x > 0.0 => g.rms_eps = x,
            _ => {
                return Err(format!(
                    "{p}attention.layer_norm_rms_epsilon must be a positive number"
                ))
            }
        }
    }
    if let Some(v) = f.get(&format!("{p}rope.dimension_count")) {
        if let Some(n) = v.as_int() {
            g.n_rot = n;
        }
    }
    // The rotated slice never exceeds the head width.
    g.n_rot = g.n_rot.min(g.head_dim).max(2) & !1;
    if let Some(v) = f.get("tokenizer.ggml.tokens") {
        if v.ty != crate::metadata::MetaType::Array || v.count == 0 {
            return Err("tokenizer.ggml.tokens is not a list of tokens".to_string());
        }
        g.n_vocab = v.count as i64;
    }
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    Ok(())
}

/// The `qwen35moe` guard: every key its geometry is read from must be present.
pub fn check_architecture_qwen35moe(g: &GgufFile) -> String {
    const KEYS: &[&str] = &[
        "qwen35moe.block_count",
        "qwen35moe.embedding_length",
        "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv",
        "qwen35moe.attention.key_length",
        "qwen35moe.attention.value_length",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.expert_feed_forward_length",
        "qwen35moe.expert_shared_feed_forward_length",
        "qwen35moe.ssm.conv_kernel",
        "qwen35moe.ssm.state_size",
        "qwen35moe.ssm.group_count",
        "qwen35moe.ssm.time_step_rank",
        "qwen35moe.ssm.inner_size",
        "qwen35moe.full_attention_interval",
        "qwen35moe.nextn_predict_layers",
    ];
    for k in KEYS {
        match g.get(k) {
            None => return format!("missing {k}"),
            Some(v) if v.as_int().is_none() => return format!("{k} is not a number"),
            _ => {}
        }
    }
    let nl = g
        .get("qwen35moe.block_count")
        .and_then(MetaValue::as_int)
        .unwrap_or(0);
    let mtp = g
        .get("qwen35moe.nextn_predict_layers")
        .and_then(MetaValue::as_int)
        .unwrap_or(0);
    if mtp >= nl {
        return "qwen35moe.nextn_predict_layers is not smaller than qwen35moe.block_count".to_string();
    }
    if g.get("qwen35moe.full_attention_interval")
        .and_then(MetaValue::as_int)
        .unwrap_or(0)
        < 2
    {
        return "qwen35moe.full_attention_interval must be at least 2 (the GDN / attention pattern)"
            .to_string();
    }
    String::new()
}

/// The `qwen35` geometry (dense GDN + gated-attention hybrid): plain pre-norm
/// residual, dense SwiGLU FFN, no MoE / MTP / hyper-connections. Shares the
/// GDN + QSA layer split with `qwen4exp` but is read from the `qwen35.` prefix
/// and pins no expert/MTP fields.
pub fn read_geometry_qwen35(f: &GgufFile, g: &mut ModelGeometry) -> std::result::Result<(), String> {
    let read = |key: &str| -> std::result::Result<i64, String> {
        f.get(key)
            .and_then(crate::metadata::MetaValue::as_int)
            .ok_or_else(|| format!("missing or non-numeric {key}"))
    };
    let p = "qwen35.";
    *g = ModelGeometry::qwen4exp_defaults();
    g.family = ModelFamily::Qwen35;
    g.hc = 0;
    g.hc_lr = 0;
    g.idx_q_heads = 0;
    g.idx_key_dim = 0;
    g.idx_block = 0;
    g.idx_top_k = 0;
    g.n_expert = 0;
    g.n_expert_used = 0;
    g.n_mtp_layers = 0;

    g.n_layers = read(&format!("{p}block_count"))?;
    g.n_embd = read(&format!("{p}embedding_length"))?;
    g.n_head = read(&format!("{p}attention.head_count"))?;
    g.n_head_kv = read(&format!("{p}attention.head_count_kv"))?;
    g.head_dim = read(&format!("{p}attention.key_length"))?;
    let value_length = read(&format!("{p}attention.value_length"))?;
    if value_length != g.head_dim {
        return Err(format!(
            "{p}attention.value_length = {value_length} and attention.key_length = {}: the attention kernels use one head width for both",
            g.head_dim
        ));
    }
    g.n_ff = read(&format!("{p}feed_forward_length"))?;
    g.n_ff_shexp = g.n_ff;
    g.ssm_d_conv = read(&format!("{p}ssm.conv_kernel"))?;
    g.ssm_state_size = read(&format!("{p}ssm.state_size"))?;
    g.ssm_k_heads = read(&format!("{p}ssm.group_count"))?;
    g.ssm_v_heads = read(&format!("{p}ssm.time_step_rank"))?;
    g.ssm_value_dim = read(&format!("{p}ssm.inner_size"))?;
    g.qsa_interval = read(&format!("{p}full_attention_interval"))?;
    if let Some(v) = f.get(&format!("{p}attention.layer_norm_rms_epsilon")) {
        match v.as_float() {
            Some(x) if x > 0.0 => g.rms_eps = x,
            _ => return Err(format!("{p}attention.layer_norm_rms_epsilon must be a positive number")),
        }
    }
    if let Some(v) = f.get(&format!("{p}rope.dimension_count")) {
        if let Some(n) = v.as_int() {
            g.n_rot = n;
        }
    }
    g.n_rot = g.n_rot.min(g.head_dim).max(2) & !1;
    if let Some(v) = f.get("tokenizer.ggml.tokens") {
        if v.ty != crate::metadata::MetaType::Array || v.count == 0 {
            return Err("tokenizer.ggml.tokens is not a list of tokens".to_string());
        }
        g.n_vocab = v.count as i64;
    }
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    Ok(())
}

/// The `qwen35` guard: every key its geometry is read from must be present.
pub fn check_architecture_qwen35(f: &GgufFile) -> String {
    const KEYS: &[&str] = &[
        "qwen35.block_count",
        "qwen35.embedding_length",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
        "qwen35.feed_forward_length",
        "qwen35.ssm.conv_kernel",
        "qwen35.ssm.state_size",
        "qwen35.ssm.group_count",
        "qwen35.ssm.time_step_rank",
        "qwen35.ssm.inner_size",
        "qwen35.full_attention_interval",
    ];
    for k in KEYS {
        match f.get(k) {
            None => return format!("missing {k}"),
            Some(v) if v.as_int().is_none() => return format!("{k} is not a number"),
            _ => {}
        }
    }
    let interval = f
        .get("qwen35.full_attention_interval")
        .and_then(crate::metadata::MetaValue::as_int)
        .unwrap_or(0);
    if interval < 2 {
        return "qwen35.full_attention_interval must be at least 2 (the GDN / attention pattern)".to_string();
    }
    String::new()
}

/// The architecture guard: `""` when the engine can read the model whose
/// metadata shard is `f`.
pub fn check_architecture(f: &GgufFile) -> String {
    let arch = match f.get("general.architecture") {
        Some(v) => v.as_str().unwrap_or_default().to_string(),
        None => return "missing general.architecture".to_string(),
    };
    let family = ModelFamily::from_arch(&arch);
    if family == ModelFamily::Qwen35Moe {
        return check_architecture_qwen35moe(f);
    }
    if family == ModelFamily::Qwen35 {
        return check_architecture_qwen35(f);
    }
    let mut g = ModelGeometry::default();
    let err = read_geometry(f, &mut g);
    if err.is_empty() {
        return geometry_unsupported(&g);
    }
    err
}

/// `""` when the compiled shapes can run `g`, else the first key whose value
/// they are not compiled for. Only the expert count, the experts per token
/// and the vocabulary are free.
pub fn geometry_unsupported(g: &ModelGeometry) -> String {
    let c = ModelGeometry::qwen4exp_defaults();
    struct Pin {
        key: &'static str,
        get: fn(&ModelGeometry) -> i64,
    }
    let pins = [
        Pin {
            key: "block_count",
            get: |g| g.n_layers,
        },
        Pin {
            key: "embedding_length",
            get: |g| g.n_embd,
        },
        Pin {
            key: "full_attention_interval",
            get: |g| g.qsa_interval,
        },
        Pin {
            key: "ssm.state_size",
            get: |g| g.ssm_state_size,
        },
        Pin {
            key: "ssm.group_count",
            get: |g| g.ssm_k_heads,
        },
        Pin {
            key: "ssm.time_step_rank",
            get: |g| g.ssm_v_heads,
        },
        Pin {
            key: "ssm.conv_kernel",
            get: |g| g.ssm_d_conv,
        },
        Pin {
            key: "ssm.inner_size",
            get: |g| g.ssm_value_dim,
        },
        Pin {
            key: "attention.head_count",
            get: |g| g.n_head,
        },
        Pin {
            key: "attention.head_count_kv",
            get: |g| g.n_head_kv,
        },
        Pin {
            key: "attention.key_length",
            get: |g| g.head_dim,
        },
        Pin {
            key: "rope.dimension_count",
            get: |g| g.n_rot,
        },
        Pin {
            key: "attention.indexer.head_count",
            get: |g| g.idx_q_heads,
        },
        Pin {
            key: "attention.indexer.key_length",
            get: |g| g.idx_key_dim,
        },
        Pin {
            key: "attention.compress_ratios",
            get: |g| g.idx_block,
        },
        Pin {
            key: "attention.indexer.top_k",
            get: |g| g.idx_top_k,
        },
        Pin {
            key: "hyper_connection.count",
            get: |g| g.hc,
        },
        Pin {
            key: "hyper_connection.low_rank",
            get: |g| g.hc_lr,
        },
        Pin {
            key: "expert_feed_forward_length",
            get: |g| g.n_ff,
        },
    ];
    for pin in pins {
        if (pin.get)(g) != (pin.get)(&c) {
            return format!(
                "qwen4exp.{} = {}, the kernels are compiled for {}",
                pin.key,
                (pin.get)(g),
                (pin.get)(&c)
            );
        }
    }
    if (g.rms_eps - c.rms_eps).abs() > 1e-6 * c.rms_eps {
        return format!(
            "qwen4exp.attention.layer_norm_rms_epsilon = {:e}, the kernels are compiled for {:e}",
            g.rms_eps, c.rms_eps
        );
    }
    String::new()
}
