//! The engine: loads a GGUF, owns every layer's weights and runs the forward
//! pass for both model families.
//!
//! qwen4exp keeps a hyper-connection residual stack (`hc` streams); qwen35moe
//! a plain pre-norm residual. The layer pattern (`is_qsa_layer`) decides GDN
//! vs attention per layer in both families.

use crate::backend::QuantTensor;
use crate::error::{Error, Result};
use crate::gdn::GdnLayer;
use crate::hyper::HyperConnection;
use crate::moe::MoeLayer;
use crate::qsa::QsaLayer;
use crate::session::Session;
use crate::weights::WeightStore;
use std::sync::Arc;
use strata_gguf::geometry::{geometry_unsupported, read_geometry};
use strata_gguf::{GgufModel, ModelFamily, ModelGeometry};

/// The per-layer `(attn_norm, post_attention_norm)` pairs of qwen35moe.
pub type NormPairs = Vec<(Arc<Vec<f32>>, Arc<Vec<f32>>)>;

/// Which kind of layer sits at one index.
#[allow(clippy::type_complexity)]
#[derive(Debug, Clone)]
pub enum Layer {
    Gdn(GdnLayer<'static>),
    Qsa(QsaLayer<'static>),
}

/// A dense SwiGLU FFN (`gate` / `up` project to `n_ff`, `down` back to `n_embd`).
pub(crate) struct DenseFfn<'a> {
    pub gate: QuantTensor<'a>,
    pub up: QuantTensor<'a>,
    pub down: QuantTensor<'a>,
}

/// A loaded model, ready to run.
pub struct Engine {
    pub model: &'static GgufModel,
    pub geometry: ModelGeometry,
    pub family: ModelFamily,
    /// Kept for late-bound tensor lookups (MTP and future features).
    #[allow(dead_code)]
    pub(crate) weights: WeightStore<'static>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) moe: Vec<MoeLayer<'static>>,
    /// qwen35's dense SwiGLU FFN weights (empty for the MoE families).
    pub(crate) ffn: Vec<DenseFfn<'static>>,
    hc_attn: Vec<HyperConnection<'static>>,
    hc_ffn: Vec<HyperConnection<'static>>,
    embed: QuantTensor<'static>,
    pub(crate) output: QuantTensor<'static>,
    /// qwen4exp's final hyper-connection read (`output_hc_*`).
    output_hc: Option<HyperConnection<'static>>,
    /// qwen35moe's final norm (`output_norm.weight`).
    pub(crate) output_norm: Option<Arc<Vec<f32>>>,
    /// qwen35moe's per-layer `(attn_norm, post_attention_norm)`.
    pub(crate) qwen35_norms: NormPairs,
    pub tokenizer: Option<strata_tokenizer::Tokenizer>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("family", &self.family)
            .field("n_layers", &self.geometry.n_layers)
            .field("n_embd", &self.geometry.n_embd)
            .field("n_vocab", &self.geometry.n_vocab)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Opens and validates the model at `path`, then loads every weight view.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Arc<Engine>> {
        // The mapping outlives the process; leaking the handle keeps every
        // weight view borrowable as 'static without self-referential tricks.
        let model: &'static GgufModel = Box::leak(Box::new(GgufModel::open(path)?));

        let family = model
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(ModelFamily::from_arch)
            .unwrap_or_default();
        if !family.is_supported() {
            return Err(Error::UnsupportedFamily {
                family: family.not_supported_message(),
            });
        }
        let mut geometry = ModelGeometry::default();
        if family == ModelFamily::Qwen35Moe {
            strata_gguf::geometry::read_geometry_qwen35moe(model.meta(), &mut geometry)
                .map_err(|message| Error::UnsupportedGeometry { message })?;
        } else if family == ModelFamily::Qwen35 {
            strata_gguf::geometry::read_geometry_qwen35(model.meta(), &mut geometry)
                .map_err(|message| Error::UnsupportedGeometry { message })?;
        } else {
            let err = read_geometry(model.meta(), &mut geometry);
            if !err.is_empty() {
                return Err(Error::UnsupportedGeometry { message: err });
            }
            // The reference engine pins its kernels to one shape and refuses
            // anything else; this port is geometry-driven, so a different
            // shape is a warning rather than a refusal.
            let pin = geometry_unsupported(&geometry);
            if !pin.is_empty() {
                tracing::warn!("{pin} (running anyway: this port is geometry-driven)");
            }
        }
        let mut weights = WeightStore::new(model);
        let n_layers = geometry.n_layers as usize;
        let mut layers = Vec::with_capacity(n_layers);
        let mut moe = Vec::with_capacity(n_layers);
        let mut ffn = Vec::with_capacity(n_layers);
        let mut hc_attn = Vec::new();
        let mut hc_ffn = Vec::new();

        for l in 0..n_layers {
            let p = format!("blk.{l}.");
            if geometry.is_qsa_layer(l as i64) {
                layers.push(Layer::Qsa(load_qsa(&mut weights, &geometry, &p, l)?));
            } else {
                layers.push(Layer::Gdn(load_gdn(&mut weights, &geometry, &p)?));
            }
            if family == ModelFamily::Qwen35 {
                // Dense SwiGLU FFN instead of the MoE: gate/up to n_ff, down to n_embd.
                ffn.push(DenseFfn {
                    gate: weights.tensor_required(&format!("{p}ffn_gate.weight"))?,
                    up: weights.tensor_required(&format!("{p}ffn_up.weight"))?,
                    down: weights.tensor_required(&format!("{p}ffn_down.weight"))?,
                });
            } else {
                moe.push(crate::moe::load_layer(
                    &mut weights,
                    l,
                    geometry.n_expert as usize,
                    geometry.n_expert_used as usize,
                )?);
            }
            if geometry.has_hyper_connections() {
                hc_attn.push(load_hc(&mut weights, &geometry, &p, "hc_attn_")?);
                hc_ffn.push(load_hc(&mut weights, &geometry, &p, "hc_ffn_")?);
            }
        }

        let embed = weights.tensor_required("token_embd.weight")?;
        let output = weights.tensor_required("output.weight")?;
        let output_hc = if geometry.has_hyper_connections() {
            Some(load_hc_global(&mut weights, &geometry, "output_hc_")?)
        } else {
            None
        };
        let output_norm = if matches!(family, ModelFamily::Qwen35Moe | ModelFamily::Qwen35) {
            Some(
                weights
                    .f32_vec("output_norm.weight")
                    .ok_or_else(|| Error::MissingTensor {
                        name: "output_norm.weight".to_string(),
                    })?,
            )
        } else {
            None
        };
        let mut qwen35_norms: NormPairs = Vec::new();
        if matches!(family, ModelFamily::Qwen35Moe | ModelFamily::Qwen35) {
            for l in 0..n_layers {
                let p = format!("blk.{l}.");
                let attn_norm =
                    weights
                        .f32_vec(&format!("{p}attn_norm.weight"))
                        .ok_or_else(|| Error::MissingTensor {
                            name: format!("{p}attn_norm.weight"),
                        })?;
                let post_norm = weights
                    .f32_vec(&format!("{p}post_attention_norm.weight"))
                    .ok_or_else(|| Error::MissingTensor {
                        name: format!("{p}post_attention_norm.weight"),
                    })?;
                qwen35_norms.push((attn_norm, post_norm));
            }
        }

        let tokenizer = strata_tokenizer::Tokenizer::from_gguf(model.meta()).ok();

        Ok(Arc::new(Self {
            model,
            geometry,
            family,
            weights,
            layers,
            moe,
            ffn,
            hc_attn,
            hc_ffn,
            embed,
            output,
            output_hc,
            output_norm,
            qwen35_norms,
            tokenizer,
        }))
    }

    pub fn gdn_layer(&self, l: usize) -> Option<&GdnLayer<'static>> {
        match &self.layers[l] {
            Layer::Gdn(g) => Some(g),
            Layer::Qsa(_) => None,
        }
    }

    pub fn qsa_layer(&self, l: usize) -> Option<&QsaLayer<'static>> {
        match &self.layers[l] {
            Layer::Qsa(q) => Some(q),
            Layer::Gdn(_) => None,
        }
    }

    /// The embedding row of `token`.
    pub fn embed_row(&self, token: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; self.embed.cols];
        strata_gguf::dequant::dequantize_row(self.embed.quant, self.embed.row(token as usize), &mut out)
            .expect("embedding row");
        out
    }

    /// One token through the whole network; returns the full logits. The
    /// session's position advances and the token joins its history.
    pub fn forward(&self, session: &mut Session, token: u32) -> Vec<f32> {
        session.tokens.push(token as i32);
        let logits = match self.family {
            ModelFamily::Qwen35Moe => self.forward_qwen35(session, token),
            ModelFamily::Qwen35 => self.forward_qwen35_dense(session, token),
            _ => self.forward_qwen4exp(session, token),
        };
        session.pos += 1;
        logits
    }

    fn forward_qwen4exp(&self, s: &mut Session, token: u32) -> Vec<f32> {
        let g = &self.geometry;
        let n = g.n_embd as usize;
        let hc = g.hc as usize;
        // Cloned out of the session: the borrow must not outlive the call that
        // takes it, or every mutable borrow of the session below conflicts.
        let be_arc = Arc::clone(&s.backend);
        let be: &dyn crate::backend::ComputeBackend = be_arc.as_ref();
        // The residual stack: the embedding broadcast to every stream.
        let emb = self.embed_row(token);
        for c in 0..hc {
            s.r[c * n..(c + 1) * n].copy_from_slice(&emb);
        }

        for l in 0..g.n_layers as usize {
            // attention half.
            self.hc_attn[l]
                .read(
                    &s.r,
                    &mut s.scratch.mixed,
                    &mut s.scratch.inject,
                    &mut s.scratch.xn,
                    &mut s.scratch.lo,
                    &mut s.scratch.gate_all,
                    &mut s.scratch.deq,
                    be,
                )
                .expect("hc read");
            let mixed = s.scratch.mixed.clone();
            let block_out = match &self.layers[l] {
                Layer::Gdn(gdn) => {
                    let gi = gdn_index(g, l);
                    gdn.forward(&mixed, &mut s.gdn[gi], &mut s.scratch.gdn[gi], be)
                        .expect("gdn forward")
                }
                Layer::Qsa(qsa) => {
                    let qi = qsa_index(g, l);
                    qsa.forward(&mixed, s.pos, &mut s.qsa[qi], &mut s.scratch.qsa[qi], be)
                        .expect("qsa forward")
                }
            };
            self.hc_attn[l].write(&mut s.r, &block_out, &s.scratch.inject);

            // ffn half.
            self.hc_ffn[l]
                .read(
                    &s.r,
                    &mut s.scratch.mixed,
                    &mut s.scratch.inject2,
                    &mut s.scratch.xn,
                    &mut s.scratch.lo,
                    &mut s.scratch.gate_all,
                    &mut s.scratch.deq,
                    be,
                )
                .expect("hc read");
            let mixed = s.scratch.mixed.clone();
            let routing = self.moe[l].route(&mixed, &mut s.scratch.moe_logits, &mut s.scratch.deq, be);
            let block_out = self.moe[l]
                .combine(
                    &mixed,
                    &routing,
                    &mut s.scratch.moe_h,
                    &mut s.scratch.moe_part,
                    &mut s.scratch.moe_out,
                    &mut s.scratch.deq,
                    be,
                )
                .expect("moe combine");
            self.hc_ffn[l].write(&mut s.r, &block_out, &s.scratch.inject2);
        }

        // head: the final hyper-connection read with no injection.
        let hc_out = self.output_hc.as_ref().expect("output hc");
        hc_out
            .read(
                &s.r,
                &mut s.scratch.mixed,
                &mut s.scratch.inject,
                &mut s.scratch.xn,
                &mut s.scratch.lo,
                &mut s.scratch.gate_all,
                &mut s.scratch.deq,
                be,
            )
            .expect("output hc read");
        let mixed = s.scratch.mixed.clone();
        let mut logits = vec![0.0f32; self.output.rows];
        be.gemv(&self.output, &mixed, &mut logits, &mut s.scratch.deq)
            .expect("head gemv");
        logits
    }
}

/// The GDN layer index for absolute layer `l` (only GDN layers are stored).
pub fn gdn_index(g: &ModelGeometry, l: usize) -> usize {
    (0..l).filter(|&i| !g.is_qsa_layer(i as i64)).count()
}

/// The QSA cache index for absolute layer `l`.
pub fn qsa_index(g: &ModelGeometry, l: usize) -> usize {
    (0..l).filter(|&i| g.is_qsa_layer(i as i64)).count()
}

fn load_gdn<'a>(weights: &mut WeightStore<'a>, g: &ModelGeometry, p: &str) -> Result<GdnLayer<'a>> {
    let qkv = weights.tensor_required(&format!("{p}attn_qkv.weight"))?;
    let gate = weights.tensor_required(&format!("{p}attn_gate.weight"))?;
    let out = weights.tensor_required(&format!("{p}ssm_out.weight"))?;
    let conv1d = weights.tensor_required(&format!("{p}ssm_conv1d.weight"))?;
    let alpha = weights.tensor_required(&format!("{p}ssm_alpha.weight"))?;
    let beta = weights.tensor_required(&format!("{p}ssm_beta.weight"))?;
    let ssm_a = weights
        .f32_vec(&format!("{p}ssm_a"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}ssm_a"),
        })?;
    let dt_bias = weights
        .f32_vec(&format!("{p}ssm_dt.bias"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}ssm_dt.bias"),
        })?;
    let ssm_norm = weights
        .f32_vec(&format!("{p}ssm_norm.weight"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}ssm_norm.weight"),
        })?;
    let conv_channels = conv1d.rows;
    let mut conv_flat = vec![0.0f32; conv_channels * g.ssm_d_conv as usize];
    for c in 0..conv_channels {
        strata_gguf::dequant::dequantize_row(
            conv1d.quant,
            conv1d.row(c),
            &mut conv_flat[c * g.ssm_d_conv as usize..(c + 1) * g.ssm_d_conv as usize],
        )
        .map_err(|e| Error::Backend {
            message: format!("{p}ssm_conv1d.weight: {e}"),
        })?;
    }
    Ok(GdnLayer {
        qkv,
        gate,
        out,
        conv_flat,
        alpha,
        beta,
        ssm_a,
        dt_bias,
        ssm_norm,
        n_embd: g.n_embd as usize,
        n_k_heads: g.ssm_k_heads as usize,
        n_v_heads: g.ssm_v_heads as usize,
        state_size: g.ssm_state_size as usize,
        d_conv: g.ssm_d_conv as usize,
        conv_channels,
        value_dim: g.ssm_value_dim as usize,
        eps: g.rms_eps,
        silu_gate: g.family == ModelFamily::Qwen35Moe,
    })
}

fn load_qsa<'a>(weights: &mut WeightStore<'a>, g: &ModelGeometry, p: &str, l: usize) -> Result<QsaLayer<'a>> {
    let q = weights.tensor_required(&format!("{p}attn_q.weight"))?;
    let k = weights.tensor_required(&format!("{p}attn_k.weight"))?;
    let v = weights.tensor_required(&format!("{p}attn_v.weight"))?;
    let o = weights.tensor_required(&format!("{p}attn_output.weight"))?;
    let q_norm = weights
        .f32_vec(&format!("{p}attn_q_norm.weight"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}attn_q_norm.weight"),
        })?;
    let k_norm = weights
        .f32_vec(&format!("{p}attn_k_norm.weight"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}attn_k_norm.weight"),
        })?;
    let dense = !g.has_indexer();
    let (iq_proj, ik_proj, iq_norm, ik_norm) = if dense {
        (None, None, None, None)
    } else {
        (
            Some(weights.tensor_required(&format!("{p}indexer.q_proj.weight"))?),
            Some(weights.tensor_required(&format!("{p}indexer.k_proj.weight"))?),
            Some(
                weights
                    .f32_vec(&format!("{p}indexer.q_norm.weight"))
                    .ok_or_else(|| Error::MissingTensor {
                        name: format!("{p}indexer.q_norm.weight"),
                    })?,
            ),
            Some(
                weights
                    .f32_vec(&format!("{p}indexer.k_norm.weight"))
                    .ok_or_else(|| Error::MissingTensor {
                        name: format!("{p}indexer.k_norm.weight"),
                    })?,
            ),
        )
    };
    let _ = l;
    Ok(QsaLayer {
        q,
        k,
        v,
        o,
        q_norm,
        k_norm,
        iq_proj,
        ik_proj,
        iq_norm,
        ik_norm,
        n_head: g.n_head as usize,
        n_head_kv: g.n_head_kv as usize,
        head_dim: g.head_dim as usize,
        n_rot: g.n_rot as usize,
        freq_base: 1e7,
        eps: g.rms_eps,
        idx_q_heads: g.idx_q_heads as usize,
        idx_key_dim: g.idx_key_dim.max(1) as usize,
        idx_block: g.idx_block.max(1) as usize,
        idx_top_k: g.idx_top_k as usize,
        dense,
    })
}

fn load_hc<'a>(
    weights: &mut WeightStore<'a>,
    g: &ModelGeometry,
    p: &str,
    prefix: &str,
) -> Result<HyperConnection<'a>> {
    let norm = weights
        .f32_vec(&format!("{p}{prefix}norm.weight"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{p}{prefix}norm.weight"),
        })?;
    let down = weights.tensor_required(&format!("{p}{prefix}down.weight"))?;
    let up = weights.tensor_required(&format!("{p}{prefix}up.weight"))?;
    let inject = Some(weights.tensor_required(&format!("{p}{prefix}inject.weight"))?);
    Ok(HyperConnection {
        norm,
        down,
        up,
        inject,
        hc: g.hc as usize,
        n_embd: g.n_embd as usize,
        hc_lr: g.hc_lr as usize,
        eps: g.rms_eps,
    })
}

fn load_hc_global<'a>(
    weights: &mut WeightStore<'a>,
    g: &ModelGeometry,
    prefix: &str,
) -> Result<HyperConnection<'a>> {
    let norm = weights
        .f32_vec(&format!("{prefix}norm.weight"))
        .ok_or_else(|| Error::MissingTensor {
            name: format!("{prefix}norm.weight"),
        })?;
    let down = weights.tensor_required(&format!("{prefix}down.weight"))?;
    let up = weights.tensor_required(&format!("{prefix}up.weight"))?;
    // The final read has no injection: the head never writes back.
    Ok(HyperConnection {
        norm,
        down,
        up,
        inject: None,
        hc: g.hc as usize,
        n_embd: g.n_embd as usize,
        hc_lr: g.hc_lr as usize,
        eps: g.rms_eps,
    })
}
