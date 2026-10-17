//! The mixture-of-experts half: router, routed experts, the always-on shared
//! expert and the combine.
//!
//! Router: softmax over ALL experts, top-k by descending probability (ties by
//! ascending index), the selected weights renormalised once here (never
//! again at combine) with a `2^-14` floor on the sum. The router sees its
//! activation rounded to bf16 — fp16 here is a measured logit error that
//! flips selections.
//!
//! The shared expert is always on with its own scalar sigmoid gate and is
//! added PLAIN (not router-weighted) at the combine.

use crate::backend::QuantTensor;
use crate::error::Result;
use crate::hyper::bf16_round;
use crate::ops::{sigmoid, silu};
use crate::weights::{ExpertView, WeightStore};
use std::sync::Arc;

/// One MoE layer's weights.
#[derive(Debug, Clone)]
pub struct MoeLayer<'a> {
    pub router: QuantTensor<'a>,   // [n_expert, n_embd]
    pub gate_exps: ExpertView<'a>, // [n_expert][n_embd, n_ff]
    pub up_exps: ExpertView<'a>,
    pub down_exps: ExpertView<'a>,   // [n_expert][n_ff, n_embd]
    pub gate_shexp: QuantTensor<'a>, // [n_ff, n_embd]
    pub up_shexp: QuantTensor<'a>,
    pub down_shexp: QuantTensor<'a>,     // [n_embd, n_ff]
    pub gate_inp_shexp: QuantTensor<'a>, // [1, n_embd]

    pub n_expert: usize,
    pub top_k: usize,
    pub n_ff: usize,
}

/// The router's choice for one token.
#[derive(Debug, Clone)]
pub struct Routing {
    pub ids: Vec<usize>,
    /// Renormalised weights, aligned with `ids`.
    pub weights: Vec<f32>,
}

impl MoeLayer<'_> {
    /// Softmax over all experts, top-k, renormalised. `logits` must hold
    /// `n_expert` floats; `deq` is the row-dequant scratch (>= n_embd).
    pub fn route(
        &self,
        x: &[f32],
        logits: &mut [f32],
        deq: &mut [f32],
        be: &dyn crate::backend::ComputeBackend,
    ) -> Routing {
        let x_bf: Vec<f32> = x.iter().map(|&v| bf16_round(v)).collect();
        be.gemv(&self.router, &x_bf, logits, deq).expect("router gemv");
        // softmax over all experts (max-shifted, f64 sum).
        let logits = &mut logits[..self.n_expert];
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for v in logits.iter_mut() {
            *v = (*v - mx).exp();
            sum += f64::from(*v);
        }
        let inv = (1.0 / sum) as f32;
        for v in logits.iter_mut() {
            *v *= inv;
        }
        // stable top-k: descending probability, ties by ascending index.
        let mut idx: Vec<usize> = (0..self.n_expert).collect();
        let k = self.top_k.min(self.n_expert);
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        idx.sort_unstable(); // ascending id order, as the reference returns
        let mut ws = 0.0f64;
        let mut w = Vec::with_capacity(k);
        for &i in &idx {
            let v = logits[i];
            w.push(v);
            ws += f64::from(v);
        }
        let s = ws.max(2f64.powi(-14));
        for v in &mut w {
            *v = (*v as f64 / s) as f32;
        }
        Routing { ids: idx, weights: w }
    }

    /// The routed experts' weighted sum plus the always-on shared expert.
    /// `h`/`part` hold `n_ff` floats; `down_out` holds `n_embd`.
    #[allow(clippy::too_many_arguments)]
    pub fn combine(
        &self,
        x: &[f32],
        routing: &Routing,
        h: &mut [f32],
        part: &mut [f32],
        down_out: &mut [f32],
        scratch: &mut [f32],
        be: &dyn crate::backend::ComputeBackend,
    ) -> Result<Vec<f32>> {
        let n = self.down_exps.cols; // n_embd
        let mut out = vec![0.0f32; n];
        // routed experts.
        for (j, &e) in routing.ids.iter().enumerate() {
            let w = routing.weights[j];
            if w == 0.0 {
                continue;
            }
            be.expert_gemv(&self.gate_exps, e, x, h, scratch)
                .expect("expert gate");
            be.expert_gemv(&self.up_exps, e, x, part, scratch)
                .expect("expert up");
            for (g, u) in h.iter_mut().zip(part.iter()) {
                *g = silu(*g) * *u;
            }
            be.expert_gemv(&self.down_exps, e, h, down_out, scratch)
                .expect("expert down");
            for (o, p) in out.iter_mut().zip(down_out.iter()) {
                *o += w * *p;
            }
        }
        // shared expert: silu(gate)·up @ down, scaled by one sigmoid gate.
        be.gemv(&self.gate_shexp, x, h, scratch).expect("shexp gate");
        be.gemv(&self.up_shexp, x, part, scratch).expect("shexp up");
        for (g, u) in h.iter_mut().zip(part.iter()) {
            *g = silu(*g) * *u;
        }
        be.gemv(&self.down_shexp, h, down_out, scratch)
            .expect("shexp down");
        let mut sgate = vec![0.0f32; 1];
        be.gemv(&self.gate_inp_shexp, x, &mut sgate, scratch)
            .expect("shexp gate input");
        let g = sigmoid(sgate[0]);
        for (o, p) in out.iter_mut().zip(down_out.iter()) {
            *o += g * *p;
        }
        Ok(out)
    }
}

/// Loads one MoE layer's weights for `layer` from the store.
pub fn load_layer<'a>(
    store: &mut WeightStore<'a>,
    layer: usize,
    n_expert: usize,
    top_k: usize,
) -> Result<MoeLayer<'a>> {
    let p = format!("blk.{layer}.");
    let router = store.tensor_required(&format!("{p}ffn_gate_inp.weight"))?;
    let gate_exps = store.expert(&format!("{p}ffn_gate_exps.weight")).ok_or_else(|| {
        crate::error::Error::MissingTensor {
            name: format!("{p}ffn_gate_exps.weight"),
        }
    })?;
    let up_exps = store.expert(&format!("{p}ffn_up_exps.weight")).ok_or_else(|| {
        crate::error::Error::MissingTensor {
            name: format!("{p}ffn_up_exps.weight"),
        }
    })?;
    let down_exps = store.expert(&format!("{p}ffn_down_exps.weight")).ok_or_else(|| {
        crate::error::Error::MissingTensor {
            name: format!("{p}ffn_down_exps.weight"),
        }
    })?;
    let gate_shexp = store.tensor_required(&format!("{p}ffn_gate_shexp.weight"))?;
    let up_shexp = store.tensor_required(&format!("{p}ffn_up_shexp.weight"))?;
    let down_shexp = store.tensor_required(&format!("{p}ffn_down_shexp.weight"))?;
    let gate_inp_shexp = store.tensor_required(&format!("{p}ffn_gate_inp_shexp.weight"))?;
    let n_ff = gate_exps.cols;
    Ok(MoeLayer {
        router,
        gate_exps,
        up_exps,
        down_exps,
        gate_shexp,
        up_shexp,
        down_shexp,
        gate_inp_shexp,
        n_expert,
        top_k,
        n_ff,
    })
}

/// Silu-gated SwiGLU helper shared by the expert paths.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * *u;
    }
}

/// A convenience alias so the sampler can round-trip `Arc` norm vectors.
pub type NormVec = Arc<Vec<f32>>;
