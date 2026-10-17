//! The hyper-connection: qwen4exp's gated residual stack.
//!
//! The residual is a stack of `hc` streams, each `n_embd` wide (stream `c` at
//! row `c·n_embd`) — not one vector with a gate. `read` mixes the stack into
//! the vector a mixer consumes and produces the injection the matching
//! `write` gates with; `write` adds the block output to every stream, scaled
//! per stream by `2·sigmoid(inject/hc)` (centred on 1: `inject = 0` is a
//! plain residual add).
//!
//! The `/hc` inside the SiLU bottleneck and the bf16 rounding of the
//! activations before the dots are both part of the contract.

use crate::backend::QuantTensor;
use crate::error::Result;
use crate::ops::{rms_norm_into, sigmoid, silu};
use half::bf16;
use std::sync::Arc;

/// Rounds an activation to bf16 precision, as the reference contract does
/// before the hyper-connection dots.
#[inline]
pub fn bf16_round(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// One half's hyper-connection weights (`hc_attn_*` or `hc_ffn_*`).
#[derive(Debug, Clone)]
pub struct HyperConnection<'a> {
    pub norm: Arc<Vec<f32>>,   // [hc · n_embd], F32
    pub down: QuantTensor<'a>, // [hc_lr, hc · n_embd]
    pub up: QuantTensor<'a>,   // [hc · n_embd, hc_lr]
    /// `None` for the final read (the head has no write-back).
    pub inject: Option<QuantTensor<'a>>, // [hc, hc · n_embd]
    pub hc: usize,
    pub n_embd: usize,
    pub hc_lr: usize,
    pub eps: f64,
}

impl HyperConnection<'_> {
    /// Reads the stack into `mixed` (the mixer's input) and `inject` (the
    /// write gate's pre-activation). `gate_all` must hold `hc·n_embd`.
    #[allow(clippy::too_many_arguments)]
    pub fn read(
        &self,
        r: &[f32],
        mixed: &mut [f32],
        inject: &mut [f32],
        xn: &mut [f32],
        lo: &mut [f32],
        gate_all: &mut [f32],
        scratch: &mut [f32],
        be: &dyn crate::backend::ComputeBackend,
    ) -> Result<()> {
        let hc = self.hc;
        let n = self.n_embd;
        debug_assert_eq!(r.len(), hc * n);
        // (i) per-stream RMS norm with the stack's own norm weights.
        for c in 0..hc {
            let stream = &r[c * n..(c + 1) * n];
            let w = &self.norm[c * n..(c + 1) * n];
            rms_norm_into(stream, w, self.eps, &mut xn[c * n..(c + 1) * n]);
        }
        // (ii) SiLU bottleneck; the /hc is inside the SiLU.
        let xn_bf: Vec<f32> = xn.iter().map(|&v| bf16_round(v)).collect();
        let mut lo_raw = vec![0.0f32; self.hc_lr];
        be.gemv(&self.down, &xn_bf, &mut lo_raw, scratch)
            .expect("hc down");
        for (l, v) in lo.iter_mut().zip(&lo_raw) {
            *l = silu(*v / hc as f32);
        }
        let lo_bf: Vec<f32> = lo.iter().map(|&v| bf16_round(v)).collect();
        // (iii) injection (the final read has none).
        if let Some(w) = &self.inject {
            be.gemv(w, &xn_bf, inject, scratch).expect("hc inject");
        } else {
            inject.fill(0.0);
        }
        // (iv) gated mean over the streams: one gemv for every (c, d) gate.
        be.gemv(&self.up, &lo_bf, gate_all, scratch).expect("hc up");
        for d in 0..n {
            let mut acc = 0.0f64;
            for c in 0..hc {
                let g = sigmoid(gate_all[c * n + d]);
                acc += f64::from(xn[c * n + d]) * f64::from(g);
            }
            mixed[d] = (acc / hc as f64) as f32;
        }
        Ok(())
    }

    /// Writes the block output back into the stack, gated per stream.
    pub fn write(&self, r: &mut [f32], block_out: &[f32], inject: &[f32]) {
        let hc = self.hc;
        let n = self.n_embd;
        for c in 0..hc {
            let w = 2.0 * sigmoid(inject[c] / hc as f32);
            let stream = &mut r[c * n..(c + 1) * n];
            for (s, b) in stream.iter_mut().zip(block_out) {
                *s += *b * w;
            }
        }
    }
}
