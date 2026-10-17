//! The Gated-DeltaNet layer (qwen4exp's 36 linear-attention layers and
//! qwen35moe's 30).
//!
//! Order per token: qkv projection, causal depthwise conv + SiLU over the
//! whole channel block, L2 norm of q and k (v is not normed), the alpha/beta
//! gates, the delta-rule recurrence (decay BEFORE the rank-1 update, readout
//! against the UPDATED state), the z gate and per-head output RMS norm, and
//! the output projection.
//!
//! The loops index heads and channels by hand, the way the reference's own
//! loops do: a transcription that reads the same is easier to check against
//! it.
#![allow(clippy::needless_range_loop)]

use crate::backend::{GdnState, QuantTensor};
use crate::error::Result;
use crate::ops::{l2_norm, rms_norm_into, sigmoid, silu, softplus};
use std::sync::Arc;

/// One GDN layer's weights.
#[derive(Debug, Clone)]
pub struct GdnLayer<'a> {
    pub qkv: QuantTensor<'a>,  // [2·S·h_k + value_dim, n_embd]
    pub gate: QuantTensor<'a>, // [value_dim, n_embd]
    pub out: QuantTensor<'a>,  // [n_embd, value_dim]
    /// The conv taps dequantized once: `conv_flat[c·d_conv + i]` is tap `i`
    /// of channel `c`.
    pub conv_flat: Vec<f32>,
    pub alpha: QuantTensor<'a>,  // [h_v, n_embd]
    pub beta: QuantTensor<'a>,   // [h_v, n_embd]
    pub ssm_a: Arc<Vec<f32>>,    // [h_v]
    pub dt_bias: Arc<Vec<f32>>,  // [h_v]
    pub ssm_norm: Arc<Vec<f32>>, // [S]

    pub n_embd: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub state_size: usize,
    pub d_conv: usize,
    pub conv_channels: usize,
    pub value_dim: usize,
    pub eps: f64,
    /// qwen4exp gates the output with sigmoid; qwen35moe with SiLU.
    pub silu_gate: bool,
}

/// Per-token scratch for one GDN layer.
#[derive(Debug, Default)]
pub struct GdnScratch {
    pub qkv: Vec<f32>,      // conv_channels
    pub conv_out: Vec<f32>, // conv_channels
    pub alpha: Vec<f32>,    // h_v
    pub beta: Vec<f32>,     // h_v
    pub dec: Vec<f32>,      // h_v
    pub o: Vec<f32>,        // value_dim
    pub y: Vec<f32>,        // value_dim
    pub z: Vec<f32>,        // value_dim
    pub d: Vec<f32>,        // S per head, reused
    pub sk: Vec<f32>,       // S per head, reused
    pub row: Vec<f32>,      // dequant scratch (max row width)
}

impl GdnScratch {
    pub fn new(layer: &GdnLayer<'_>) -> Self {
        Self {
            qkv: vec![0.0; layer.conv_channels],
            conv_out: vec![0.0; layer.conv_channels],
            alpha: vec![0.0; layer.n_v_heads],
            beta: vec![0.0; layer.n_v_heads],
            dec: vec![0.0; layer.n_v_heads],
            o: vec![0.0; layer.value_dim],
            y: vec![0.0; layer.value_dim],
            z: vec![0.0; layer.value_dim],
            d: vec![0.0; layer.state_size],
            sk: vec![0.0; layer.state_size],
            row: vec![0.0; layer.n_embd.max(layer.conv_channels).max(layer.value_dim)],
        }
    }
}

impl GdnLayer<'_> {
    /// One token through the layer; returns the block output (`n_embd`).
    /// `be` runs the projections; the recurrence itself is a host loop.
    pub fn forward(
        &self,
        x: &[f32],
        state: &mut GdnState,
        sc: &mut GdnScratch,
        be: &dyn crate::backend::ComputeBackend,
    ) -> Result<Vec<f32>> {
        let s = self.state_size;
        let hk = self.n_k_heads;
        let hv = self.n_v_heads;
        let qk = s * hk; // 2048

        // 1. qkv projection.
        be.gemv(&self.qkv, x, &mut sc.qkv, &mut sc.row).expect("qkv gemv");

        // 2. causal depthwise conv + SiLU over the whole channel block.
        //    inp = [conv_state (oldest first) | x]; the state slides.
        for c in 0..self.conv_channels {
            let w = &self.conv_flat[c * self.d_conv..(c + 1) * self.d_conv];
            let mut acc = 0.0f64;
            for i in 0..self.d_conv {
                let inp = if i + 1 < self.d_conv {
                    state.conv[i * self.conv_channels + c]
                } else {
                    sc.qkv[c]
                };
                acc += f64::from(inp) * f64::from(w[i]);
            }
            sc.conv_out[c] = silu(acc as f32);
        }
        // slide the conv state: drop the oldest, append the new input.
        if self.d_conv > 1 {
            let ch = self.conv_channels;
            let keep = self.d_conv - 1;
            state.conv.copy_within(ch..(keep - 1) * ch + ch, 0);
            for c in 0..ch {
                state.conv[(keep - 1) * ch + c] = sc.qkv[c];
            }
        }

        // 3. L2 norm q and k per head; q carries 1/sqrt(S).
        let q_scale = 1.0 / (s as f64).sqrt();
        for h in 0..hk {
            let head = &mut sc.conv_out[h * s..(h + 1) * s];
            l2_norm(head, self.eps);
        }
        for h in 0..hk {
            let head = &mut sc.conv_out[qk + h * s..qk + (h + 1) * s];
            l2_norm(head, self.eps);
        }
        for h in 0..hk {
            for i in 0..s {
                sc.conv_out[h * s + i] = (f64::from(sc.conv_out[h * s + i]) * q_scale) as f32;
            }
        }

        // 4. gates.
        be.gemv(&self.alpha, x, &mut sc.alpha, &mut sc.row)
            .expect("alpha gemv");
        be.gemv(&self.beta, x, &mut sc.beta, &mut sc.row)
            .expect("beta gemv");
        for h in 0..hv {
            let a = sc.alpha[h];
            let gate = softplus(a + self.dt_bias[h]) * self.ssm_a[h];
            sc.dec[h] = gate.exp();
        }

        // 5. delta-rule recurrence. State layout (i, h, j), j fastest.
        //    qkv layout: q = [0, qk), k = [qk, 2·qk), v = [2·qk, ...).
        let v_base = 2 * qk;
        for h in 0..hv {
            let g = h % hk;
            let dec = sc.dec[h];
            let beta = sigmoid(sc.beta[h]);
            let base = h * s;
            let k = &sc.conv_out[qk + g * s..qk + (g + 1) * s];
            let v = &sc.conv_out[v_base + h * s..v_base + (h + 1) * s];
            // pass 1: decay the state and form sk = (S·dec)·k.
            for j in 0..s {
                let mut acc = 0.0f64;
                for i in 0..s {
                    let idx = i * (hv * s) + base + j;
                    let decayed = state.state[idx] * dec;
                    state.state[idx] = decayed;
                    acc += f64::from(decayed) * f64::from(k[i]);
                }
                sc.sk[j] = acc as f32;
            }
            // d = (v - sk) · beta
            for j in 0..s {
                sc.d[j] = (v[j] - sc.sk[j]) * beta;
            }
            // pass 2: rank-1 update, then readout against the updated state.
            for i in 0..s {
                let ki = k[i];
                for j in 0..s {
                    let idx = i * (hv * s) + base + j;
                    state.state[idx] += ki * sc.d[j];
                }
            }
            for j in 0..s {
                let mut acc = 0.0f64;
                for i in 0..s {
                    acc +=
                        f64::from(state.state[i * (hv * s) + base + j]) * f64::from(sc.conv_out[g * s + i]);
                }
                sc.o[h * s + j] = acc as f32;
            }
        }

        // 6. z projection, per-head output RMS norm, sigmoid/SiLU gate.
        be.gemv(&self.gate, x, &mut sc.z, &mut sc.row).expect("gate gemv");
        for h in 0..hv {
            let o = &mut sc.o[h * s..(h + 1) * s];
            let mut normed = vec![0.0f32; s];
            rms_norm_into(o, &self.ssm_norm, self.eps, &mut normed);
            for i in 0..s {
                let g = if self.silu_gate {
                    silu(sc.z[h * s + i])
                } else {
                    sigmoid(sc.z[h * s + i])
                };
                sc.y[h * s + i] = normed[i] * g;
            }
        }

        // 7. output projection.
        let mut out = vec![0.0f32; self.n_embd];
        be.gemv(&self.out, &sc.y, &mut out, &mut sc.row)
            .expect("out gemv");
        Ok(out)
    }
}
