//! The qwen35moe forward pass: plain pre-norm residual, dense attention (no
//! indexer), SiLU-gated GDN, 256-expert top-8 MoE, and the final norm + head.
//!
//! ```text
//! x = token_embd[tok]
//! for l in 0..n_layers:
//!    h = rms(x) · attn_norm
//!    x += layer(h)                      # GDN (SiLU gate) or dense attention
//!    h2 = rms(x) · post_attention_norm
//!    x += moe(h2)                       # top-8 + shared
//! logits = output.weight @ (rms(x) · output_norm)
//! ```

use crate::engine::{gdn_index, qsa_index, Engine, Layer};
use crate::ops::rms_norm_into;
use crate::session::Session;
use std::sync::Arc;

impl Engine {
    /// One token through the qwen35moe network; returns the full logits.
    pub fn forward_qwen35(&self, s: &mut Session, token: u32) -> Vec<f32> {
        let g = &self.geometry;
        let be_arc = Arc::clone(&s.backend);
        let be: &dyn crate::backend::ComputeBackend = be_arc.as_ref();

        let emb = self.embed_row(token);
        s.x.copy_from_slice(&emb);

        for l in 0..g.n_layers as usize {
            let (attn_norm, post_norm) = &self.qwen35_norms[l];
            // attention half.
            rms_norm_into(&s.x, attn_norm, g.rms_eps, &mut s.scratch.mixed);
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
            for (x, b) in s.x.iter_mut().zip(&block_out) {
                *x += *b;
            }

            // ffn half.
            rms_norm_into(&s.x, post_norm, g.rms_eps, &mut s.scratch.mixed);
            let mixed = s.scratch.mixed.clone();
            let routing = self.moe[l].route(&mixed, &mut s.scratch.moe_logits, &mut s.scratch.deq, be);
            let moe_out = self.moe[l]
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
            for (x, m) in s.x.iter_mut().zip(&moe_out) {
                *x += *m;
            }
        }

        // head: rms(x) · output_norm, then the output projection.
        let out_norm = self.output_norm.as_ref().expect("output_norm");
        rms_norm_into(&s.x, out_norm, g.rms_eps, &mut s.scratch.mixed);
        let mixed = s.scratch.mixed.clone();
        let mut logits = vec![0.0f32; self.output.rows];
        be.gemv(&self.output, &mixed, &mut logits, &mut s.scratch.deq)
            .expect("head gemv");
        logits
    }

    /// One token through the dense `qwen35` network: the same GDN / QSA layer
    /// split as `forward_qwen35`, but the FFN half is a dense SwiGLU MLP
    /// (`ffn_gate` → SiLU → `ffn_up`, then `ffn_down`) instead of the MoE.
    pub fn forward_qwen35_dense(&self, s: &mut Session, token: u32) -> Vec<f32> {
        let g = &self.geometry;
        let be_arc = Arc::clone(&s.backend);
        let be: &dyn crate::backend::ComputeBackend = be_arc.as_ref();

        let emb = self.embed_row(token);
        s.x.copy_from_slice(&emb);

        for l in 0..g.n_layers as usize {
            let (attn_norm, post_norm) = &self.qwen35_norms[l];
            // attention half.
            rms_norm_into(&s.x, attn_norm, g.rms_eps, &mut s.scratch.mixed);
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
            for (x, b) in s.x.iter_mut().zip(&block_out) {
                *x += *b;
            }

            // dense FFN half: SwiGLU(gate, up) → down.
            rms_norm_into(&s.x, post_norm, g.rms_eps, &mut s.scratch.mixed);
            let mixed = s.scratch.mixed.clone();
            let ffn = &self.ffn[l];
            be.gemv(&ffn.gate, &mixed, &mut s.scratch.moe_h, &mut s.scratch.deq)
                .expect("ffn gate");
            be.gemv(&ffn.up, &mixed, &mut s.scratch.moe_part, &mut s.scratch.deq)
                .expect("ffn up");
            let n_ff = s.scratch.moe_h.len();
            for i in 0..n_ff {
                let gv = s.scratch.moe_h[i];
                s.scratch.moe_h[i] = gv / (1.0 + (-gv).exp()) * s.scratch.moe_part[i];
            }
            be.gemv(&ffn.down, &s.scratch.moe_h, &mut s.scratch.moe_out, &mut s.scratch.deq)
                .expect("ffn down");
            for (x, m) in s.x.iter_mut().zip(&s.scratch.moe_out) {
                *x += *m;
            }
        }

        let out_norm = self.output_norm.as_ref().expect("output_norm");
        rms_norm_into(&s.x, out_norm, g.rms_eps, &mut s.scratch.mixed);
        let mixed = s.scratch.mixed.clone();
        let mut logits = vec![0.0f32; self.output.rows];
        be.gemv(&self.output, &mixed, &mut logits, &mut s.scratch.deq)
            .expect("head gemv");
        logits
    }
}
