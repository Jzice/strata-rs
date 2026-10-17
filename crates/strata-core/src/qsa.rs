//! The full-attention (QSA) layer: qwen4exp's 12 sparse-attention layers with
//! the indexer, and qwen35moe's 10 dense-attention layers (`dense = true`).
//!
//! Per token: the projections, RMS norm + NEOX partial RoPE on q/k (and the
//! indexer query), the raw indexer key appended and pooled per completed
//! block, the per-cell ReLU-head scores, the top-`width` cell selection
//! (the incomplete tail is force-selected, so attention stays causal),
//! softmax GQA over the gathered cells, the sigmoid gate from the query's
//! second half, and the output projection.

#![allow(clippy::needless_range_loop)]

use crate::backend::{QuantTensor, RopeCfg};
use crate::error::Result;
use crate::ops::{dot, rms_norm, rope_neox, sigmoid};
use std::sync::Arc;

/// One attention layer's weights.
#[derive(Debug, Clone)]
pub struct QsaLayer<'a> {
    pub q: QuantTensor<'a>,    // [n_head·2·head_dim, n_embd]
    pub k: QuantTensor<'a>,    // [n_head_kv·head_dim, n_embd]
    pub v: QuantTensor<'a>,    // [n_head_kv·head_dim, n_embd]
    pub o: QuantTensor<'a>,    // [n_embd, n_head·head_dim]
    pub q_norm: Arc<Vec<f32>>, // [head_dim]
    pub k_norm: Arc<Vec<f32>>,
    /// The indexer projections; `None` for dense attention (qwen35moe).
    pub iq_proj: Option<QuantTensor<'a>>, // [idx_q_heads·idx_key_dim, n_embd]
    pub ik_proj: Option<QuantTensor<'a>>, // [idx_key_dim, n_embd]
    pub iq_norm: Option<Arc<Vec<f32>>>,
    pub ik_norm: Option<Arc<Vec<f32>>>,

    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub freq_base: f64,
    pub eps: f64,
    pub idx_q_heads: usize,
    pub idx_key_dim: usize,
    pub idx_block: usize,
    pub idx_top_k: usize,
    /// Dense attention (no indexer): qwen35moe.
    pub dense: bool,
}

/// Per-layer attention state for one sequence.
#[derive(Debug, Clone)]
pub struct QsaCache {
    /// `[kv_head][pos·head_dim]` f32 keys and values.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub n_kv: usize,
    /// The raw indexer tail: `(idx_block - 1)·idx_dim`, slot = pos % idx_block.
    pub idx_tail: Vec<f32>,
    /// Pooled block keys plus the spare slot at row `n_bid`.
    pub idx_pooled: Vec<f32>,
    pub n_bid: usize,
    pub spare_ready: bool,
}

impl QsaCache {
    pub fn new(layer: &QsaLayer<'_>) -> Self {
        Self {
            k: vec![Vec::new(); layer.n_head_kv],
            v: vec![Vec::new(); layer.n_head_kv],
            n_kv: 0,
            idx_tail: vec![0.0; (layer.idx_block.max(1) - 1) * layer.idx_key_dim.max(1)],
            idx_pooled: Vec::new(),
            n_bid: 0,
            spare_ready: false,
        }
    }

    pub fn reset(&mut self) {
        for k in &mut self.k {
            k.clear();
        }
        for v in &mut self.v {
            v.clear();
        }
        self.n_kv = 0;
        self.n_bid = 0;
        self.spare_ready = false;
    }
}

/// Per-token scratch.
#[derive(Debug, Default)]
pub struct QsaScratch {
    pub q_full: Vec<f32>, // n_head·2·head_dim
    pub kcur: Vec<f32>,   // n_head_kv·head_dim
    pub vcur: Vec<f32>,
    pub q_idx: Vec<f32>,   // idx_q_heads·idx_key_dim
    pub idx_raw: Vec<f32>, // idx_key_dim
    pub attn: Vec<f32>,    // n_head·head_dim
    pub scores: Vec<f64>,
    pub row: Vec<f32>,
}

impl QsaScratch {
    pub fn new(layer: &QsaLayer<'_>) -> Self {
        // The dequant scratch must cover the widest row any projection here
        // dequantizes: the input width (attn_q's cols) or the attention width.
        let widest = layer
            .q
            .cols
            .max(layer.n_head * layer.head_dim)
            .max(layer.idx_q_heads * layer.idx_key_dim)
            .max(layer.idx_key_dim);
        Self {
            q_full: vec![0.0; layer.n_head * 2 * layer.head_dim],
            kcur: vec![0.0; layer.n_head_kv * layer.head_dim],
            vcur: vec![0.0; layer.n_head_kv * layer.head_dim],
            q_idx: vec![0.0; layer.idx_q_heads * layer.idx_key_dim],
            idx_raw: vec![0.0; layer.idx_key_dim],
            attn: vec![0.0; layer.n_head * layer.head_dim],
            scores: Vec::new(),
            row: vec![0.0; widest],
        }
    }
}

impl QsaLayer<'_> {
    /// One token at `pos` through the layer; returns the block output.
    /// `be` runs the projections; the cache, indexer and softmax are host
    /// work (they are per-token control flow, not bulk matmul).
    pub fn forward(
        &self,
        x: &[f32],
        pos: usize,
        cache: &mut QsaCache,
        sc: &mut QsaScratch,
        be: &dyn crate::backend::ComputeBackend,
    ) -> Result<Vec<f32>> {
        let hd = self.head_dim;
        let nh = self.n_head;
        let nhkv = self.n_head_kv;

        // 1. projections.
        be.gemv(&self.q, x, &mut sc.q_full, &mut sc.row).expect("q");
        be.gemv(&self.k, x, &mut sc.kcur, &mut sc.row).expect("k");
        be.gemv(&self.v, x, &mut sc.vcur, &mut sc.row).expect("v");

        // 2. norm + NEOX partial RoPE on q (first half of each head) and k.
        let rope_cfg = RopeCfg {
            pos,
            n_rot: self.n_rot,
            head_dim: hd,
            freq_base: self.freq_base,
        };
        for h in 0..nh {
            let head = &mut sc.q_full[h * 2 * hd..h * 2 * hd + hd];
            rms_norm(head, &self.q_norm, self.eps);
            rope_neox(head, &rope_cfg);
        }
        for h in 0..nhkv {
            let head = &mut sc.kcur[h * hd..(h + 1) * hd];
            rms_norm(head, &self.k_norm, self.eps);
            rope_neox(head, &rope_cfg);
        }

        // 3. append K and V (raw, after norm+rope) to the cache.
        for h in 0..nhkv {
            cache.k[h].extend_from_slice(&sc.kcur[h * hd..(h + 1) * hd]);
            cache.v[h].extend_from_slice(&sc.vcur[h * hd..(h + 1) * hd]);
        }
        cache.n_kv += 1;
        let n_kv = cache.n_kv;

        // 4. indexer: append the raw key, pool completed blocks, score, select.
        let selected: Vec<usize> = if self.dense {
            (0..n_kv).collect()
        } else {
            self.indexer_step(x, pos, cache, sc, be);
            self.indexer_select(cache, sc, n_kv)
        };
        // 5. softmax GQA over the gathered cells.
        let group = nh / nhkv;
        let scale = 1.0 / (hd as f64).sqrt();
        for h in 0..nh {
            let kvh = h / group;
            let q = &sc.q_full[h * 2 * hd..h * 2 * hd + hd];
            sc.scores.clear();
            for &j in &selected {
                let k = &cache.k[kvh][j * hd..(j + 1) * hd];
                sc.scores.push(f64::from(dot(k, q)) * scale);
            }
            // softmax over the gathered cells (f64 accumulation).
            let mx = sc.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for s in &mut sc.scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for s in &mut sc.scores {
                *s *= inv;
            }
            let out = &mut sc.attn[h * hd..(h + 1) * hd];
            out.fill(0.0);
            for (si, &j) in selected.iter().enumerate() {
                let v = &cache.v[kvh][j * hd..(j + 1) * hd];
                let w = sc.scores[si] as f32;
                for (o, vv) in out.iter_mut().zip(v) {
                    *o += w * *vv;
                }
            }
        }

        // 6. gate with the query's second half, then the output projection.
        for h in 0..nh {
            for d in 0..hd {
                let g = sigmoid(sc.q_full[h * 2 * hd + hd + d]);
                sc.attn[h * hd + d] *= g;
            }
        }
        let mut out = vec![0.0f32; self.o.rows];
        be.gemv(&self.o, &sc.attn, &mut out, &mut sc.row)
            .expect("attn output");
        Ok(out)
    }

    /// Appends the raw indexer key and pools every completed block.
    ///
    /// The tail holds the block's first `r - 1` cells (slot = pos % r); the
    /// completing cell's raw key arrives as `x`'s projection and joins the
    /// mean directly. `pooled[b]` is block `b`'s key, rotated at the block's
    /// FIRST cell; `pooled[n_bid]` is the spare slot, the constant
    /// `rope(rms_norm(raw[0]), 0)`.
    fn indexer_step(
        &self,
        x: &[f32],
        pos: usize,
        cache: &mut QsaCache,
        sc: &mut QsaScratch,
        be: &dyn crate::backend::ComputeBackend,
    ) {
        let dim = self.idx_key_dim;
        let r = self.idx_block;
        let ik_proj = self.ik_proj.expect("dense layers never call this");
        let ik_norm = self.ik_norm.as_ref().expect("indexer norm");
        let iq_proj = self.iq_proj.expect("dense layers never call this");
        let iq_norm = self.iq_norm.as_ref().expect("indexer norm");

        // The raw indexer key for this cell.
        be.gemv(&ik_proj, x, &mut sc.idx_raw, &mut sc.row)
            .expect("indexer k");

        // The spare slot's constant value, captured at cell 0.
        if !cache.spare_ready {
            let mut key = sc.idx_raw.clone();
            rms_norm(&mut key, ik_norm, self.eps);
            let cfg = RopeCfg {
                pos: 0,
                n_rot: self.n_rot,
                head_dim: dim,
                freq_base: self.freq_base,
            };
            rope_neox(&mut key, &cfg);
            cache.idx_pooled = key; // row 0 is the spare slot until a block lands
            cache.spare_ready = true;
        }

        let slot = (cache.n_kv - 1) % r;
        let completes = cache.n_kv % r == 0;

        // The tail holds the block's first r-1 cells.
        if slot < r - 1 {
            cache.idx_tail[slot * dim..(slot + 1) * dim].copy_from_slice(&sc.idx_raw);
        }

        if completes {
            let b = cache.n_kv / r - 1; // the completed block
            let mut mean = vec![0.0f32; dim];
            for cell in 0..r - 1 {
                for d in 0..dim {
                    mean[d] += cache.idx_tail[cell * dim + d];
                }
            }
            for d in 0..dim {
                mean[d] = (mean[d] + sc.idx_raw[d]) / r as f32;
            }
            rms_norm(&mut mean, ik_norm, self.eps);
            let cfg = RopeCfg {
                pos: pos - (r - 1), // the block's first-cell position
                n_rot: self.n_rot,
                head_dim: dim,
                freq_base: self.freq_base,
            };
            rope_neox(&mut mean, &cfg);
            // The spare's constant value must survive the row-0 overwrite.
            let spare = cache.idx_pooled[..dim].to_vec();
            if cache.idx_pooled.len() < (b + 2) * dim {
                cache.idx_pooled.resize((b + 2) * dim, 0.0);
            }
            cache.idx_pooled[b * dim..(b + 1) * dim].copy_from_slice(&mean);
            cache.idx_pooled[(b + 1) * dim..(b + 2) * dim].copy_from_slice(&spare);
            cache.n_bid = b + 1;
        }

        // The indexer query: projection, norm, rotate at this position.
        be.gemv(&iq_proj, x, &mut sc.q_idx, &mut sc.row)
            .expect("indexer q");
        for h in 0..self.idx_q_heads {
            let qh = &mut sc.q_idx[h * dim..(h + 1) * dim];
            rms_norm(qh, iq_norm, self.eps);
            let cfg = RopeCfg {
                pos,
                n_rot: self.n_rot,
                head_dim: dim,
                freq_base: self.freq_base,
            };
            rope_neox(qh, &cfg);
        }
    }

    /// Scores every cell and selects the `width` largest (ties by ascending
    /// index), returned ascending. The incomplete tail is force-selected.
    fn indexer_select(&self, cache: &QsaCache, sc: &mut QsaScratch, n_kv: usize) -> Vec<usize> {
        let dim = self.idx_key_dim;
        let r = self.idx_block;
        let width = (self.idx_top_k + r - 1).min(n_kv);
        let n_bid = cache.n_bid;
        let complete = (n_kv / r) * r;

        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n_kv);
        for j in 0..n_kv {
            let cb = (j / r).min(n_bid);
            let pooled = &cache.idx_pooled[cb * dim..(cb + 1) * dim];
            let mut s = 0.0f32;
            for h in 0..self.idx_q_heads {
                let qh = &sc.q_idx[h * dim..(h + 1) * dim];
                let d = dot(pooled, qh).max(0.0);
                s += d;
            }
            if j >= complete {
                s += 1e9; // force-select the incomplete tail (causality)
            }
            scores.push((j, s));
        }
        // The width largest; ties by ascending index; returned ascending.
        scores.select_nth_unstable_by(width - 1, |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scores.truncate(width);
        scores.sort_unstable_by_key(|(j, _)| *j);
        scores.into_iter().map(|(j, _)| j).collect()
    }
}
