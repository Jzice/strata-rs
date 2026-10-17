//! Scalar compute primitives.
//!
//! Everything the layer forward pass needs, in plain f32 Rust: quantized
//! matrix-vector products, RMS norm, the NEOX rotary embedding, softmax and
//! the activation functions. The GPU backend (a later phase) accelerates the
//! matmul path; the semantics here are the reference.
#![allow(clippy::needless_range_loop)]

use crate::backend::{QuantTensor, RopeCfg};
use strata_gguf::dequant::dequantize_row;

/// `y = W x` for a quantized matrix: every row is dequantized into `scratch`
/// and dotted with `x`. `scratch` must hold at least `w.cols` floats.
pub fn gemv(w: &QuantTensor<'_>, x: &[f32], y: &mut [f32], scratch: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "gemv {}: x", w.name);
    assert_eq!(y.len(), w.rows, "gemv {}: y", w.name);
    assert!(scratch.len() >= w.cols, "gemv {}: scratch", w.name);
    let cols = w.cols;
    for (r, out) in y.iter_mut().enumerate() {
        let row = &mut scratch[..cols];
        dequantize_row(w.quant, w.row(r), row).expect("row dequant");
        let mut acc = 0.0f64;
        for (a, b) in row.iter().zip(x) {
            acc += f64::from(*a) * f64::from(*b);
        }
        *out = acc as f32;
    }
}

/// `y = W x` where `W` is a plain f32 row-major matrix (`rows × cols`).
pub fn gemv_f32(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for (a, b) in row.iter().zip(x) {
            acc += f64::from(*a) * f64::from(*b);
        }
        *out = acc as f32;
    }
}

/// RMS norm in place: `x = x * w * rsqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f64) {
    let mut ss = 0.0f64;
    for &v in x.iter() {
        ss += f64::from(v) * f64::from(v);
    }
    ss /= x.len() as f64;
    let scale = 1.0 / (ss + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v as f64 * scale * f64::from(w[i])) as f32;
    }
}

/// RMS norm into `out` (`out = x * w * rsqrt(...)`), leaving `x` untouched.
pub fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, out: &mut [f32]) {
    let mut ss = 0.0f64;
    for &v in x {
        ss += f64::from(v) * f64::from(v);
    }
    ss /= x.len() as f64;
    let scale = 1.0 / (ss + eps).sqrt();
    for (i, o) in out.iter_mut().enumerate() {
        *o = (f64::from(x[i]) * scale * f64::from(w[i])) as f32;
    }
}

/// L2 norm of one head, with an `eps` floor on the squared norm; the sum is
/// formed in f64 (the reference's rounding order).
pub fn l2_norm(head: &mut [f32], eps: f64) {
    let mut ss = 0.0f64;
    for &v in head.iter() {
        ss += f64::from(v) * f64::from(v);
    }
    let inv = 1.0 / (ss + eps).sqrt();
    for v in head.iter_mut() {
        *v = (*v as f64 * inv) as f32;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus with the large-argument shortcut the reference uses.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Softmax in place over the whole slice, max-shifted, summed in f64.
pub fn softmax(xs: &mut [f32]) {
    let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in xs.iter_mut() {
        *v = (*v - mx).exp();
        sum += f64::from(*v);
    }
    let inv = 1.0 / sum;
    for v in xs.iter_mut() {
        *v = (*v as f64 * inv) as f32;
    }
}

/// In-place NEOX-style partial rotary embedding on a `[head_dim]` slice:
/// pairs `(i, i + n_rot/2)`, dims `[n_rot, head_dim)` pass through.
pub fn rope_neox(head: &mut [f32], cfg: &RopeCfg) {
    // The rotated slice never exceeds the head (the reference's shapes keep
    // it inside; a smaller geometry must not index out of bounds).
    let n_rot = cfg.n_rot.min(cfg.head_dim).min(head.len()) & !1;
    let pos = cfg.pos as f64;
    let half = n_rot / 2;
    for i in 0..half {
        let theta = cfg.freq_base.powf(-2.0 * i as f64 / n_rot as f64);
        let angle = theta * pos;
        let (s, c) = angle.sin_cos();
        let a = head[i];
        let b = head[i + half];
        head[i] = (f64::from(a) * c - f64::from(b) * s) as f32;
        head[i + half] = (f64::from(a) * s + f64::from(b) * c) as f32;
    }
}

/// The RoPE angles for one position: `(cos, sin)` per rotated pair.
pub fn rope_table(cfg: &RopeCfg) -> Vec<(f32, f32)> {
    let half = cfg.n_rot / 2;
    let pos = cfg.pos as f64;
    (0..half)
        .map(|i| {
            let theta = cfg.freq_base.powf(-2.0 * i as f64 / cfg.n_rot as f64);
            let angle = theta * pos;
            let (s, c) = angle.sin_cos();
            (c as f32, s as f32)
        })
        .collect()
}

/// Dot product accumulated in f64.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        acc += f64::from(*x) * f64::from(*y);
    }
    acc as f32
}

/// `out += a` elementwise.
pub fn add_into(out: &mut [f32], a: &[f32]) {
    for (o, a) in out.iter_mut().zip(a) {
        *o += *a;
    }
}

/// `y = W_e x` for one expert of a 3-D expert tensor stored row-major as
/// `[n_expert][rows = in][cols = out]`; accumulated in f64.
pub fn expert_gemv(
    w: &crate::weights::ExpertView<'_>,
    e: usize,
    x: &[f32],
    y: &mut [f32],
    scratch: &mut [f32],
) {
    debug_assert_eq!(x.len(), w.rows);
    debug_assert_eq!(y.len(), w.cols);
    debug_assert!(scratch.len() >= w.cols);
    let cols = w.cols;
    let mut acc = vec![0.0f64; cols];
    let row = &mut scratch[..cols];
    for i in 0..w.rows {
        let xi = f64::from(x[i]);
        if xi == 0.0 {
            continue;
        }
        dequantize_row(w.quant, w.expert_row(e, i), row).expect("expert row");
        for (a, wv) in acc.iter_mut().zip(row.iter()) {
            *a += xi * f64::from(*wv);
        }
    }
    for (o, a) in y.iter_mut().zip(&acc) {
        *o = *a as f32;
    }
}

/// `a += b * scale` elementwise (the hyper-connection write).
pub fn add_scaled(a: &mut [f32], b: &[f32], scale: f32) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y * scale;
    }
}
