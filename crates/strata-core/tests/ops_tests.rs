//! Unit tests for the scalar compute primitives.

#![allow(clippy::float_cmp, clippy::needless_range_loop)]

use half::{bf16, f16};
use strata_core::backend::{QuantTensor, RopeCfg};
use strata_core::ops::*;
use strata_gguf::quant::QuantType;

/// An f32 matrix as bytes, in the engine's `[rows, cols]` view.
fn f32_bytes(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(rows * cols * 4);
    for r in 0..rows {
        for c in 0..cols {
            v.extend_from_slice(&f(r, c).to_le_bytes());
        }
    }
    v
}

#[test]
fn gemv_matches_manual_dot() {
    let rows = 3;
    let cols = 4;
    let data = f32_bytes(rows, cols, |r, c| (r * cols + c) as f32 + 1.0);
    let w = QuantTensor::new("w", QuantType::F32, rows, cols, &data);
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 3];
    let mut scratch = vec![0.0f32; cols];
    gemv(&w, &x, &mut y, &mut scratch);
    for r in 0..rows {
        let expect: f64 = (0..cols)
            .map(|c| f64::from((r * cols + c) as f32 + 1.0) * f64::from(x[c]))
            .sum();
        assert!(
            (f64::from(y[r]) - expect).abs() < 1e-4,
            "row {r}: {} vs {expect}",
            y[r]
        );
    }
}

#[test]
fn gemv_q8_0_matches_dequant_dot() {
    // One Q8_0 block of 32: d = 0.5, quants 1, 2, -3, ...
    let mut block = vec![0u8; 34];
    block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    for j in 0..32 {
        block[2 + j] = (j as i32 % 7 - 3) as i8 as u8;
    }
    let w = QuantTensor::new("w", QuantType::Q8_0, 1, 32, &block);
    let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 1.0).collect();
    let mut y = [0.0f32];
    let mut scratch = vec![0.0f32; 32];
    gemv(&w, &x, &mut y, &mut scratch);
    let expect: f64 = (0..32)
        .map(|j| f64::from((j as i32 % 7 - 3) as f32) * 0.5 * f64::from(x[j]))
        .sum();
    assert!((f64::from(y[0]) - expect).abs() < 1e-4, "{} vs {expect}", y[0]);
}

#[test]
fn gemv_bf16_rounds_like_the_reference() {
    // A bf16 weight of 1.0039063 (not representable) dequantizes exactly.
    let rows = 1;
    let cols = 2;
    let mut data = Vec::new();
    let v = bf16::from_f32(1.003_906_3);
    data.extend_from_slice(&v.to_bits().to_le_bytes());
    data.extend_from_slice(&bf16::from_f32(-2.0).to_bits().to_le_bytes());
    let w = QuantTensor::new("w", QuantType::Bf16, rows, cols, &data);
    let x = [1.0f32, 1.0];
    let mut y = [0.0f32];
    let mut scratch = vec![0.0f32; cols];
    gemv(&w, &x, &mut y, &mut scratch);
    assert_eq!(y[0], v.to_f32() - 2.0);
}

#[test]
fn rms_norm_known_values() {
    let mut x = [3.0f32, 4.0];
    let w = [2.0f32, 2.0];
    rms_norm(&mut x, &w, 0.0);
    // ss = (9+16)/2 = 12.5; scale = 1/sqrt(12.5); x = [3,4]*2/sqrt(12.5)
    let s = 2.0 / 12.5f64.sqrt();
    assert!((x[0] - (3.0 * s as f32)).abs() < 1e-6);
    assert!((x[1] - (4.0 * s as f32)).abs() < 1e-6);
}

#[test]
fn l2_norm_gives_unit_heads() {
    let mut head = [3.0f32, 4.0, 0.0];
    l2_norm(&mut head, 0.0);
    let norm = (head.iter().map(|v| v * v).sum::<f32>()).sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "norm was {norm}");
}

#[test]
fn rope_neox_position_zero_is_identity() {
    let mut head = [1.0f32, 2.0, 3.0, 4.0];
    let cfg = RopeCfg {
        pos: 0,
        n_rot: 4,
        head_dim: 4,
        freq_base: 1e7,
    };
    rope_neox(&mut head, &cfg);
    assert_eq!(head, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rope_neox_rotates_pairs() {
    // n_rot = 2: one pair (0, 1); theta = 1 at i = 0.
    let mut head = [1.0f32, 0.0];
    let cfg = RopeCfg {
        pos: 1,
        n_rot: 2,
        head_dim: 2,
        freq_base: 1.0,
    };
    rope_neox(&mut head, &cfg);
    // angle = 1 rad: (1*cos1 - 0, 1*sin1 + 0)
    assert!((head[0] - 1.0f64.cos() as f32).abs() < 1e-6);
    assert!((head[1] - 1.0f64.sin() as f32).abs() < 1e-6);
}

#[test]
fn softmax_sums_to_one() {
    let mut xs = [1.0f32, 2.0, 3.0, 0.5];
    softmax(&mut xs);
    let sum: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(xs[2] > xs[1] && xs[1] > xs[0] && xs[0] > xs[3]);
}

#[test]
fn silu_and_sigmoid_known_values() {
    assert!((silu(0.0) - 0.0).abs() < 1e-7);
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    // softplus's large-argument shortcut.
    assert_eq!(softplus(100.0), 100.0);
    assert!((softplus(0.0) - 2.0f64.ln() as f32).abs() < 1e-6);
}

#[test]
fn sampler_greedy_picks_argmax() {
    use strata_core::sampler::{sample, SamplerParams};
    let mut logits = [0.1f32, 5.0, 0.2, 4.9];
    let params = SamplerParams {
        temperature: 0.0,
        ..SamplerParams::default()
    };
    let pick = sample(&mut logits, &[], &params, 0);
    assert_eq!(pick, 1);
}

#[test]
fn sampler_penalties_push_repeats_down() {
    use strata_core::sampler::{sample, SamplerParams};
    // Token 1 has the highest logit but is in the history with a strong
    // repeat penalty, so greedy must pick another token.
    let mut logits = [0.0f32, 5.0, 4.0, 0.0];
    let params = SamplerParams {
        temperature: 0.0,
        penalty_repeat: 100.0,
        penalty_last_n: 8,
        ..SamplerParams::default()
    };
    let pick = sample(&mut logits, &[1], &params, 0);
    assert_eq!(pick, 2, "the repeated token must lose to the runner-up");
}

#[test]
fn sampler_top_k_limits_the_field() {
    use strata_core::sampler::{sample, SamplerParams};
    // With top_k = 1 the draw is deterministic even at temperature 1.
    let mut logits = [0.0f32, 1.0, 2.0, 3.0];
    let params = SamplerParams {
        temperature: 1.0,
        top_k: 1,
        seed: 42,
        ..SamplerParams::default()
    };
    let pick = sample(&mut logits, &[], &params, 0);
    assert_eq!(pick, 3);
}
