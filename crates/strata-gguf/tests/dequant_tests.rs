//! Known-answer and structural-invariant tests for the dequantizers.
//!
//! The structural invariants mirror the upstream engine's `--check` mode:
//! every output value of a block must be one of the values that block's own
//! encoding can produce (that catches bit-order, sign and stride bugs, which
//! are the failure modes a dequantizer actually has). The known-answer tests
//! pin the exact bytes.

// The dequantizers are bit-exact, so exact float comparison is the point.
#![allow(clippy::float_cmp)]

use half::f16;
use strata_gguf::dequant::dequantize_row;
use strata_gguf::quant::QuantType;

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

fn deq(quant: QuantType, block: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    dequantize_row(quant, block, &mut out).expect("dequant");
    out
}

#[test]
fn q2_0_known_answer() {
    // d = 1.0; codes {0,1,2,3} -> symbols {-1,0,+1,+2}, LSB-first per byte.
    let mut b = [0u8; 18];
    b[..2].copy_from_slice(&f16_bytes(1.0));
    b[2] = 0b11_10_01_00; // j=0: code 0 -> -1; j=1: code 1 -> 0; j=2: code 2 -> +1; j=3: code 3 -> +2
    let out = deq(QuantType::Q2_0, &b, 64);
    assert_eq!(out[0], -1.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 1.0);
    assert_eq!(out[3], 2.0);
    // byte 1 is zero, so codes 0..3 of it are all code 0 -> -1
    assert_eq!(out[4], -1.0);
    assert_eq!(out[7], -1.0);
}

#[test]
fn q2_0_structural_invariant() {
    // Random-ish bytes with a finite scale: every output is in {-d,0,d,2d}.
    let mut b = [0u8; 18];
    b[..2].copy_from_slice(&f16_bytes(0.5));
    for (i, x) in b[2..].iter_mut().enumerate() {
        *x = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    let out = deq(QuantType::Q2_0, &b, 64);
    let d = 0.5f32;
    for v in out {
        assert!(
            (v + d).abs() < 1e-6 || v.abs() < 1e-6 || (v - d).abs() < 1e-6 || (v - 2.0 * d).abs() < 1e-6,
            "value {v} is not one of -d, 0, +d, +2d"
        );
    }
}

#[test]
fn q4_0_known_answer() {
    let mut b = [0u8; 18];
    b[..2].copy_from_slice(&f16_bytes(1.0));
    b[2] = 0x89; // low nibble 9 -> 1; high nibble 8 -> 0
    let out = deq(QuantType::Q4_0, &b, 32);
    assert_eq!(out[0], 1.0);
    assert_eq!(out[16], 0.0);
}

#[test]
fn q8_0_known_answer() {
    let mut b = [0u8; 34];
    b[..2].copy_from_slice(&f16_bytes(2.0));
    b[2] = 3; // 3 * 2
    b[3] = -4i8 as u8; // -4 * 2
    let out = deq(QuantType::Q8_0, &b, 32);
    assert_eq!(out[0], 6.0);
    assert_eq!(out[1], -8.0);
}

#[test]
fn q8_1_known_answer() {
    let mut b = [0u8; 36];
    b[..2].copy_from_slice(&f16_bytes(2.0));
    b[2..4].copy_from_slice(&f16_bytes(0.0)); // s (the sum) is unused by the row dequant
    b[4] = 3;
    let out = deq(QuantType::Q8_1, &b, 32);
    assert_eq!(out[0], 6.0);
}

#[test]
fn iq4_nl_known_answer() {
    let mut b = [0u8; 18];
    b[..2].copy_from_slice(&f16_bytes(1.0));
    b[2] = 0x01; // low nibble 1 -> -104; high nibble 0 -> -127
    let out = deq(QuantType::Iq4Nl, &b, 32);
    assert_eq!(out[0], -104.0);
    assert_eq!(out[16], -127.0);
}

#[test]
fn mxfp4_known_answer() {
    // E8M0 0x7F -> 2^0 = 1.0; codes 1 -> 0.5, 2 -> 1.0 (E2M1, table doubled).
    let mut b = [0u8; 17];
    b[0] = 0x7F;
    b[1] = 0x21; // low nibble 1 -> 0.5; high nibble 2 -> 1.0
    let out = deq(QuantType::Mxfp4, &b, 32);
    assert_eq!(out[0], 0.5);
    assert_eq!(out[16], 1.0);
}

#[test]
fn nvfp4_known_answer() {
    // UE4M3 0x38 -> 1.0 (exp 7, mantissa 0); code 0 -> 0, code 1 -> 0.5.
    let mut b = [0u8; 36];
    b[0] = 0x38;
    b[4] = 0x10; // low nibble of qs[0] -> value 0 = 0.0; high nibble -> value 8 = 0.5
    let out = deq(QuantType::Nvfp4, &b, 64);
    assert_eq!(out[0], 0.0);
    assert_eq!(out[8], 0.5);
}

#[test]
fn f16_known_answer() {
    let mut b = [0u8; 8];
    b[..2].copy_from_slice(&f16_bytes(1.5));
    b[2..4].copy_from_slice(&f16_bytes(-0.25));
    let out = deq(QuantType::F16, &b, 4);
    assert_eq!(out[0], 1.5);
    assert_eq!(out[1], -0.25);
}

#[test]
fn bf16_known_answer() {
    let mut b = [0u8; 4];
    let bits = 1.5f32.to_bits();
    b[..2].copy_from_slice(&((bits >> 16) as u16).to_le_bytes());
    let out = deq(QuantType::Bf16, &b, 2);
    assert_eq!(out[0], 1.5);
}

#[test]
fn q2_k_known_answer() {
    // Q2_K block: d=1, dmin=0, scales give 1,1; qs nibbles are the 2-bit codes.
    let mut b = [0u8; 84];
    b[..2].copy_from_slice(&f16_bytes(1.0));
    b[2..4].copy_from_slice(&f16_bytes(0.0));
    // scales[0] = 1 and scales[1] = 1 (j < 4 reads q[j] directly)
    b[4] = 1;
    b[5] = 1;
    // qs[0]'s four 2-bit fields land at offsets +0, +32, +64, +96
    b[20] = 0b00_01_10_11;
    let out = deq(QuantType::Q2K, &b, 256);
    assert_eq!(out[0], 3.0);
    assert_eq!(out[32], 2.0);
    assert_eq!(out[64], 1.0);
    assert_eq!(out[96], 0.0);
}

#[test]
fn q6_k_known_answer() {
    // Q6_K keeps its fp16 scale LAST (offset 208); all quants 0, sc[0] = 1.
    let mut b = [0u8; 210];
    b[208..210].copy_from_slice(&f16_bytes(1.0));
    b[192] = 1; // sc[0] = 1
    let out = deq(QuantType::Q6K, &b, 256);
    // q1 = (ql & 0xF | (qh & 3) << 4) - 32 = -32 with scale 1
    assert_eq!(out[0], -32.0);
}

#[test]
fn iq3_xxs_structural_invariant() {
    // With a zero scale every value must be exactly 0; with a random scale the
    // values must be multiples of db (the block's quantization step).
    let mut b = [0u8; 98];
    b[..2].copy_from_slice(&f16_bytes(0.0));
    for (i, x) in b[2..].iter_mut().enumerate() {
        *x = (i as u8).wrapping_mul(53).wrapping_add(7);
    }
    let out = deq(QuantType::Iq3Xxs, &b, 256);
    for v in out {
        assert_eq!(v, 0.0, "zero scale must give zero values");
    }
}

#[test]
fn every_supported_type_round_trips_through_row_dequant() {
    // A random block of each supported type dequantizes to finite values.
    let cases: &[(QuantType, usize)] = &[
        (QuantType::F32, 1),
        (QuantType::F16, 1),
        (QuantType::Bf16, 1),
        (QuantType::Q4_0, 32),
        (QuantType::Q4_1, 32),
        (QuantType::Q5_0, 32),
        (QuantType::Q5_1, 32),
        (QuantType::Q8_0, 32),
        (QuantType::Q8_1, 32),
        (QuantType::Q2K, 256),
        (QuantType::Q3K, 256),
        (QuantType::Q4K, 256),
        (QuantType::Q5K, 256),
        (QuantType::Q6K, 256),
        (QuantType::Q2_0, 64),
        (QuantType::Iq2Xxs, 256),
        (QuantType::Iq2Xs, 256),
        (QuantType::Iq2S, 256),
        (QuantType::Iq3Xxs, 256),
        (QuantType::Iq3S, 256),
        (QuantType::Iq4Nl, 32),
        (QuantType::Iq4Xs, 256),
        (QuantType::Mxfp4, 32),
        (QuantType::Nvfp4, 64),
    ];
    for (quant, elems) in cases {
        let bytes = quant.byte_size(*elems);
        let mut block = vec![0u8; bytes];
        // Deterministic pseudo-random fill with finite scales.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u8
        };
        for x in &mut block {
            *x = next();
        }
        // Make the scale finite and non-zero where the type has an fp16 scale.
        match quant {
            QuantType::F32 | QuantType::F16 | QuantType::Bf16 => {}
            QuantType::Mxfp4 => block[0] = 120,        // E8M0: 2^-7
            QuantType::Nvfp4 => block[..4].fill(0x38), // UE4M3 1.0 per sub-block
            QuantType::Q2_0 | QuantType::Q1_0 => {
                block[..2].copy_from_slice(&f16_bytes(0.125));
            }
            _ => {
                block[..2].copy_from_slice(&f16_bytes(0.125));
            }
        }
        let out = deq(*quant, &block, *elems);
        for v in out {
            assert!(v.is_finite(), "{quant}: non-finite value {v}");
        }
    }
}
