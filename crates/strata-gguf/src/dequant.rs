//! Row dequantization for every quantization type Strata's model files use.
//!
//! The scalar reference implementations are transcribed from Strata's
//! `include/strata/artifact/dequant.hpp` and `metal/src/ref_quant.cpp`
//! (themselves transcribed from ggml at commit 3cf03257), with the lookup
//! tables extracted verbatim into [`crate::tables`]. The i-quant families
//! live in [`crate::dequant_iq`].
//!
//! The integer<->float casts and the short binding names are deliberate:
//! every line is a bit-exact transcription of the reference, and renaming for
//! lint appeasement is how a transcription drifts.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use crate::error::{Error, Result};
use crate::quant::QuantType;
use half::f16;

#[inline]
pub(crate) fn rd16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

#[inline]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    f16::from_bits(h).to_f32()
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Dequantizes one row of `n` elements from `block` (which must hold at least
/// `QuantType::byte_size(n)` bytes) into `out` (exactly `n` elements).
pub fn dequantize_row(quant: QuantType, block: &[u8], out: &mut [f32]) -> Result<()> {
    let bs = quant.block_size();
    let ts = quant.type_size();
    let n = out.len();
    let n_blocks = n.div_ceil(bs);
    if n % bs != 0 || block.len() < n_blocks * ts {
        return Err(Error::Truncated {
            expected: n_blocks * ts,
            found: block.len(),
        });
    }
    for (i, chunk) in out.chunks_mut(bs).enumerate() {
        let b = &block[i * ts..i * ts + ts];
        dequant_block(quant, b, chunk)?;
    }
    Ok(())
}

/// One block: `out.len()` must equal `quant.block_size()`.
fn dequant_block(quant: QuantType, b: &[u8], out: &mut [f32]) -> Result<()> {
    match quant {
        QuantType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        QuantType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_to_f32(rd16(&b[i * 2..]));
            }
        }
        QuantType::Bf16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_to_f32(rd16(&b[i * 2..]));
            }
        }
        QuantType::Q2_0 => dequant_q2_0(b, out),
        QuantType::Q8_0 => dequant_q8_0(b, out),
        QuantType::Q8_1 => dequant_q8_1(b, out),
        QuantType::Q4_0 => dequant_q4_0(b, out),
        QuantType::Q4_1 => dequant_q4_1(b, out),
        QuantType::Q5_0 => dequant_q5_0(b, out),
        QuantType::Q5_1 => dequant_q5_1(b, out),
        QuantType::Iq4Nl => dequant_iq4_nl(b, out),
        QuantType::Iq4Xs => dequant_iq4_xs(b, out),
        QuantType::Q6K => dequant_q6_k(b, out),
        QuantType::Q4K => dequant_q4_k(b, out),
        QuantType::Q5K => dequant_q5_k(b, out),
        QuantType::Q3K => dequant_q3_k(b, out),
        QuantType::Q2K => dequant_q2_k(b, out),
        QuantType::Iq2Xxs => crate::dequant_iq::dequant_iq2_xxs(b, out),
        QuantType::Iq2Xs => crate::dequant_iq::dequant_iq2_xs(b, out),
        QuantType::Iq2S => crate::dequant_iq::dequant_iq2_s(b, out),
        QuantType::Iq3Xxs => crate::dequant_iq::dequant_iq3_xxs(b, out),
        QuantType::Iq3S => crate::dequant_iq::dequant_iq3_s(b, out),
        QuantType::Mxfp4 => dequant_mxfp4(b, out),
        QuantType::Nvfp4 => dequant_nvfp4(b, out),
        QuantType::Iq1M | QuantType::Q1_0 | QuantType::I8 => {
            return Err(Error::UnsupportedQuant {
                name: quant.name().to_string(),
            })
        }
    }
    Ok(())
}

/// Q2_0: Strata's ternary 2-bit block. 64 elements from 18 bytes:
/// `fp16 d`, then 4 two-bit codes per byte, LSB first; code `{0,1,2,3}` maps
/// to symbol `{-1,0,+1,+2}` and `v = (code - 1) * d`.
fn dequant_q2_0(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let qs = &b[2..];
    for (j, o) in out.iter_mut().enumerate() {
        let code = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        *o = (code as f32 - 1.0) * d;
    }
}

fn dequant_q8_0(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    for (j, o) in out.iter_mut().enumerate() {
        *o = (b[2 + j] as i8) as f32 * d;
    }
}

/// Q8_1: `fp16 d`, `fp16 s` (= d * sum(qs)), `int8 qs[32]`.
fn dequant_q8_1(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    for (j, o) in out.iter_mut().enumerate() {
        *o = (b[4 + j] as i8) as f32 * d;
    }
}

fn dequant_q4_0(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    for j in 0..16 {
        out[j] = ((b[2 + j] & 0x0F) as f32 - 8.0) * d;
        out[j + 16] = ((b[2 + j] >> 4) as f32 - 8.0) * d;
    }
}

fn dequant_q4_1(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let m = f16_to_f32(rd16(&b[2..]));
    for j in 0..16 {
        out[j] = (b[4 + j] & 0x0F) as f32 * d + m;
        out[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
    }
}

fn dequant_q5_0(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let h = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..];
    for j in 0..16 {
        let x0 = (qs[j] & 0x0F) as u32 | ((h >> j) & 1) << 4;
        let x1 = (qs[j] >> 4) as u32 | ((h >> (j + 16)) & 1) << 4;
        out[j] = (x0 as f32 - 16.0) * d;
        out[j + 16] = (x1 as f32 - 16.0) * d;
    }
}

fn dequant_q5_1(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let m = f16_to_f32(rd16(&b[2..]));
    let h = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..];
    for j in 0..16 {
        let x0 = (qs[j] & 0x0F) as u32 | ((h >> j) & 1) << 4;
        let x1 = (qs[j] >> 4) as u32 | ((h >> (j + 16)) & 1) << 4;
        out[j] = x0 as f32 * d + m;
        out[j + 16] = x1 as f32 * d + m;
    }
}

fn dequant_iq4_nl(b: &[u8], out: &mut [f32]) {
    use crate::tables::KVALUES_IQ4NL;
    let d = f16_to_f32(rd16(b));
    for j in 0..16 {
        out[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0x0F) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

fn dequant_iq4_xs(b: &[u8], out: &mut [f32]) {
    use crate::tables::KVALUES_IQ4NL;
    let d = f16_to_f32(rd16(b));
    let scales_h = rd16(&b[2..]);
    let scales_l = &b[4..8];
    let qs = &b[8..];
    let mut y = 0;
    for ib in 0..8usize {
        let ls =
            ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as u32 | ((scales_h as u32 >> (2 * ib)) & 3) << 4;
        let dl = d * (ls as f32 - 32.0);
        for j in 0..16usize {
            out[y + j] = dl * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            out[y + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
        y += 32;
    }
}

fn dequant_q6_k(b: &[u8], out: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16_to_f32(rd16(&b[208..]));
    let mut y = 0;
    let mut ql_off = 0;
    let mut qh_off = 0;
    let mut sc_off = 0;
    for _ in 0..2 {
        for l in 0..32usize {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0x0F) as i32 | (((qh[qh_off + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32 | (((qh[qh_off + l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 3) as i32) << 4) - 32;
            out[y + l] = d * sc[sc_off + is] as f32 * q1 as f32;
            out[y + l + 32] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
            out[y + l + 64] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
            out[y + l + 96] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
        }
        y += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}

/// The 6-bit scale/min unpacking shared by Q4_K / Q5_K / Q2_K.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let mn = f16_to_f32(rd16(&b[2..]));
    let scales = &b[4..16];
    let q = &b[16..144];
    let mut y = 0;
    let mut q_off = 0;
    let mut is = 0;
    for _ in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * sc as f32;
        let m1 = mn * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc as f32;
        let m2 = mn * m as f32;
        for l in 0..32usize {
            out[y + l] = d1 * (q[q_off + l] & 0x0F) as f32 - m1;
            out[y + 32 + l] = d2 * (q[q_off + l] >> 4) as f32 - m2;
        }
        y += 64;
        q_off += 32;
        is += 2;
    }
}

fn dequant_q5_k(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let mn = f16_to_f32(rd16(&b[2..]));
    let sc = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    let mut y = 0;
    let mut ql_off = 0;
    let mut is = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _ in 0..4 {
        let (s, m) = get_scale_min_k4(is, sc);
        let d1 = d * s as f32;
        let m1 = mn * m as f32;
        let (s, m) = get_scale_min_k4(is + 1, sc);
        let d2 = d * s as f32;
        let m2 = mn * m as f32;
        for l in 0..32usize {
            out[y + l] =
                d1 * ((ql[ql_off + l] & 0x0F) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - m1;
            out[y + 32 + l] =
                d2 * ((ql[ql_off + l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - m2;
        }
        y += 64;
        ql_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

fn dequant_q3_k(b: &[u8], out: &mut [f32]) {
    let hm = &b[0..32];
    let q = &b[32..96];
    let d_all = f16_to_f32(rd16(&b[108..]));
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes([b[96 + i * 4], b[97 + i * 4], b[98 + i * 4], b[99 + i * 4]]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|w| w.to_le_bytes().map(|b| b as i8))
        .collect();

    let mut y = 0;
    let mut is = 0;
    let mut m: u8 = 1;
    let mut q_off = 0;
    for _ in 0..2 {
        let mut shift = 0;
        for _ in 0..4 {
            let dl = d_all * (scales[is] - 32) as f32;
            is += 1;
            for l in 0..16usize {
                out[y + l] = dl
                    * (((q[q_off + l] >> shift) & 3) as i8 as f32 - if hm[l] & m != 0 { 0.0 } else { 4.0 });
            }
            let dl = d_all * (scales[is] - 32) as f32;
            is += 1;
            for l in 0..16usize {
                out[y + 16 + l] = dl
                    * (((q[q_off + l + 16] >> shift) & 3) as i8 as f32
                        - if hm[l + 16] & m != 0 { 0.0 } else { 4.0 });
            }
            y += 32;
            shift += 2;
            m <<= 1;
        }
        q_off += 32;
    }
}

/// Q2_K: 256 elements, 84 bytes: `fp16 d`, `fp16 dmin`, `scales[16]`
/// (4-bit scale/min pairs), `qs[64]`. Two 128-element groups, each 32 bytes
/// of qs, each split into four 32-element sub-groups alternating d1/d2 over
/// the four 2-bit fields of every byte.
fn dequant_q2_k(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let mn = f16_to_f32(rd16(&b[2..]));
    let sc = &b[4..20];
    let q = &b[20..84];
    let mut y = 0;
    let mut q_off = 0;
    let mut is = 0;
    for _ in 0..2 {
        let (s, m) = get_scale_min_k4(is, sc);
        let d1 = d * s as f32;
        let m1 = mn * m as f32;
        let (s, m) = get_scale_min_k4(is + 1, sc);
        let d2 = d * s as f32;
        let m2 = mn * m as f32;
        for l in 0..32usize {
            out[y + l] = d1 * (q[q_off + l] & 3) as f32 - m1;
            out[y + l + 32] = d2 * ((q[q_off + l] >> 2) & 3) as f32 - m2;
            out[y + l + 64] = d1 * ((q[q_off + l] >> 4) & 3) as f32 - m1;
            out[y + l + 96] = d2 * ((q[q_off + l] >> 6) & 3) as f32 - m2;
        }
        y += 128;
        q_off += 32;
        is += 2;
    }
}

/// E8M0 scale byte -> f32: the value is `2^(e-127)`, which in f32 bits is
/// exactly `e << 23` (both formats use a bias of 127); 0 is zero and 0xFF is
/// NaN.
fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        0 => 0.0,
        0xFF => f32::NAN,
        _ => f32::from_bits((e as u32) << 23),
    }
}

/// UE4M3 scale byte -> f32 (E4M3, no inf; 0x7F/0xFF are NaN).
fn ue4m3_to_f32(e: u8) -> f32 {
    let sign = (e & 0x80) != 0;
    let exp = ((e >> 3) & 0xF) as i32;
    let man = (e & 0x7) as i32;
    let v = if exp == 0 {
        (man as f32) / 8.0 * 2f32.powi(-6)
    } else if exp == 15 && man == 7 {
        f32::NAN
    } else {
        2f32.powi(exp - 7) * (1.0 + man as f32 / 8.0)
    };
    if sign {
        -v
    } else {
        v
    }
}

/// MXFP4: one E8M0 scale, 16 bytes of 4-bit E2M1 codes (low nibble j, high
/// nibble j+16).
fn dequant_mxfp4(b: &[u8], out: &mut [f32]) {
    use crate::tables::KVALUES_FP4;
    let d = e8m0_to_f32(b[0]);
    for j in 0..16usize {
        out[j] = d * KVALUES_FP4[(b[1 + j] & 0x0F) as usize] as f32 * 0.5;
        out[j + 16] = d * KVALUES_FP4[(b[1 + j] >> 4) as usize] as f32 * 0.5;
    }
}

/// NVFP4: four UE4M3 sub-block scales, 32 bytes of E2M1 codes; sub-block `s`
/// keeps value `j` in the low nibble of `qs[8s + j]` and value `j + 8` in the
/// high one.
fn dequant_nvfp4(b: &[u8], out: &mut [f32]) {
    use crate::tables::KVALUES_FP4;
    let d = [
        ue4m3_to_f32(b[0]),
        ue4m3_to_f32(b[1]),
        ue4m3_to_f32(b[2]),
        ue4m3_to_f32(b[3]),
    ];
    for s in 0..4usize {
        let ds = d[s];
        for j in 0..8usize {
            out[s * 16 + j] = ds * KVALUES_FP4[(b[4 + 8 * s + j] & 0x0F) as usize] as f32 * 0.5;
            out[s * 16 + j + 8] = ds * KVALUES_FP4[(b[4 + 8 * s + j] >> 4) as usize] as f32 * 0.5;
        }
    }
}
