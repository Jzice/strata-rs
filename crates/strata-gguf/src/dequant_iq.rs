//! The i-quant dequantizers (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S),
//! transcribed from ggml at commit 3cf03257 via Strata's
//! `metal/src/ref_quant.cpp`. The grid and sign tables come from
//! [`crate::tables`], extracted verbatim from the vendored ggml-common.h.
//!
//! As in [`crate::dequant`]: bit-exact transcription, so the casts and short
//! names stay as the reference spells them.
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use crate::dequant::{f16_to_f32, rd16};
use crate::tables::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

#[inline]
fn sign_of(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// IQ2_XXS: 256 elements from 66 bytes.
pub fn dequant_iq2_xxs(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let qs = &b[2..66];
    let mut y = 0;
    for ib32 in 0..8usize {
        let aux32 = [
            u32::from_le_bytes(qs[ib32 * 8..ib32 * 8 + 4].try_into().unwrap()),
            u32::from_le_bytes(qs[ib32 * 8 + 4..ib32 * 8 + 8].try_into().unwrap()),
        ];
        let aux8 = aux32[0].to_le_bytes();
        let db = d * (0.5 + ((aux32[1] >> 28) as f32)) * 0.25;
        for l in 0..4usize {
            let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
            for j in 0..8usize {
                out[y + j] = db * grid[j] as f32 * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

/// IQ2_XS: 256 elements from 74 bytes.
pub fn dequant_iq2_xs(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let q8 = &b[2..66];
    let scales = &b[66..74];
    let mut y = 0;
    for ib32 in 0..8usize {
        let db0 = d * (0.5 + (scales[ib32] & 15) as f32) * 0.25;
        let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
        for l in 0..4usize {
            let q = rd16(&q8[2 * (4 * ib32 + l)..]);
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            let db = if l < 2 { db0 } else { db1 };
            for j in 0..8usize {
                out[y + j] = db * grid[j] as f32 * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

/// IQ2_S: 256 elements from 82 bytes.
pub fn dequant_iq2_s(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let mut qs_off = 2usize;
    let qh = &b[66..74];
    let scales = &b[74..82];
    let mut signs_off = 34usize; // qs + 32
    let mut y = 0;
    for ib32 in 0..8usize {
        let db0 = d * (0.5 + (scales[ib32] & 15) as f32) * 0.25;
        let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
        for l in 0..4usize {
            let dl = if l < 2 { db0 } else { db1 };
            let idx = (b[qs_off + l] as usize) | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let signs = b[signs_off + l];
            for j in 0..8usize {
                out[y + j] = dl * grid[j] as f32 * sign_of(signs, j);
            }
            y += 8;
        }
        qs_off += 4;
        signs_off += 4;
    }
}

/// IQ3_XXS: 256 elements from 98 bytes.
pub fn dequant_iq3_xxs(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let qs = &b[2..66];
    let sas = &b[66..98];
    let mut y = 0;
    let mut qs_off = 0;
    for ib32 in 0..8usize {
        let aux32 = u32::from_le_bytes(sas[ib32 * 4..ib32 * 4 + 4].try_into().unwrap());
        let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
        for l in 0..4usize {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4usize {
                out[y + j] = db * g1[j] as f32 * sign_of(signs, j);
                out[y + j + 4] = db * g2[j] as f32 * sign_of(signs, j + 4);
            }
            y += 8;
        }
        qs_off += 8;
    }
}

/// IQ3_S: 256 elements from 110 bytes.
pub fn dequant_iq3_s(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(rd16(b));
    let mut qs_off = 2usize;
    let qh = &b[66..74];
    let signs_base = 74usize;
    let scales = &b[106..110];
    let mut y = 0;
    for ib32 in (0..8usize).step_by(2) {
        let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 15) as f32);
        let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
        for l in 0..4usize {
            let g1_idx = (b[qs_off + 2 * l] as usize) | (((qh[0] as usize) << (8 - 2 * l)) & 256);
            let g2_idx = (b[qs_off + 2 * l + 1] as usize) | (((qh[0] as usize) << (7 - 2 * l)) & 256);
            let g1 = IQ3S_GRID[g1_idx].to_le_bytes();
            let g2 = IQ3S_GRID[g2_idx].to_le_bytes();
            let signs = b[signs_base + l];
            for j in 0..4usize {
                out[y + j] = db1 * g1[j] as f32 * sign_of(signs, j);
                out[y + j + 4] = db1 * g2[j] as f32 * sign_of(signs, j + 4);
            }
            y += 8;
        }
        qs_off += 8;
        for l in 0..4usize {
            let g1_idx = (b[qs_off + 2 * l] as usize) | (((qh[1] as usize) << (8 - 2 * l)) & 256);
            let g2_idx = (b[qs_off + 2 * l + 1] as usize) | (((qh[1] as usize) << (7 - 2 * l)) & 256);
            let g1 = IQ3S_GRID[g1_idx].to_le_bytes();
            let g2 = IQ3S_GRID[g2_idx].to_le_bytes();
            let signs = b[signs_base + 4 + l];
            for j in 0..4usize {
                out[y + j] = db2 * g1[j] as f32 * sign_of(signs, j);
                out[y + j + 4] = db2 * g2[j] as f32 * sign_of(signs, j + 4);
            }
            y += 8;
        }
        qs_off += 8;
    }
}
