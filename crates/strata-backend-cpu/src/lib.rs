//! # strata-backend-cpu
//!
//! The CPU implementation of [`strata_core::backend::ComputeBackend`]: every
//! primitive delegates to the scalar kernels in `strata_core::ops`, which are
//! the reference the GPU backend is checked against.

use strata_core::backend::{ComputeBackend, QuantTensor, RopeCfg};
use strata_core::error::Result;
use strata_core::ops;
use strata_core::weights::ExpertView;

/// The CPU backend. A unit struct: all state lives in the engine's buffers.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn gemv(&self, w: &QuantTensor<'_>, x: &[f32], y: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        ops::gemv(w, x, y, scratch);
        Ok(())
    }

    fn expert_gemv(
        &self,
        w: &ExpertView<'_>,
        e: usize,
        x: &[f32],
        y: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<()> {
        ops::expert_gemv(w, e, x, y, scratch);
        Ok(())
    }

    fn rms_norm(&self, x: &mut [f32], w: &[f32], eps: f64) -> Result<()> {
        ops::rms_norm(x, w, eps);
        Ok(())
    }

    fn rope_neox(&self, head: &mut [f32], cfg: &RopeCfg) -> Result<()> {
        ops::rope_neox(head, cfg);
        Ok(())
    }

    fn softmax(&self, xs: &mut [f32]) -> Result<()> {
        ops::softmax(xs);
        Ok(())
    }

    fn l2_norm(&self, head: &mut [f32], eps: f64) -> Result<()> {
        ops::l2_norm(head, eps);
        Ok(())
    }

    fn add(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
        ops::add_into(a, b);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use half::f16;
    use strata_gguf::quant::QuantType;

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
    fn gemv_matches_a_manual_dot() {
        let be = CpuBackend::new();
        assert_eq!(be.name(), "cpu");
        let data = f32_bytes(2, 3, |r, c| (r * 3 + c) as f32 + 1.0);
        let w = QuantTensor::new("w", QuantType::F32, 2, 3, &data);
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 2];
        let mut scratch = [0.0f32; 3];
        be.gemv(&w, &x, &mut y, &mut scratch).unwrap();
        assert!((y[0] - 14.0).abs() < 1e-4); // 1·1 + 2·2 + 3·3
        assert!((y[1] - 32.0).abs() < 1e-4); // 4·1 + 5·2 + 6·3
    }

    #[test]
    fn q8_0_gemv_matches_dequant_dot() {
        let be = CpuBackend::new();
        // One Q8_0 block: d = 0.5, quants cycling -3..3.
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        for j in 0..32 {
            block[2 + j] = (j as i32 % 7 - 3) as i8 as u8;
        }
        let w = QuantTensor::new("w", QuantType::Q8_0, 1, 32, &block);
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 1.0).collect();
        let mut y = [0.0f32];
        let mut scratch = [0.0f32; 32];
        be.gemv(&w, &x, &mut y, &mut scratch).unwrap();
        let expect: f64 = (0..32)
            .map(|j| f64::from((j as i32 % 7 - 3) as f32) * 0.5 * f64::from(x[j]))
            .sum();
        assert!((f64::from(y[0]) - expect).abs() < 1e-4);
    }

    #[test]
    fn elementwise_primitives_match_the_reference() {
        let be = CpuBackend::new();
        let mut x = [3.0f32, 4.0];
        be.rms_norm(&mut x, &[2.0, 2.0], 0.0).unwrap();
        let s = 2.0 / 12.5f64.sqrt();
        assert!((x[0] - (3.0 * s as f32)).abs() < 1e-6);

        let mut head = [1.0f32, 0.0];
        let cfg = RopeCfg {
            pos: 0,
            n_rot: 2,
            head_dim: 2,
            freq_base: 1.0,
        };
        be.rope_neox(&mut head, &cfg).unwrap();
        assert_eq!(head, [1.0, 0.0], "position 0 is the identity");

        let mut xs = [1.0f32, 2.0, 3.0];
        be.softmax(&mut xs).unwrap();
        assert!((xs.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let mut h = [3.0f32, 4.0];
        be.l2_norm(&mut h, 0.0).unwrap();
        assert!((h.iter().map(|v| v * v).sum::<f32>()).sqrt() - 1.0 < 1e-6);

        let mut a = [1.0f32, 2.0];
        be.add(&mut a, &[10.0, 20.0]).unwrap();
        assert_eq!(a, [11.0, 22.0]);
    }
}
