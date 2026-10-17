//! Shared compute types and the backend seam.
//!
//! [`QuantTensor`] is the engine's view of a weight matrix: the ggml type,
//! the shape and the raw bytes straight from the memory-mapped GGUF.
//!
//! The layers call the scalar primitives in [`crate::ops`] directly, which is
//! the CPU reference path. [`ComputeBackend`] is the seam a device backend
//! plugs into: `strata-backend-cpu` implements it over the same primitives,
//! and the wgpu backend implements it in WGSL. A device backend takes over
//! the primitives it accelerates and falls back to the CPU path for the rest.

use crate::error::Result;
use crate::weights::ExpertView;
use strata_gguf::quant::QuantType;

/// The compute primitives a backend provides.
///
/// Every method has a CPU implementation in [`crate::ops`]; a device backend
/// overrides the ones it accelerates.
pub trait ComputeBackend: Send + Sync {
    /// The backend's name, for logs and `/v1/status`.
    fn name(&self) -> &'static str;

    /// `y = W x` for a quantized matrix. `scratch` holds one dequantized row.
    fn gemv(&self, w: &QuantTensor<'_>, x: &[f32], y: &mut [f32], scratch: &mut [f32]) -> Result<()>;

    /// `y = W_e x` for one expert of a 3-D expert tensor.
    fn expert_gemv(
        &self,
        w: &ExpertView<'_>,
        e: usize,
        x: &[f32],
        y: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<()>;

    /// In-place RMS normalisation with a per-channel scale.
    fn rms_norm(&self, x: &mut [f32], w: &[f32], eps: f64) -> Result<()>;

    /// In-place NEOX partial rotary embedding.
    fn rope_neox(&self, head: &mut [f32], cfg: &RopeCfg) -> Result<()>;

    /// In-place softmax.
    fn softmax(&self, xs: &mut [f32]) -> Result<()>;

    /// In-place unit-length normalisation of one head.
    fn l2_norm(&self, head: &mut [f32], eps: f64) -> Result<()>;

    /// `a += b` elementwise.
    fn add(&self, a: &mut [f32], b: &[f32]) -> Result<()>;
}

/// The CPU backend: the reference implementation, over [`crate::ops`].
///
/// This is what every device backend is measured against, and what the engine
/// runs when no GPU backend is handed to it.
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
        crate::ops::gemv(w, x, y, scratch);
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
        crate::ops::expert_gemv(w, e, x, y, scratch);
        Ok(())
    }

    fn rms_norm(&self, x: &mut [f32], w: &[f32], eps: f64) -> Result<()> {
        crate::ops::rms_norm(x, w, eps);
        Ok(())
    }

    fn rope_neox(&self, head: &mut [f32], cfg: &RopeCfg) -> Result<()> {
        crate::ops::rope_neox(head, cfg);
        Ok(())
    }

    fn softmax(&self, xs: &mut [f32]) -> Result<()> {
        crate::ops::softmax(xs);
        Ok(())
    }

    fn l2_norm(&self, head: &mut [f32], eps: f64) -> Result<()> {
        crate::ops::l2_norm(head, eps);
        Ok(())
    }

    fn add(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
        crate::ops::add_into(a, b);
        Ok(())
    }
}

/// A quantized matrix in ggml row-major layout: `rows` rows of `cols`
/// elements, stored as `data` in the given [`QuantType`].
#[derive(Debug, Clone, Copy)]
pub struct QuantTensor<'a> {
    pub name: &'a str,
    pub quant: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> QuantTensor<'a> {
    /// Builds a view, checking that `data` holds exactly the bytes the shape
    /// and type need.
    pub fn new(name: &'a str, quant: QuantType, rows: usize, cols: usize, data: &'a [u8]) -> Self {
        debug_assert_eq!(
            data.len(),
            quant.byte_size(rows * cols),
            "tensor {name}: {} bytes for {rows}x{cols} of {quant}",
            data.len()
        );
        Self {
            name,
            quant,
            rows,
            cols,
            data,
        }
    }

    /// The byte length of one row.
    pub fn row_bytes(&self) -> usize {
        self.quant.byte_size(self.cols)
    }

    /// The bytes of row `r`.
    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }
}

/// RoPE parameters for one attention forward pass.
#[derive(Debug, Clone, Copy)]
pub struct RopeCfg {
    /// Token position of the row being processed.
    pub pos: usize,
    /// Number of leading head dimensions that are rotated (NEOX pairs).
    pub n_rot: usize,
    /// Head dimension (stride between heads in the buffer).
    pub head_dim: usize,
    /// RoPE base frequency (theta).
    pub freq_base: f64,
}

/// The Gated-DeltaNet recurrent state of one layer.
///
/// `state` is `S·h_v·S` with layout `(i, h, j)`, `j` fastest: `state[h·S + j]`
/// is one column, and the stride between `i` values is `h_v·S`.
#[derive(Debug, Clone)]
pub struct GdnState {
    pub state: Vec<f32>,
    /// The causal depthwise conv buffer, `(d_conv - 1) · channels`,
    /// oldest-first.
    pub conv: Vec<f32>,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub state_size: usize,
    pub d_conv: usize,
    pub conv_channels: usize,
}

impl GdnState {
    pub fn new(
        n_k_heads: usize,
        n_v_heads: usize,
        state_size: usize,
        d_conv: usize,
        conv_channels: usize,
    ) -> Self {
        Self {
            state: vec![0.0; state_size * n_v_heads * state_size],
            conv: vec![0.0; (d_conv - 1) * conv_channels],
            n_k_heads,
            n_v_heads,
            state_size,
            d_conv,
            conv_channels,
        }
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.conv.fill(0.0);
    }
}
