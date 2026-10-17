//! The weight store: tensor views over the memory-mapped GGUF, plus the small
//! f32 tensors (norms, per-head parameters) materialized once at load.

use crate::backend::QuantTensor;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use strata_gguf::quant::QuantType;
use strata_gguf::{GgufModel, TensorInfo};

/// A 3-D expert tensor (`ffn_gate_exps.weight` and friends): `n_expert`
/// matrices of `[rows, cols]` each, expert-major.
#[derive(Debug, Clone, Copy)]
pub struct ExpertView<'a> {
    pub name: &'a str,
    pub quant: QuantType,
    pub n_expert: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

impl<'a> ExpertView<'a> {
    fn row_bytes(&self) -> usize {
        self.quant.byte_size(self.cols)
    }

    /// The bytes of input-dim row `i` of expert `e`.
    pub fn expert_row(&self, e: usize, i: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        let expert_base = e * self.rows * rb;
        &self.data[expert_base + i * rb..expert_base + (i + 1) * rb]
    }
}

/// Every weight the engine reads, by GGUF name.
#[derive(Debug)]
pub struct WeightStore<'a> {
    model: &'a GgufModel,
    views: HashMap<String, QuantTensor<'a>>,
    experts: HashMap<String, ExpertView<'a>>,
    /// Small tensors dequantized once (norms, `ssm_a`, `ssm_dt.bias`, ...).
    flat: HashMap<String, Arc<Vec<f32>>>,
}

impl<'a> WeightStore<'a> {
    /// Indexes the model's tensors. Only the directory is walked here; the
    /// data stays mapped on demand.
    pub fn new(model: &'a GgufModel) -> Self {
        let mut views = HashMap::new();
        let mut experts = HashMap::new();
        for shard in model.shards() {
            for t in shard.tensors() {
                if let Some(v) = view_of(model, t) {
                    views.insert(t.name.clone(), v);
                } else if let Some(v) = expert_view_of(model, t) {
                    experts.insert(t.name.clone(), v);
                }
            }
        }
        Self {
            model,
            views,
            experts,
            flat: HashMap::new(),
        }
    }

    /// The matrix view named `name`, or `None` when absent.
    pub fn tensor(&self, name: &str) -> Option<QuantTensor<'a>> {
        self.views.get(name).copied()
    }

    /// The expert tensor named `name`, or `None` when absent.
    pub fn expert(&self, name: &str) -> Option<ExpertView<'a>> {
        self.experts.get(name).copied()
    }

    /// The matrix view named `name`; an error naming it when absent.
    pub fn tensor_required(&self, name: &str) -> Result<QuantTensor<'a>> {
        self.tensor(name).ok_or_else(|| Error::MissingTensor {
            name: name.to_string(),
        })
    }

    /// A small 1-D tensor as f32, dequantized on first use.
    pub fn f32_vec(&mut self, name: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(v) = self.flat.get(name) {
            return Some(Arc::clone(v));
        }
        let t = self.tensor(name)?;
        let n = t.rows * t.cols;
        let mut out = vec![0.0f32; n];
        strata_gguf::dequant::dequantize_row(t.quant, t.data, &mut out).ok()?;
        let arc = Arc::new(out);
        self.flat.insert(name.to_string(), Arc::clone(&arc));
        Some(arc)
    }

    /// A scalar tensor as f32.
    pub fn f32_scalar(&mut self, name: &str) -> Result<f32> {
        self.f32_vec(name)
            .and_then(|v| v.first().copied())
            .ok_or_else(|| Error::MissingTensor {
                name: name.to_string(),
            })
    }

    /// The model's architecture family.
    pub fn family(&self) -> strata_gguf::ModelFamily {
        self.model
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(strata_gguf::ModelFamily::from_arch)
            .unwrap_or_default()
    }
}

/// Builds a view of `t` from whichever shard holds it. 1-D tensors become a
/// single row (the norms, `ssm_a`, `ssm_dt.bias`, ...).
fn view_of<'a>(model: &'a GgufModel, t: &'a TensorInfo) -> Option<QuantTensor<'a>> {
    if t.shape.len() > 2 {
        return None;
    }
    let quant = QuantType::from_id(t.ty).ok()?;
    let (rows, cols) = match t.shape.len() {
        1 => (1, *t.shape.first()? as usize),
        _ => (*t.shape.get(1)? as usize, *t.shape.first()? as usize),
    };
    let data = model.tensor_data(&t.name)?.ok()?;
    Some(QuantTensor::new(&t.name, quant, rows, cols, data))
}

/// Builds an expert view of a 3-D tensor: dims `[cols, rows, n_expert]`.
fn expert_view_of<'a>(model: &'a GgufModel, t: &'a TensorInfo) -> Option<ExpertView<'a>> {
    if t.shape.len() != 3 {
        return None;
    }
    let quant = QuantType::from_id(t.ty).ok()?;
    let cols = *t.shape.first()? as usize;
    let rows = *t.shape.get(1)? as usize;
    let n_expert = *t.shape.get(2)? as usize;
    let data = model.tensor_data(&t.name)?.ok()?;
    Some(ExpertView {
        name: &t.name,
        quant,
        n_expert,
        rows,
        cols,
        data,
    })
}
