//! Per-sequence state: the residual stack, every layer's recurrent state and
//! KV cache, the position counter and the token history the sampler's
//! penalties read.

use crate::backend::{ComputeBackend, CpuBackend, GdnState};
use crate::engine::Engine;
use crate::gdn::GdnScratch;
use crate::qsa::{QsaCache, QsaScratch};
use std::sync::Arc;

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("backend", &self.backend.name())
            .field("pos", &self.pos)
            .field("tokens", &self.tokens.len())
            .finish_non_exhaustive()
    }
}

/// All scratch buffers one forward pass needs.
#[derive(Debug, Default)]
pub struct Scratch {
    pub xn: Vec<f32>,
    pub lo: Vec<f32>,
    pub gate_all: Vec<f32>,
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
    pub inject2: Vec<f32>,
    pub block_out: Vec<f32>,
    pub moe_h: Vec<f32>,
    pub moe_part: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub moe_logits: Vec<f32>,
    pub deq: Vec<f32>,
    pub gdn: Vec<GdnScratch>,
    pub qsa: Vec<QsaScratch>,
}

/// One decoding session.
pub struct Session {
    pub engine: Arc<Engine>,
    /// The backend that runs the projections. Defaults to the CPU reference.
    pub backend: Arc<dyn ComputeBackend>,
    // `dyn ComputeBackend` is not `Debug`, so `Session` implements it by hand
    // (the backend's name is what a caller wants to see anyway).
    /// The hyper-connection residual stack (`hc·n_embd`), qwen4exp only.
    pub r: Vec<f32>,
    /// The plain residual (`n_embd`), qwen35moe only.
    pub x: Vec<f32>,
    pub gdn: Vec<GdnState>,
    pub qsa: Vec<QsaCache>,
    pub scratch: Scratch,
    /// The next token's position.
    pub pos: usize,
    /// Every token seen, for the sampler's penalties.
    pub tokens: Vec<i32>,
}

impl Session {
    pub fn new(engine: Arc<Engine>) -> Self {
        let g = &engine.geometry;
        // The widest row any gemv dequantizes: the hyper-connection weights read
        // the whole `hc·n_embd` stack, and a dense FFN's down-projection reads a
        // row of width `n_ff`, so both bound the scratch.
        let widest = g
            .hc_dim()
            .max(g.n_embd)
            .max(g.ssm_value_dim)
            .max(g.n_ff)
            .max(1) as usize;
        let mut scratch = Scratch {
            xn: vec![0.0; g.hc_dim().max(g.n_embd) as usize],
            lo: vec![0.0; g.hc_lr.max(1) as usize],
            gate_all: vec![0.0; g.hc_dim().max(g.n_embd) as usize],
            mixed: vec![0.0; g.n_embd as usize],
            inject: vec![0.0; g.hc.max(1) as usize],
            inject2: vec![0.0; g.hc.max(1) as usize],
            block_out: vec![0.0; g.n_embd as usize],
            moe_h: vec![0.0; g.n_ff.max(1) as usize],
            moe_part: vec![0.0; g.n_ff.max(1) as usize],
            moe_out: vec![0.0; g.n_embd as usize],
            moe_logits: vec![0.0; g.n_expert as usize],
            deq: vec![0.0; widest],
            gdn: Vec::new(),
            qsa: Vec::new(),
        };
        let mut gdn = Vec::new();
        let mut qsa = Vec::new();
        for l in 0..g.n_layers as usize {
            if g.is_qsa_layer(l as i64) {
                let layer = engine.qsa_layer(l).expect("qsa layer");
                qsa.push(QsaCache::new(layer));
                scratch.qsa.push(QsaScratch::new(layer));
            } else {
                let layer = engine.gdn_layer(l).expect("gdn layer");
                gdn.push(GdnState::new(
                    layer.n_k_heads,
                    layer.n_v_heads,
                    layer.state_size,
                    layer.d_conv,
                    layer.conv_channels,
                ));
                scratch.gdn.push(GdnScratch::new(layer));
            }
        }
        Self {
            r: vec![0.0; g.hc_dim().max(g.n_embd) as usize],
            x: vec![0.0; g.n_embd as usize],
            engine,
            backend: Arc::new(CpuBackend::new()),
            gdn,
            qsa,
            scratch,
            pos: 0,
            tokens: Vec::new(),
        }
    }

    /// A session that runs its matmuls on `backend`.
    pub fn with_backend(engine: Arc<Engine>, backend: Arc<dyn ComputeBackend>) -> Self {
        let mut s = Self::new(engine);
        s.backend = backend;
        s
    }

    /// The backend this session computes with.
    pub fn backend(&self) -> &dyn ComputeBackend {
        self.backend.as_ref()
    }

    /// Clears every state for a new sequence.
    pub fn reset(&mut self) {
        self.r.fill(0.0);
        self.x.fill(0.0);
        for s in &mut self.gdn {
            s.reset();
        }
        for c in &mut self.qsa {
            c.reset();
        }
        self.pos = 0;
        self.tokens.clear();
    }
}
