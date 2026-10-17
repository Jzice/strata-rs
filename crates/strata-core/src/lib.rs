//! # strata-core
//!
//! Engine core: the model families (`qwen4exp`, `qwen35moe`), the layer
//! implementations, the samplers and the session/state management.
//!
//! ```text
//! artifact layer   strata-gguf (reader, geometry, dequant) + strata-tokenizer
//! this crate       Engine (load, forward), layers (GDN/QSA/MoE/HC), sampler
//! backends         strata-backend-cpu (today: the scalar ops in `ops`)
//!                  strata-backend-wgpu (a later phase)
//! ```

pub mod backend;
pub mod engine;
pub mod error;
pub mod gdn;
pub mod generate;
pub mod hyper;
pub mod moe;
pub mod ops;
pub mod qsa;
pub mod qwen35;
pub mod sampler;
pub mod session;
pub mod weights;

pub use engine::{Engine, Layer};
pub use error::{Error, Result};
pub use generate::{eos_ids, generate, FinishReason, GenerateParams, GenerateStats, Stop};
pub use sampler::{sample, SamplerParams};
pub use session::Session;
