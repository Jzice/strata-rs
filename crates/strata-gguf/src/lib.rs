//! # strata-gguf
//!
//! GGUF model file reading, metadata / model-geometry parsing and ggml
//! quantization support.
//!
//! The reader memory-maps the file and validates the header, so a 40 GB model
//! costs no resident memory beyond the tensors that are actually touched.
//! Dequantization is exact against the ggml reference implementation.

pub mod dequant;
pub mod dequant_iq;
pub mod error;
pub mod family;
pub mod geometry;
pub mod metadata;
pub mod quant;
pub mod reader;
pub mod tables;

pub use error::{Error, Result};
pub use family::ModelFamily;
pub use geometry::ModelGeometry;
pub use quant::QuantType;
pub use reader::{split_paths, GgufFile, GgufModel, TensorInfo};
