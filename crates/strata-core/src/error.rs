use thiserror::Error;

/// Errors produced by the engine core.
#[derive(Debug, Error)]
pub enum Error {
    #[error("model family {family:?} is not supported by this build")]
    UnsupportedFamily { family: String },

    #[error("model geometry is unsupported: {message}")]
    UnsupportedGeometry { message: String },

    #[error("tensor '{name}' is missing from the model file")]
    MissingTensor { name: String },

    #[error("tensor '{name}' has shape {found:?}, expected {expected:?}")]
    WrongTensorShape {
        name: String,
        found: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("quantization {quant:?} is not supported by this backend")]
    UnsupportedQuant { quant: String },

    #[error("context length {requested} exceeds the model maximum {max}")]
    ContextTooLong { requested: usize, max: usize },

    #[error("sampling failed: {message}")]
    Sampling { message: String },

    #[error("backend error: {message}")]
    Backend { message: String },

    #[error("model file error: {0}")]
    Gguf(#[from] strata_gguf::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
