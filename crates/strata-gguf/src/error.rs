use thiserror::Error;

/// Errors produced by GGUF reading, metadata parsing and dequantization.
#[derive(Debug, Error)]
pub enum Error {
    #[error("cannot open GGUF file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("not a GGUF file (bad magic {magic:#x})")]
    BadMagic { magic: u32 },

    #[error("unsupported GGUF version {version} (expected 2 or 3)")]
    UnsupportedVersion { version: u32 },

    #[error("truncated GGUF file: expected {expected} more bytes, found {found}")]
    Truncated { expected: usize, found: usize },

    #[error("invalid metadata value type {value_type}")]
    BadValueType { value_type: u32 },

    #[error("invalid metadata key length {len} (max 65535)")]
    BadKeyLength { len: u64 },

    #[error("metadata key '{key}' is not valid UTF-8")]
    KeyNotUtf8 { key: String },

    #[error("string is not valid UTF-8")]
    StringNotUtf8,

    #[error("tensor data is not aligned to {alignment} bytes")]
    BadTensorAlignment { alignment: usize },

    #[error("tensor '{name}' is missing from the file")]
    TensorNotFound { name: String },

    #[error("tensor '{name}' has {found} dimensions, expected {expected}")]
    WrongTensorDims {
        name: String,
        found: usize,
        expected: usize,
    },

    #[error("metadata key '{key}' is missing")]
    MissingMetadata { key: String },

    #[error("metadata key '{key}' has the wrong type")]
    WrongMetadataType { key: String },

    #[error("unsupported quantization type {name:?}")]
    UnsupportedQuant { name: String },

    #[error("model geometry is unsupported: {message}")]
    UnsupportedGeometry { message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
