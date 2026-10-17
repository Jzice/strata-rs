use thiserror::Error;

/// Errors produced by tokenization and template rendering.
#[derive(Debug, Error)]
pub enum Error {
    #[error("vocabulary entry {index} is missing")]
    MissingVocabEntry { index: usize },

    #[error("vocabulary is empty; the GGUF has no tokenizer.ggml.tokens")]
    EmptyVocab,

    #[error("token type {value} is unknown")]
    UnknownTokenType { value: u32 },

    #[error("chat template is missing from the GGUF metadata")]
    MissingTemplate,

    #[error("chat template failed to render: {message}")]
    TemplateRender { message: String },

    #[error("failed to decode bytes as UTF-8")]
    NotUtf8,
}

pub type Result<T> = std::result::Result<T, Error>;
