use thiserror::Error;

/// Errors produced by the HTTP server.
#[derive(Debug, Error)]
pub enum Error {
    #[error("model is not loaded")]
    NotLoaded,

    #[error("model is already loaded")]
    AlreadyLoaded,

    #[error("model '{model}' was not found")]
    ModelNotFound { model: String },

    #[error("request body is not valid JSON: {message}")]
    BadBody { message: String },

    #[error("invalid request: {message}")]
    BadRequest { message: String },

    #[error("unauthorized")]
    Unauthorized,

    #[error("engine error: {0}")]
    Engine(#[from] strata_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
