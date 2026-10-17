//! # strata-tokenizer
//!
//! BPE tokenizer whose vocabulary, merges and special tokens are read from the
//! GGUF file itself (`tokenizer.ggml.*` metadata keys), matching the upstream
//! engine's behaviour, plus chat-template rendering for the model families.

pub mod bpe;
pub mod byte_level;
pub mod error;
pub mod pretok;
pub mod template;
pub mod tokenizer;

pub use error::{Error, Result};
pub use template::{ChatTemplate, Message};
pub use tokenizer::{TokenType, Tokenizer};
