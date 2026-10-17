//! The `qwen35` pre-tokenizer pattern.
//!
//! Transcribed from the ORACLE rather than from family resemblance:
//! llama.cpp `src/llama-vocab.cpp` L396, `LLAMA_VOCAB_PRE_TYPE_QWEN35`. The
//! `[\p{L}\p{M}]` (not `\p{L}`) and the `\p{M}` in the punctuation negation
//! are the whole point: a combining mark is `\p{M}`, so without it the mark
//! gets swallowed into the following punctuation run and a token like `_j`
//! never forms. Writing the QWEN3 pattern for `qwen35` looks right,
//! round-trips perfectly, and is wrong.
//!
//! The negative lookahead `(?!\S)` needs `fancy-regex`; the standard `regex`
//! crate cannot express it.

use fancy_regex::Regex;

/// The `qwen35` split pattern, shipped rather than re-derived so a port
/// cannot get `\p{M}` wrong again.
pub const QWEN35_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// The compiled pre-tokenizer. `Regex` is not `Clone`, so the clone
/// recompiles the pattern: a tokenizer is copied once per request, not per
/// token.
#[derive(Debug)]
pub struct PreTokenizer {
    re: Regex,
}

impl Clone for PreTokenizer {
    fn clone(&self) -> Self {
        // Unreachable in practice: the pattern compiled once already, so it
        // compiles again.
        let re = match Self::new(QWEN35_PATTERN) {
            Ok(p) => p.re,
            Err(_) => Self::new(r".").expect("fallback pattern").re,
        };
        Self { re }
    }
}

impl PreTokenizer {
    /// Compiles the `qwen35` pattern (or another named pattern, for future
    /// pre-tokenizers).
    pub fn qwen35() -> Result<Self, fancy_regex::Error> {
        Self::new(QWEN35_PATTERN)
    }

    pub fn new(pattern: &str) -> Result<Self, fancy_regex::Error> {
        Ok(Self {
            re: Regex::new(pattern)?,
        })
    }

    /// The pattern's pieces over `text`, in order. The concatenation of the
    /// pieces is `text` itself (the pattern is total over any input).
    pub fn split<'t>(&self, text: &'t str) -> Vec<&'t str> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match self.re.find(rest) {
                Ok(Some(m)) if m.start() == 0 => {
                    out.push(m.as_str());
                    rest = &rest[m.end()..];
                }
                Ok(Some(m)) => {
                    // A gap before the match (should not happen with a total
                    // pattern, but stay safe): emit it as its own piece.
                    out.push(&rest[..m.start()]);
                    out.push(m.as_str());
                    rest = &rest[m.end()..];
                }
                _ => {
                    out.push(rest);
                    break;
                }
            }
        }
        out
    }
}
