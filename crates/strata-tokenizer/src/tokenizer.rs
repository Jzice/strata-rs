//! The BPE tokenizer, whose vocabulary, merges and special tokens are read
//! from the GGUF file itself (`tokenizer.ggml.*` metadata keys).
//!
//! The decisive property is the round trip: byte-level BPE maps each BYTE to
//! a printable unicode character so any UTF-8 input is representable with no
//! UNK token. Every failure mode of that mapping - the wrong offset for a
//! byte, a merge applied in the wrong order, a pre-tokenizer split that drops
//! a character - still produces plausible token ids. `decode(encode(s)) == s`
//! is the check that sees them, and it is exact because the byte layer is
//! lossless.

use crate::bpe::Merges;
use crate::byte_level::{decode_token, encode_bytes};
use crate::error::{Error, Result};
use crate::pretok::PreTokenizer;
use std::collections::HashMap;

/// A token's role in the vocabulary (the GGUF `tokenizer.ggml.token_type`
/// values this tokenizer distinguishes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => return None,
        })
    }
}

/// How many pre-tokenizer pieces the encoder remembers.
const PIECE_CACHE_MAX: usize = 200_000;

/// The tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    merges: Merges,
    token_types: Vec<TokenType>,
    pre: String,
    pre_tok: PreTokenizer,
    /// Special literals matched as single tokens: `token -> id`.
    special_tokens: HashMap<String, u32>,
    /// The type-4 (USER_DEFINED) literals, matched with or without
    /// `parse_special`.
    always: Vec<String>,
    /// All special literals, matched only with `parse_special` (type 3).
    all_special: Vec<String>,
    /// Remembered pieces: `piece -> ids`.
    piece_cache: HashMap<String, Vec<u32>>,
}

impl Tokenizer {
    /// Builds the tokenizer from the GGUF's metadata. Returns an error when
    /// the file does not carry a byte-level BPE tokenizer.
    pub fn from_gguf(file: &strata_gguf::GgufFile) -> Result<Self> {
        let get_arr = |key: &str| -> Result<Vec<String>> {
            let v = file.get(key).ok_or(Error::MissingVocabEntry { index: 0 })?;
            if v.ty != strata_gguf::metadata::MetaType::Array {
                return Err(Error::EmptyVocab);
            }
            Ok(v.items
                .iter()
                .map(|item| item.as_str().unwrap_or_default().to_string())
                .collect())
        };

        let tokens = get_arr("tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(Error::EmptyVocab);
        }
        let merges_list = get_arr("tokenizer.ggml.merges")?;
        let model = file
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2")
            .to_string();
        if model != "gpt2" {
            return Err(Error::TemplateRender {
                message: format!("expected a byte-level BPE tokenizer (model 'gpt2'), got '{model}'"),
            });
        }
        let token_types = file
            .get("tokenizer.ggml.token_type")
            .map(|v| {
                v.items
                    .iter()
                    .map(|item| {
                        TokenType::from_id(item.as_int().unwrap_or(1) as u32).unwrap_or(TokenType::Normal)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let pre = file
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen35")
            .to_string();

        Self::new(tokens, &merges_list, token_types, pre)
    }

    /// Builds the tokenizer from its parts. `pre` names the pre-tokenizer
    /// (`qwen35` today; an unknown name falls back to it with a warning).
    pub fn new(
        tokens: Vec<String>,
        merges_list: &[String],
        token_types: Vec<TokenType>,
        pre: String,
    ) -> Result<Self> {
        let mut ids: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            if ids.insert(t.clone(), i as u32).is_some() {
                return Err(Error::TemplateRender {
                    message: format!(
                        "vocabulary has duplicate tokens: entry {i} ({t:?}) repeats an earlier one"
                    ),
                });
            }
        }
        let merges = Merges::from_pairs(merges_list);
        // A merge naming a token outside the vocabulary would make BPE
        // silently produce different ids than the model was trained with.
        for pair in merges_list {
            let (left, right) = pair.split_once(' ').ok_or_else(|| Error::TemplateRender {
                message: format!("merge '{pair}' is not a pair"),
            })?;
            if !ids.contains_key(left) || !ids.contains_key(right) {
                return Err(Error::TemplateRender {
                    message: format!("merge '{pair}' names a token outside the vocabulary"),
                });
            }
        }

        // The literals matched directly instead of being run through BPE.
        // GGUF token types: 3 = CONTROL, 4 = USER_DEFINED. The two classes
        // behave DIFFERENTLY: type-4 is matched always, with or without
        // parse_special; type-3 only when parse_special is set.
        let mut special_tokens = HashMap::new();
        for (i, ty) in token_types.iter().enumerate() {
            if matches!(ty, TokenType::Control | TokenType::UserDefined) {
                special_tokens.insert(tokens[i].clone(), i as u32);
            }
        }
        let mut always: Vec<String> = token_types
            .iter()
            .enumerate()
            .filter(|(_, ty)| **ty == TokenType::UserDefined)
            .map(|(i, _)| tokens[i].clone())
            .collect();
        let mut all_special: Vec<String> = special_tokens.keys().cloned().collect();
        // Longest literal first, or `` could match a shorter
        // prefix of itself.
        always.sort_by_key(|s| std::cmp::Reverse(s.len()));
        all_special.sort_by_key(|s| std::cmp::Reverse(s.len()));

        let pre_tok = match pre.as_str() {
            "qwen35" | "" => PreTokenizer::qwen35().map_err(|e| Error::TemplateRender {
                message: format!("pre-tokenizer pattern failed to compile: {e}"),
            })?,
            other => {
                tracing::warn!("unknown pre-tokenizer '{other}', using qwen35");
                PreTokenizer::qwen35().map_err(|e| Error::TemplateRender {
                    message: format!("pre-tokenizer pattern failed to compile: {e}"),
                })?
            }
        };

        Ok(Self {
            tokens,
            ids,
            merges,
            token_types,
            pre,
            pre_tok,
            special_tokens,
            always,
            all_special,
            piece_cache: HashMap::new(),
        })
    }

    /// The vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The token at `id`, or `None` outside the vocabulary.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// The id of an exact token string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// The special literals and their ids (`token -> id`).
    pub fn special_ids(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// The pre-tokenizer's name (`qwen35` for both model families).
    pub fn pre(&self) -> &str {
        &self.pre
    }

    /// The token's type, when the vocabulary declares one.
    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.token_types.get(id as usize).copied()
    }

    /// Tokenizes `text`.
    ///
    /// `parse_special` controls only the type-3 CONTROL literals such as
    /// ``; the type-4 USER_DEFINED ones such as `<think>` are
    /// matched either way.
    pub fn encode(&mut self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let literals: &[String] = if parse_special {
            &self.all_special
        } else {
            &self.always
        };
        if literals.is_empty() {
            return self.encode_plain(text);
        }
        // Owned (literal, id, length) triples, longest first, so the scan
        // below borrows nothing from `self` and `encode_plain` can take
        // `&mut self`.
        let specials: Vec<(String, u32, usize)> = literals
            .iter()
            .map(|lit| (lit.clone(), self.special_tokens[lit], lit.len()))
            .collect();
        let mut out = Vec::new();
        let mut pos = 0;
        // The earliest literal match wins; on a tie the longest literal,
        // exactly as the reference's longest-first alternation does.
        while pos < text.len() {
            let mut best: Option<(usize, u32, usize)> = None;
            for (lit, id, len) in &specials {
                if let Some(at) = text[pos..].find(lit) {
                    let start = pos + at;
                    match best {
                        Some((bs, _, _)) if bs <= start => {}
                        _ => best = Some((start, *id, *len)),
                    }
                }
            }
            if let Some((start, id, len)) = best {
                if start > pos {
                    out.extend(self.encode_plain(&text[pos..start])?);
                }
                out.push(id);
                pos = start + len;
            } else {
                out.extend(self.encode_plain(&text[pos..])?);
                break;
            }
        }
        Ok(out)
    }

    /// The ordinary path: split with the pre-tokenizer, byte-map each piece,
    /// BPE it, and look the pieces up.
    fn encode_plain(&mut self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for piece in self.pre_tok.split(text) {
            let cached = self.piece_cache.get(piece).cloned();
            let ids = if let Some(ids) = cached {
                ids
            } else {
                let mapped = encode_bytes(piece.as_bytes());
                let mut ids = Vec::new();
                for tok in self.merges.apply(&mapped) {
                    let id = self
                        .ids
                        .get(&tok)
                        .copied()
                        .ok_or(Error::MissingVocabEntry { index: usize::MAX })?; // no single id names the failure
                    ids.push(id);
                }
                if self.piece_cache.len() < PIECE_CACHE_MAX {
                    self.piece_cache.insert(piece.to_string(), ids.clone());
                }
                ids
            };
            out.extend(ids);
        }
        Ok(out)
    }

    /// The raw bytes of one token (a multi-byte character can be split
    /// across tokens).
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let token = self
            .tokens
            .get(id as usize)
            .ok_or(Error::MissingVocabEntry { index: id as usize })?;
        decode_token(token).ok_or(Error::NotUtf8)
    }

    /// Decodes ids back into text. Invalid UTF-8 is replaced lossily, as the
    /// reference does.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.token_bytes(id) {
                Ok(b) => bytes.extend_from_slice(&b),
                Err(_) => bytes.extend_from_slice(&[0xEF, 0xBF, 0xBD]), // U+FFFD
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
