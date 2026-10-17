//! GGUF metadata value types (the KV section of the header).
//!
//! Mirrors the reference reader in Strata's `gguf_reader.hpp`: every value
//! type is parsed, arrays keep their element type and count, and integer
//! payloads are normalised into `i64`/`u64` regardless of width.

use crate::error::{Error, Result};
use std::fmt;

/// The GGUF metadata value type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetaType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetaType {
    pub fn from_id(id: u32) -> Result<Self> {
        use MetaType::*;
        Ok(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return Err(Error::BadValueType { value_type: id }),
        })
    }
}

/// One metadata value. Scalars keep their payload in `int`/`float`; strings in
/// `string`; arrays in `items` with the element type in `elem`.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaValue {
    pub ty: MetaType,
    /// Integer payloads (unsigned widths stored zero-extended).
    pub int: i64,
    /// Unsigned payload for the wide types (u64; i64 keys store the bit pattern).
    pub uint: u64,
    /// Float payloads (f32 widened to f64).
    pub float: f64,
    /// String payload.
    pub string: String,
    /// Array element type.
    pub elem: MetaType,
    /// Array length (may exceed `items.len()` when items were capped).
    pub count: u64,
    /// Array items (capped at [`MAX_ARRAY_ITEMS_KEPT`]).
    pub items: Vec<MetaValue>,
}

/// Arrays keep at most this many items; longer ones keep the count only.
pub const MAX_ARRAY_ITEMS_KEPT: usize = 64;

impl MetaValue {
    /// True for the non-string, non-array types.
    pub fn is_num(&self) -> bool {
        !matches!(self.ty, MetaType::String | MetaType::Array)
    }

    /// The value as f64 (integers widen exactly).
    pub fn num(&self) -> f64 {
        match self.ty {
            MetaType::F32 | MetaType::F64 => self.float,
            _ => self.int as f64,
        }
    }

    /// The value as i64 when it is an integer type.
    pub fn as_int(&self) -> Option<i64> {
        if self.is_num() && !matches!(self.ty, MetaType::F32 | MetaType::F64) {
            Some(self.int)
        } else {
            None
        }
    }

    /// The value as f64 when it is a number.
    pub fn as_float(&self) -> Option<f64> {
        if self.is_num() {
            Some(self.num())
        } else {
            None
        }
    }

    /// The value as a string when it is one.
    pub fn as_str(&self) -> Option<&str> {
        (self.ty == MetaType::String).then_some(self.string.as_str())
    }
}

impl fmt::Display for MetaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            MetaType::String => write!(f, "{:?}", self.string),
            MetaType::Array => write!(f, "[{} x {:?}]", self.count, self.elem),
            MetaType::F32 | MetaType::F64 => write!(f, "{}", self.float),
            MetaType::Bool => write!(f, "{}", self.int != 0),
            _ => write!(f, "{}", self.int),
        }
    }
}
