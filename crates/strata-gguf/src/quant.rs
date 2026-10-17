//! ggml quantization types, using **Strata's type-id scheme** (which differs
//! from upstream ggml: Strata's `Q2_0` is id 42, a 64-element ternary block;
//! `TQ1_0`/`TQ2_0` are 34/35, `MXFP4`/`NVFP4` are 39/40, `Q1_0` is 41).
//!
//! Block sizes and byte sizes follow Strata's `gguf_reader.hpp`
//! (`block_geometry`) and the block structs in its vendored
//! `third_party/ggml/ggml-common.h`.

use crate::error::{Error, Result};
use std::fmt;

/// A ggml quantization type (or plain element type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    Iq1M = 29,
    Bf16 = 30,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl QuantType {
    /// Builds a `QuantType` from its Strata ggml id.
    pub fn from_id(id: u32) -> Result<Self> {
        use QuantType::*;
        Ok(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            29 => Iq1M,
            30 => Bf16,
            39 => Mxfp4,
            40 => Nvfp4,
            41 => Q1_0,
            42 => Q2_0,
            _ => {
                return Err(Error::UnsupportedQuant {
                    name: format!("type id {id}"),
                })
            }
        })
    }

    /// The Strata ggml id.
    pub fn id(self) -> u32 {
        self as u32
    }

    /// The type name as it appears in GGUF files and Strata's tools.
    pub fn name(self) -> &'static str {
        use QuantType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Iq2Xxs => "IQ2_XXS",
            Iq2Xs => "IQ2_XS",
            Iq3Xxs => "IQ3_XXS",
            Iq4Nl => "IQ4_NL",
            Iq3S => "IQ3_S",
            Iq2S => "IQ2_S",
            Iq4Xs => "IQ4_XS",
            I8 => "I8",
            Iq1M => "IQ1_M",
            Bf16 => "BF16",
            Mxfp4 => "MXFP4",
            Nvfp4 => "NVFP4",
            Q1_0 => "Q1_0",
            Q2_0 => "Q2_0",
        }
    }

    /// Elements per quantization block.
    pub fn block_size(self) -> usize {
        use QuantType::*;
        match self {
            F32 | F16 | Bf16 | I8 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl | Mxfp4 => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Iq2Xxs | Iq2Xs | Iq3Xxs | Iq3S | Iq2S | Iq4Xs | Iq1M => 256,
            Nvfp4 => 64,
            Q1_0 => 128, // QK1_0 = 128: 1 bit per element + fp16 scale
            Q2_0 => 64,  // QK2_0 = 64: 2 ternary bits per element + fp16 scale
        }
    }

    /// Bytes per quantization block.
    pub fn type_size(self) -> usize {
        use QuantType::*;
        match self {
            F32 => 4,
            F16 | Bf16 => 2,
            I8 => 1,
            Q4_0 | Iq4Nl | Q2_0 | Q1_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36, // fp16 d, fp16 s, 32 int8
            Q2K => 84,
            Q3K => 110,
            Q4K => 144,
            Q5K => 176,
            Q6K => 210,
            Iq2Xxs => 66,
            Iq2Xs => 74,
            Iq3Xxs => 98,
            Iq1M => 56,
            Iq2S => 82,
            Iq3S => 110,
            Iq4Xs => 136,
            Mxfp4 => 17, // E8M0 scale + 16 bytes of 4-bit values
            Nvfp4 => 36, // 4 UE4M3 scales + 32 bytes of E2M1 codes
        }
    }

    /// True for the plain (non-quantized) float types.
    pub fn is_float(self) -> bool {
        matches!(self, QuantType::F32 | QuantType::F16 | QuantType::Bf16)
    }

    /// Number of ggml blocks needed to store `n_elements`.
    pub fn n_blocks(self, n_elements: usize) -> usize {
        n_elements.div_ceil(self.block_size())
    }

    /// Number of bytes needed to store `n_elements`.
    pub fn byte_size(self, n_elements: usize) -> usize {
        self.n_blocks(n_elements) * self.type_size()
    }
}

impl fmt::Display for QuantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
