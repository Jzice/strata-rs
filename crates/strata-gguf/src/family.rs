//! The model families, from the GGUF's `general.architecture` string.

use std::fmt;

/// Which model family a GGUF holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    #[default]
    Unknown,
    /// Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5,
    /// the Coder): the hybrid GDN + gated-attention engine path.
    Qwen4Exp,
    /// Qwen3.6-35B-A3B (Unsloth's MTP GGUFs): plain pre-norm residual, dense
    /// attention, nextn MTP block.
    Qwen35Moe,
    /// Qwen3.5-9B (and fine-tunes): the dense GDN + gated-attention hybrid with
    /// a plain pre-norm residual and a dense SwiGLU FFN (no MoE, no MTP, no
    /// hyper-connections).
    Qwen35,
}

impl ModelFamily {
    pub fn from_arch(arch: &str) -> Self {
        match arch {
            "qwen4exp" => ModelFamily::Qwen4Exp,
            "qwen35moe" => ModelFamily::Qwen35Moe,
            "qwen35" => ModelFamily::Qwen35,
            _ => ModelFamily::Unknown,
        }
    }

    /// The GGUF architecture string (also the metadata key prefix).
    pub fn arch(self) -> &'static str {
        match self {
            ModelFamily::Qwen4Exp => "qwen4exp",
            ModelFamily::Qwen35Moe => "qwen35moe",
            ModelFamily::Qwen35 => "qwen35",
            ModelFamily::Unknown => "",
        }
    }

    /// The model's name for messages.
    pub fn title(self) -> &'static str {
        match self {
            ModelFamily::Qwen4Exp => "Qwen3.8-Flash-Next",
            ModelFamily::Qwen35Moe => "Qwen3.6-35B-A3B",
            ModelFamily::Qwen35 => "Qwen3.5 (dense GDN+QSA)",
            ModelFamily::Unknown => "an unknown model",
        }
    }

    /// Whether this replica's engine runs the family.
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            ModelFamily::Qwen4Exp | ModelFamily::Qwen35Moe | ModelFamily::Qwen35
        )
    }

    /// The refusal for a family this build does not run.
    pub fn not_supported_message(self) -> String {
        if self.is_supported() {
            return String::new();
        }
        if self == ModelFamily::Unknown {
            return "this is not a model family Strata knows".to_string();
        }
        format!(
            "{} (architecture '{}') is not supported by this build",
            self.title(),
            self.arch()
        )
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.arch())
    }
}
