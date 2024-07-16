//! Precision module contains the definition of the precision of the model.

use crate::prelude::*;

/// The precision enumeration.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FloatPrecision {
    /// 16-bit floating point precision.
    F16,
    /// 16-bit brain floating point precision.
    BF16,
    /// 32-bit floating point precision.
    F32
}

impl Default for FloatPrecision {
    fn default() -> Self {
        FloatPrecision::F16
    }
}

impl std::fmt::Display for FloatPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatPrecision::F16 => write!(f, "fp16"),
            FloatPrecision::BF16 => write!(f, "bf16"),
            FloatPrecision::F32 => write!(f, "fp32")
        }
    }
}