//! Module for model file format enum.

use std::fmt::Display;

/// The model file format enumeration.
pub enum ModelFileFormat {
    /// The safetensors format.
    Safetensors
}

impl Display for ModelFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFileFormat::Safetensors => write!(f, "safetensors")
        }
    }
}