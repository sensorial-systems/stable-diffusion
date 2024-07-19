//! Module for model file format enum.

use crate::prelude::*;

use std::fmt::Display;

/// The model file format enumeration.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ModelFileFormat {
    /// The safetensors format.
    Safetensors
}

impl Default for ModelFileFormat {
    fn default() -> Self {
        ModelFileFormat::Safetensors
    }
}

impl Display for ModelFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFileFormat::Safetensors => write!(f, "safetensors")
        }
    }
}