//! Training kind module.

pub mod lora;

pub use lora::*;

use crate::{prelude::*, Training};

/// The training kind.
#[derive(Debug, Serialize, Deserialize)]
pub enum Target {
    /// Checkpoint training kind.
    Checkpoint,
    /// LoRA training kind.
    LoRA(LoRA),
    /// ControlNet training kind.
    ControlNet
}

impl Default for Target {
    fn default() -> Self {
        Self::LoRA(LoRA::default())
    }
}

impl Target {
    /// Set parameters.
    pub fn set_parameters(&self, command: &mut std::process::Command, training: &Training) {
        match self {
            Self::Checkpoint => {},
            Self::LoRA(lora) => lora.set_parameters(command, training),
            Self::ControlNet => {}
        }
    }
}