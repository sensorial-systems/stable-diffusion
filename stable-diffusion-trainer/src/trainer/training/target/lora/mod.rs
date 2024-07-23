//! LoRA module.

use crate::prelude::*;
use crate::Network;
use crate::Training;

/// The LoRA structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LoRA {
    /// The network to use for the training process.
    pub network: Network
}

impl LoRA {
    /// Get the network module.
    pub fn network_module(&self) -> &str {
        "networks.lora"
    }

    /// Set parameters.
    pub fn set_parameters(&self, command: &mut std::process::Command, training: &Training) {
        command
            .args(["--network_module", &self.network_module()])
            .args(["--network_dim", &self.network.dimension.to_string()])
            .args(["--network_alpha", &self.network.alpha.to_string()])
            .args(["--text_encoder_lr", &training.learning_rate.text_encoder.to_string()])
            .args(["--unet_lr", &training.learning_rate.unet.to_string()])
            // .args(["--lr_warmup_steps", &training.lr_warmup_steps.to_string()])
            ;

    }
}