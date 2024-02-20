//! The training configuration for the training process.

use crate::{prelude::*, LearningRate, Optimizer};

fn default_pretrained_model() -> String {
    "stabilityai/stable-diffusion-xl-base-1.0".to_string()
}

/// The training configuration for the training process.
#[derive(Debug, Serialize, Deserialize)]
pub struct Training {
    /// The name or path of the pretrained model to use for the training process.
    #[serde(default = "default_pretrained_model")]
    pub pretrained_model: String,
    /// The optimizer to use for the training process.
    pub optimizer: Optimizer,
    /// The learning rate to use for the training process.
    pub learning_rate: LearningRate
}

impl Default for Training {
    fn default() -> Self {
        let optimizer = Optimizer::Adafactor;
        let learning_rate = LearningRate::default();
        let pretrained_model = default_pretrained_model();
        Training { optimizer, learning_rate, pretrained_model }
    }
}

impl Training {
    /// Create a new training configuration.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the pretrained model for the training process.
    pub fn with_pretrained_model(mut self, pretrained_model: impl Into<String>) -> Self {
        self.pretrained_model = pretrained_model.into();
        self
    }

    /// Set the optimizer for the training process.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the learning rate for the training process.
    pub fn with_learning_rate(mut self, learning_rate: LearningRate) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}