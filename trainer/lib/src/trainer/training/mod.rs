//! The training configuration for the training process.

use crate::{prelude::*, LearningRate, Optimizer};

/// The training configuration for the training process.
#[derive(Debug, Serialize, Deserialize)]
pub struct Training {
    /// The optimizer to use for the training process.
    pub optimizer: Optimizer,
    /// The learning rate to use for the training process.
    pub learning_rate: LearningRate
}

impl Default for Training {
    fn default() -> Self {
        let optimizer = Optimizer::Adafactor;
        let learning_rate = LearningRate::default();
        Training { optimizer, learning_rate }
    }
}

impl Training {
    /// Create a new training configuration.
    pub fn new() -> Self {
        Default::default()
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