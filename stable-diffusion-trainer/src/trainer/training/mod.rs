//! The training configuration for the training process.

pub mod output;
pub mod optimizer;
pub mod scheduler;
pub mod bucketing;
pub mod network;
pub mod prompt;

use std::collections::HashMap;

pub use network::*;
pub use prompt::*;
pub use bucketing::*;
pub use output::*;
pub use optimizer::*;
pub use scheduler::*;

use crate::{prelude::*, utils::{ReferenceResolver, Update, Variable}};

fn default_pretrained_model() -> String { "stabilityai/stable-diffusion-xl-base-1.0".to_string() }
fn default_batch_size() -> Variable<usize> { 1.into() }

/// The training configuration for the training process.
#[derive(Debug, Serialize, Deserialize)]
pub struct Training {
    /// The prompt to use for the training process.
    pub prompt: Prompt,
    /// The output to use for the training process.
    pub output: Output,
    /// The batch size to use for the training process.
    #[serde(default = "default_batch_size")]
    pub batch_size: Variable<usize>,
    /// The name or path of the pretrained model to use for the training process.
    #[serde(default = "default_pretrained_model")]
    pub pretrained_model: String,
    /// The optimizer to use for the training process.
    pub optimizer: Variable<Optimizer>,
    /// The learning rate to use for the training process.
    pub learning_rate: LearningRate,
    /// The network to use for the training process.
    pub network: Network,
    /// Bucketing.
    pub bucketing: Option<Bucketing>
}

impl Training {
    /// Create a new training configuration.
    pub fn new(prompt: Prompt, output: Output) -> Self {
        let optimizer = Default::default();
        let learning_rate = Default::default();
        let pretrained_model = default_pretrained_model();
        let batch_size = default_batch_size();
        let network = Default::default();
        let bucketing = Default::default();
        Training { prompt, optimizer, learning_rate, pretrained_model, batch_size, network, output, bucketing }
    }

    /// Set the network configuration to use for the training process.
    pub fn with_network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// Set the pretrained model for the training process.
    pub fn with_pretrained_model(mut self, pretrained_model: impl Into<String>) -> Self {
        self.pretrained_model = pretrained_model.into();
        self
    }

    /// Set the optimizer for the training process.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer.into();
        self
    }

    /// Set the learning rate for the training process.
    pub fn with_learning_rate(mut self, learning_rate: LearningRate) -> Self {
        self.learning_rate = learning_rate.into();
        self
    }
}

impl ReferenceResolver for Training {
    fn resolve_references(&mut self, variables: &HashMap<String, serde_json::Value>) {
        self.prompt.resolve_references(variables);
        self.output.resolve_references(variables);
        self.optimizer.resolve_references(variables);
        self.learning_rate.resolve_references(variables);
        self.network.resolve_references(variables);
        self.bucketing.resolve_references(variables);
    }

}

impl Update for Training {
    fn update(&mut self, _base: Self) {
        // self.prompt.update(base.prompt);
        // self.output.update(base.output);
        // self.optimizer.update(base.optimizer);
        // self.learning_rate.update(base.learning_rate);
        // self.network.update(base.network);
        // self.bucketing.update(base.bucketing);
    }
}