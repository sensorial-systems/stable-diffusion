//! The training configuration for the training process.

pub mod output;
pub mod optimizer;
pub mod scheduler;
pub mod bucketing;
pub mod network;
pub mod prompt;
pub mod model;

pub use network::*;
pub use prompt::*;
pub use bucketing::*;
pub use output::*;
pub use optimizer::*;
pub use scheduler::*;
pub use model::*;

use crate::{prelude::*, FloatPrecision};

fn default_batch_size() -> usize { 1 }
fn default_images_repeat() -> usize { 40 }
fn default_regularization_images_repeat() -> usize { 1 }
fn default_network_module() -> String { "networks.lora".to_string() }
fn default_max_train_steps() -> usize { 1600 }
fn default_max_grad_norm() -> f32 { 1.0 }
fn default_max_data_loader_n_workers() -> usize { 0 }
fn default_noise_offset() -> f32 { 0.0 }

/// The training configuration for the training process.
#[derive(Debug, Serialize, Deserialize)]
pub struct Training {
    /// The prompt to use for the training process.
    pub prompt: Prompt,
    /// The output to use for the training process.
    pub output: Output,
    /// The batch size to use for the training process.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// The optimizer to use for the training process.
    pub optimizer: Optimizer,
    /// The learning rate to use for the training process.
    pub learning_rate: LearningRate,
    /// The network to use for the training process.
    pub network: Network,
    /// The model to be trained.
    pub model: Model,
    /// Bucketing.
    pub bucketing: Option<Bucketing>,
    /// Images repeat.
    #[serde(default = "default_images_repeat")]
    pub images_repeat: usize,
    /// Regularization images repeat.
    #[serde(default = "default_regularization_images_repeat")]
    pub regularization_images_repeat: usize,
    /// Training resolution. It will use the model's default resolution if not set.
    pub resolution: Option<(usize, usize)>,
    /// Mixed precision.
    #[serde(default)]
    pub mixed_precision: FloatPrecision,
    /// The module to use for the network.
    #[serde(default = "default_network_module")]
    pub network_module: String,
    /// The maximum number of training steps.
    #[serde(default = "default_max_train_steps")]
    pub max_train_steps: usize,
    /// The maximum gradient norm.
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    /// The maximum number of data loader workers.
    #[serde(default = "default_max_data_loader_n_workers")]
    pub max_data_loader_n_workers: usize,
    /// The noise offset.
    #[serde(default = "default_noise_offset")]
    pub noise_offset: f32,    
}

impl Training {
    /// Create a new training configuration.
    pub fn new(prompt: Prompt, output: Output) -> Self {
        let optimizer = Default::default();
        let learning_rate = Default::default();
        let model = Default::default();
        let batch_size = default_batch_size();
        let network = Default::default();
        let bucketing = Default::default();
        let images_repeat = default_images_repeat();
        let regularization_images_repeat = default_regularization_images_repeat();
        let mixed_precision = Default::default();
        let network_module = default_network_module();
        let max_train_steps = default_max_train_steps();
        let max_grad_norm = default_max_grad_norm();
        let max_data_loader_n_workers = default_max_data_loader_n_workers();
        let noise_offset = default_noise_offset();
        let resolution = Default::default();
        Training { prompt, output, batch_size, model, optimizer, network, bucketing, images_repeat, regularization_images_repeat, resolution, mixed_precision, network_module,  learning_rate, max_train_steps, max_grad_norm, max_data_loader_n_workers, noise_offset }
    }

    /// Set the network configuration to use for the training process.
    pub fn with_network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// Set the model for the training process.
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
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
