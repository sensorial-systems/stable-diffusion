//! Learning rate scheduler module.

use crate::prelude::*;
use std::fmt::Display;

fn default_amount() -> f32 { 0.0001 }
fn default_text_encoder_lr() -> f32 { 5e-05 }
fn default_unet_lr() -> f32 { 0.0001 }
fn default_lr_scheduler_num_cycles() -> usize { 1 }
fn default_lr_warmup_steps() -> usize { 48 }

/// The learning rate scheduler structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct LearningRate {
    /// The amount of the learning rate.
    #[serde(default = "default_amount")]
    pub amount: f32,
    /// The learning rate scheduler.
    pub scheduler: LearningRateScheduler,
    /// The learning rate for the text encoder.
    #[serde(default = "default_text_encoder_lr")]
    pub text_encoder: f32,
    /// The learning rate for the unet.
    #[serde(default = "default_unet_lr")]
    pub unet: f32,
    /// The number of cycles for the learning rate scheduler.
    #[serde(default = "default_lr_scheduler_num_cycles")]
    pub scheduler_num_cycles: usize,
    /// The number of warmup steps for the learning rate.
    #[serde(default = "default_lr_warmup_steps")]
    pub warmup_steps: usize
}

impl Default for LearningRate {
    fn default() -> Self {
        let amount = default_amount();
        let scheduler = Default::default();
        let text_encoder_lr = default_text_encoder_lr();
        let unet_lr = default_unet_lr();
        let lr_scheduler_num_cycles = default_lr_scheduler_num_cycles();
        let lr_warmup_steps = default_lr_warmup_steps();
        LearningRate { amount, scheduler, text_encoder: text_encoder_lr, unet: unet_lr, scheduler_num_cycles: lr_scheduler_num_cycles, warmup_steps: lr_warmup_steps }
    }
}

impl LearningRate {
    /// Create a new learning rate structure.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the amount of the learning rate.
    pub fn with_amount(mut self, amount: f32) -> Self {
        self.amount = amount.into();
        self
    }

    /// Set the learning rate scheduler.
    pub fn with_scheduler(mut self, scheduler: LearningRateScheduler) -> Self {
        self.scheduler = scheduler.into();
        self
    }
}

/// The learning rate scheduler enumeration.
#[derive(Debug, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    /// Adafactor learning rate scheduler.
    Adafactor,
    /// Constant learning rate scheduler.
    Constant,
    /// Constant with warmup learning rate scheduler.
    ConstantWithWarmup,
    /// Cosine learning rate scheduler.
    Cosine,
    /// Cosine with restarts learning rate scheduler.
    CosineWithRestarts,
    /// Linear learning rate scheduler.
    Linear,
    /// Polynomial learning rate scheduler.
    Polynomial
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        LearningRateScheduler::Cosine
    }
}

impl Display for LearningRateScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearningRateScheduler::Adafactor => write!(f, "adafactor"),
            LearningRateScheduler::Constant => write!(f, "constant"),
            LearningRateScheduler::ConstantWithWarmup => write!(f, "constant_with_warmup"),
            LearningRateScheduler::Cosine => write!(f, "cosine"),
            LearningRateScheduler::CosineWithRestarts => write!(f, "cosine_with_restarts"),
            LearningRateScheduler::Linear => write!(f, "linear"),
            LearningRateScheduler::Polynomial => write!(f, "polynomial")
        }
    }
}
