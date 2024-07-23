//! Model module.

pub mod checkpoint;
use crate::{prelude::*, Target, Training};
pub use checkpoint::*;

/// The Model structure.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "name", content = "checkpoint")]
pub enum Model {
    /// The Stable Diffusion XL model.
    #[serde(rename = "Stable Diffusion XL")]
    StableDiffusionXL(Option<Checkpoint>),
    /// The Stable Diffusion 1.5 model.
    #[serde(rename = "Stable Diffusion 1.5")]
    StableDiffusion1_5(Option<Checkpoint>)
 }

impl Model {
    /// Get the name of the model.
    pub fn name(&self) -> &str {
        match self {
            Self::StableDiffusionXL(_) => "Stable Diffusion XL",
            Self::StableDiffusion1_5(_) => "Stable Diffusion 1.5"
        }
    }

    /// Get the resolution of the model.
    pub fn resolution(&self) -> (usize, usize) {
        match self {
            Self::StableDiffusionXL(_) => (1024, 1024),
            Self::StableDiffusion1_5(_) => (512, 512)
        }
    }

    /// Get the checkpoint of the model.
    pub fn checkpoint(&self) -> Checkpoint {
        match self {
            Self::StableDiffusionXL(checkpoint) => checkpoint.clone().unwrap_or("stabilityai/stable-diffusion-xl-base-1.0".into()),
            Self::StableDiffusion1_5(checkpoint) => checkpoint.clone().unwrap_or("runwayml/stable-diffusion-v1-5".into())
        }
    }

    /// Get the training script of the model.
    pub fn training_script(&self, training: &Training) -> &str {
        match (self, &training.target) {
            (Self::StableDiffusion1_5(_), Target::LoRA(_)) => "train_network.py",
            (Self::StableDiffusion1_5(_), Target::Checkpoint) => "train_db.py",
            (Self::StableDiffusionXL(_), Target::LoRA(_)) => "sdxl_train_network.py",
            (Self::StableDiffusionXL(_), Target::Checkpoint) => "sdxl_train.py",
            _ => todo!("Training script not implemented for the model and target.")
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::StableDiffusionXL(None)
    }
}