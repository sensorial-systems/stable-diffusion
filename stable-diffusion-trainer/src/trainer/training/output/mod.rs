//! Trainer's output.

use std::path::PathBuf;

use crate::{prelude::*, FloatPrecision, ModelFileFormat};

fn default_save_every_n_epochs() -> usize { 1 }

/// The output structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    /// The name of the output.
    pub name: String,
    /// The directory to save the output to.
    pub directory: PathBuf,
    /// The format to save the model as.
    #[serde(default)]
    pub save_model_as: ModelFileFormat,
    /// The save precision.
    #[serde(default)]
    pub save_precision: FloatPrecision,
    /// The frequency to save the model.
    #[serde(default = "default_save_every_n_epochs")]
    pub save_every_n_epochs: usize    
}

impl Output {
    /// Create a new output structure.
    pub fn new(name: impl Into<String>, directory: impl Into<PathBuf>) -> Self {
        let name = name.into();
        let directory = directory.into();
        let save_model_as = Default::default();
        let save_precision = Default::default();
        let save_every_n_epochs = default_save_every_n_epochs();
        Output { name, directory, save_model_as, save_precision, save_every_n_epochs }
    }
}
