//! Trainer's parameters.

use crate::prelude::*;
use crate::{Network, Output, Prompt, Training, TrainingDataSet};

/// The parameters structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Parameters {
    /// The prompt to use for the training process.
    pub prompt: Prompt,
    /// The dataset to use for the training process.
    pub dataset: TrainingDataSet,
    /// The output to use for the training process.
    pub output: Output,
    /// The network to use for the training process.
    pub network: Network,
    /// The training to use for the training process.
    pub training: Training
}

impl Parameters {
    /// Get the parameters from a file.
    pub fn from_file(path: impl Into<std::path::PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        use path_slash::*;

        let path = path.into().canonicalize()?;
        let parent = path.parent().unwrap();
        let file = std::fs::File::open(path.clone())?;
        let reader = std::io::BufReader::new(file);
        let mut parameters: Parameters = serde_json::from_reader(reader)?;

        // TODO: Simplify this. Wrap it in a function.
        if parameters.dataset.training.path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(parameters.dataset.training.path());
            parameters.dataset.training.set_path(path);
        }
        if parameters.dataset.regularization.is_some() && parameters.dataset.regularization.as_ref().unwrap().path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(parameters.dataset.regularization.as_ref().unwrap().path());
            parameters.dataset.regularization.as_mut().unwrap().set_path(path);
        }
        if parameters.output.directory.is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(parameters.output.directory);
            parameters.output.directory = path;
        }
        parameters.output.name = parameters.output.name
            .replace("{network.dimension}", &parameters.network.dimension.to_string())
            .replace("{network.alpha}", &parameters.network.alpha.to_string())
            .replace("{prompt.instance}", &parameters.prompt.instance)
            .replace("{prompt.class}", &parameters.prompt.class);

        Ok(parameters)
    }

    /// Create a new parameters structure.
    pub fn new(prompt: Prompt, dataset: TrainingDataSet, output: Output) -> Self {
        let network = Default::default();
        let training = Default::default();
        Parameters { prompt, dataset, output, network, training }
    }

    /// Set the network configuration to use for the training process.
    pub fn with_network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// Set the training configuration to use for the training process.
    pub fn with_training(mut self, training: Training) -> Self {
        self.training = training;
        self
    }
}