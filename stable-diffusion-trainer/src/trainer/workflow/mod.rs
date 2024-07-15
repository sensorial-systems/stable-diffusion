//! Trainer's workflow.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::prelude::*;
use crate::{Captioning, Training, TrainingDataSet};

/// The workflow structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Workflow {
    /// The captioning parameters.
    pub captioning: Option<Captioning>,
    /// The dataset to use for the training process.
    pub dataset: TrainingDataSet,
    /// The training to use for the training process.
    pub training: Option<Training>
}

impl Workflow {
    /// Load workflow from a file.
    pub fn from_file(path: impl Into<std::path::PathBuf>) -> anyhow::Result<Self> {
        use path_slash::*;

        let path = path.into().canonicalize()?;
        let parent = path.parent().unwrap();
        let file = std::fs::File::open(path.clone())?;
        let reader = std::io::BufReader::new(file);
        let mut parameters: Workflow = serde_json::from_reader(reader)?;

        // TODO: Simplify this. Wrap it in a function.
        if parameters.dataset.training.path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(parameters.dataset.training.path());
            parameters.dataset.training.set_path(path);
        }
        if parameters.dataset.regularization.is_some() && parameters.dataset.regularization.as_ref().unwrap().path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(parameters.dataset.regularization.as_ref().unwrap().path());
            parameters.dataset.regularization.as_mut().unwrap().set_path(path);
        }
        if let Some(training) = parameters.training.as_mut() {
            if training.output.directory.is_relative() {
                let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(training.output.directory.clone());
                training.output.directory = path;
            }    
        }
        parameters.replacements();
        Ok(parameters)
    }

    fn replace_string(&self, string: impl ToString) -> String {
        let string = string.to_string();
        string.replace("{network.dimension}", &self.training.as_ref().map(|training| training.network.dimension.to_string()).unwrap_or("no-training-network-dimension".to_string()))
            .replace("{network.alpha}", &self.training.as_ref().map(|training| training.network.alpha.to_string()).unwrap_or("no-training-network-alpha".to_string()))
            .replace("{prompt.instance}", &self.training.as_ref().map(|training| training.prompt.instance.clone()).unwrap_or("no-training-prompt-instance".to_string()))
            .replace("{prompt.class}", &self.training.as_ref().map(|training| training.prompt.class.clone()).unwrap_or("no-training-prompt-class".to_string()))
            .replace("{training.optimizer}", &self.training.as_ref().map(|training| training.optimizer.to_string()).unwrap_or("no-training-optimizer".to_string()))
            .replace("{training.learning_rate.scheduler}", &self.training.as_ref().map(|training| training.learning_rate.scheduler.to_string()).unwrap_or("no-training-learning-rate-scheduler".to_string()))
            .replace("{training.time.start}", &SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string())
            // .replace("{training.time.duration}", )
    }

    fn replacements(&mut self) {
        let parameters = self;
        parameters.training = parameters.training.take().map(|mut training| {
            training.output.directory = std::path::PathBuf::from(parameters.replace_string(training.output.directory.display()));
            training.output.name = parameters.replace_string(&training.output.name);    
            training
        });
        parameters.dataset.training = parameters.replace_string(parameters.dataset.training.path().display()).into();
        parameters.captioning = parameters
            .captioning
            .take()
            .map(|mut captioning| {
                captioning.replace = captioning.replace.iter().map(|(from, to)| {
                    (parameters.replace_string(from), parameters.replace_string(to))
                }).collect();
                captioning
            });
    }

    /// Create a new workflow structure.
    pub fn new(dataset: TrainingDataSet) -> Self {
        let training = Default::default();
        let captioning = Default::default();
        Workflow { captioning, dataset, training }
    }

    /// Set the training configuration.
    pub fn with_training(mut self, training: Option<Training>) -> Self {
        self.training = training;
        self
    }
}