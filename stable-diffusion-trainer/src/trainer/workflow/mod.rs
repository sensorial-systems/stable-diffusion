//! Trainer's workflow.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::prelude::*;
use crate::{Captioning, Training, TrainingDataSet};
use json_template::{Context, Deserializer, JSON};
use path_slash::*;
use serde_json::{json, Value};

/// The workflow structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Workflow {
    /// The captioning parameters.
    pub captioning: Option<Captioning>,
    /// The dataset to use for the training process.
    #[serde(default)]
    pub dataset: TrainingDataSet,
    /// The training to use for the training process.
    pub training: Option<Training>
}

impl Workflow {
    /// Load workflow from a file.
    pub fn from_file(path: impl Into<std::path::PathBuf>, input: Option<std::path::PathBuf>) -> anyhow::Result<Self> {
        let training = crate::Model::default();
        println!("{}", serde_json::to_string_pretty(&training)?);

        let path = path.into().canonicalize()?;
        let parent = path.parent().unwrap().to_path_buf();

        let time: String = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
        let data = serde_json::json!({
            "input": {
                "time": time
            }
        });
        let data = input.map(|input| {
            let input = input.canonicalize().expect(format!("Failed to canonicalize input file: {}", input.display()).as_str());
            let input: Value = Deserializer::new().deserialize(input).unwrap();
            let mut input = json!({ "input": input });
            input.add_recursive(data.clone());
            input
        }).unwrap_or(data);
        let context = Context::new().with_data(data);
        let mut workflow: Workflow = Deserializer::new().deserialize_with_context(path, &context)?;
        workflow.replace_with_absolute_paths(parent.as_path());
        Ok(workflow)
    }

    fn replace_with_absolute_paths(&mut self, parent: &std::path::Path) {
        if self.dataset.training.path().is_relative() && !self.dataset.training.path().display().to_string().is_empty() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(self.dataset.training.path());
            self.dataset.training.set_path(path);
        }
        if let Some(regularization) = self.dataset.regularization.as_mut() {
            if regularization.path().is_relative() {
                let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(regularization.path());
                regularization.set_path(path);
            }
        }
        if let Some(training) = self.training.as_mut() {
            if training.output.directory.is_relative() {
                let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(training.output.directory.clone());
                training.output.directory = path;
            }
        }
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

    /// Set the captioning configuration.
    pub fn with_captioning(mut self, captioning: Option<Captioning>) -> Self {
        self.captioning = captioning;
        self
    }
}
