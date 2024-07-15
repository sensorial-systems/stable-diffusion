//! Trainer's workflow.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::prelude::*;
use crate::utils::{Replace, Update};
use crate::{Captioning, Training, TrainingDataSet};
use path_slash::*;

/// The workflow structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Workflow {
    /// Base workflows. The current workflow will rewrite the base workflows.
    #[serde(default)]
    pub base: Vec<String>,
    /// Variables.
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,
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
    pub fn from_file(path: impl Into<std::path::PathBuf>) -> anyhow::Result<Self> {
        let path = path.into().canonicalize()?;
        let parent = path.parent().unwrap();
        let file = std::fs::File::open(path.clone())?;
        let reader = std::io::BufReader::new(file);
        let mut workflow: Workflow = serde_json::from_reader(reader)?;

        workflow.replace_with_absolute_paths(parent);

        let mut base = Workflow::default();
        for base_path in workflow.base.iter() {
            let base_path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(base_path).with_extension("json");
            let base_workflow = Self::from_file(base_path)?;
            base.update(base_workflow);
        }
        base.update(workflow);
        base.replacements();
        println!("{:#?}", base);
        Ok(base)
    }

    fn replace_with_absolute_paths(&mut self, parent: &std::path::Path) {
        if self.dataset.training.path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(self.dataset.training.path());
            self.dataset.training.set_path(path);
        }
        if self.dataset.regularization.is_some() && self.dataset.regularization.as_ref().unwrap().path().is_relative() {
            let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(self.dataset.regularization.as_ref().unwrap().path());
            self.dataset.regularization.as_mut().unwrap().set_path(path);
        }
        if let Some(training) = self.training.as_mut() {
            if training.output.directory.is_relative() {
                let path = std::path::PathBuf::from_slash(parent.to_slash().unwrap()).join(training.output.directory.clone());
                training.output.directory = path;
            }    
        }
    }

    fn replace_string(&self, string: impl ToString) -> String {
        let mut string = string.to_string();
        for (name, value) in &self.variables {
            string.replace_value(&format!("{{{name}}}"), value);
        }
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
            training.prompt.instance = parameters.replace_string(&training.prompt.instance);
            training.prompt.class = parameters.replace_string(&training.prompt.class);
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
        let variables = Default::default();
        let base = Default::default();
        Workflow { base, variables, captioning, dataset, training }
    }

    /// Set the training configuration.
    pub fn with_training(mut self, training: Option<Training>) -> Self {
        self.training = training;
        self
    }
}

impl Update for Workflow {
    fn update(&mut self, base: Self) {
        self.base.extend(base.base);
        self.variables.extend(base.variables);
        self.captioning.update(base.captioning);
        self.dataset.update(base.dataset);
        self.training.update(base.training);
    }
}