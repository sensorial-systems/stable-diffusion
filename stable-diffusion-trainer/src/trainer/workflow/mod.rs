//! Trainer's workflow.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::prelude::*;
use crate::utils::{ReferenceResolver, Update};
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
        let time: String = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_string();
        let time = serde_json::Value::String(time);
        base.variables.insert("time".to_string(), time);
        base.resolve_references(&base.variables.clone());
        Ok(base)
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

impl ReferenceResolver for Workflow {
    fn resolve_references(&mut self, variables: &HashMap<String, serde_json::Value>) {
        for value in &mut self.variables.values_mut() {
            value.resolve_references(variables);
        }
        self.captioning.resolve_references(variables);
        self.dataset.resolve_references(variables);
        self.training.resolve_references(variables);
    }
}