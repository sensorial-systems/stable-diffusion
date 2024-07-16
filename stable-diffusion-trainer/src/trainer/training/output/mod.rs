//! Trainer's output.

use std::path::PathBuf;

use crate::{prelude::*, utils::ReferenceResolver};

/// The output structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    /// The name of the output.
    pub name: String,
    /// The directory to save the output to.
    pub directory: PathBuf
}

impl Output {
    /// Create a new output structure.
    pub fn new(name: impl Into<String>, directory: impl Into<PathBuf>) -> Self {
        let name = name.into();
        let directory = directory.into();
        Output { name, directory }
    }
}

impl ReferenceResolver for Output {
    fn resolve_references(&mut self, variables: &std::collections::HashMap<String, serde_json::Value>) {
        self.name.resolve_references(variables);
        self.directory.resolve_references(variables);
    }
}