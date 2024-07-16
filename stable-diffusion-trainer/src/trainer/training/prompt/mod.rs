//! Prompt module.

use crate::{prelude::*, utils::ReferenceResolver};

/// The prompt structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Prompt {
    /// The instance prompt.
    pub instance: String,
    /// The class prompt.
    pub class: String
}

impl Prompt {
    /// Create a new prompt structure.
    pub fn new(instance: impl Into<String>, class: impl Into<String>) -> Self {
        let instance = instance.into();
        let class = class.into();
        Prompt { instance, class }
    }
}

impl ReferenceResolver for Prompt {
    fn resolve_references(&mut self, variables: &std::collections::HashMap<String, serde_json::Value>) {
        self.instance.resolve_references(variables);
        self.class.resolve_references(variables);
    }
}