//! Prompt module.

use crate::prelude::*;

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
