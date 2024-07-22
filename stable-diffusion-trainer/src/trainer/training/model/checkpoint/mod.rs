//! Checkpoint module.

use crate::prelude::*;

/// The Checkpoint structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint(String);

impl From<&str> for Checkpoint {
    fn from(checkpoint: &str) -> Self {
        Self::new(checkpoint)
    }
}

impl From<String> for Checkpoint {
    fn from(checkpoint: String) -> Self {
        Self::new(checkpoint)
    }
}

impl Checkpoint {
    /// Create a new Checkpoint.
    pub fn new(checkpoint: impl Into<String>) -> Self {
        Self(checkpoint.into())
    }

    /// Get the checkpoint as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}