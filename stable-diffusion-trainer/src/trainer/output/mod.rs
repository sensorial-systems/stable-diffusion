//! Trainer's output.

use std::path::PathBuf;

use crate::prelude::*;

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