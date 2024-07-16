//! Bucketing module.

use crate::{prelude::*, utils::{ReferenceResolver, Variable}};

fn default_min_resolution() -> Variable<usize> { 512.into() }
fn default_max_resolution() -> Variable<usize> { 1024.into() }
fn default_step() -> Variable<usize> { 64.into() }

/// Bucketing.
#[derive(Debug, Serialize, Deserialize)]
pub struct Bucketing {
    /// The minimum resolution.
    #[serde(default = "default_min_resolution")]
    pub min_resolution: Variable<usize>,
    /// The maximum resolution.
    #[serde(default = "default_max_resolution")]
    pub max_resolution: Variable<usize>,
    /// The number of steps for the bucket resolution.
    #[serde(default = "default_step")]
    pub resolution_steps: Variable<usize>
}

impl ReferenceResolver for Bucketing {
    fn resolve_references(&mut self, variables: &std::collections::HashMap<String, serde_json::Value>) {
        self.min_resolution.resolve_references(variables);
        self.max_resolution.resolve_references(variables);
        self.resolution_steps.resolve_references(variables);
    }
}