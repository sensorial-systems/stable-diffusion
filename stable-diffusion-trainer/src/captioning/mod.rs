//! Preparation parameters.

use std::collections::HashMap;

use crate::{prelude::*, utils::{ReferenceResolver, Update, Variable}};

fn default_batch_size() -> Variable<usize> { 1.into() }
fn default_num_beams() -> Variable<usize> { 1.into() }
fn default_top_p() -> Variable<f64> { 0.9.into() }
fn default_min_length() -> Variable<usize> { 5.into() }
fn default_max_length() -> Variable<usize> { 75.into() }

/// The preparation parameters.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Captioning {
    /// The batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: Variable<usize>,
    /// The number of beamsn.
    #[serde(default = "default_num_beams")]
    pub num_beams: Variable<usize>,
    /// The top_p value.
    #[serde(default = "default_top_p")]
    pub top_p: Variable<f64>,
    /// The minimum length.
    #[serde(default = "default_min_length")]
    pub min_length: Variable<usize>,
    /// The maximum length.
    #[serde(default = "default_max_length")]
    pub max_length: Variable<usize>,
    /// String replacement (From, To).
    pub replace: HashMap<String, String>
}

impl Update for Captioning {
    fn update(&mut self, base: Self) {
        self.batch_size.update(base.batch_size);
        self.num_beams.update(base.num_beams);
        self.top_p.update(base.top_p);
        self.min_length.update(base.min_length);
        self.max_length.update(base.max_length);
        self.replace.update(base.replace);
    }
}

impl ReferenceResolver for Captioning {
    fn resolve_references(&mut self, variables: &std::collections::HashMap<String, serde_json::Value>) {
        self.batch_size.resolve_references(variables);
        self.num_beams.resolve_references(variables);
        self.top_p.resolve_references(variables);
        self.min_length.resolve_references(variables);
        self.max_length.resolve_references(variables);
        for (_, to) in self.replace.iter_mut() {
            to.resolve_references(variables);
        }
    }
}