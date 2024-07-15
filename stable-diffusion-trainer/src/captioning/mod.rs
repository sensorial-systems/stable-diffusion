//! Preparation parameters.

use crate::prelude::*;

fn default_batch_size() -> usize { 1 }
fn default_num_beams() -> usize { 1 }
fn default_top_p() -> f64 { 0.9 }
fn default_min_length() -> usize { 5 }
fn default_max_length() -> usize { 75 }

/// The preparation parameters.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Captioning {
    /// The batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// The number of beamsn.
    #[serde(default = "default_num_beams")]
    pub num_beams: usize,
    /// The top_p value.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// The minimum length.
    #[serde(default = "default_min_length")]
    pub min_length: usize,
    /// The maximum length.
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    /// String replacement (From, To).
    pub replace: Vec<(String, String)>
}
