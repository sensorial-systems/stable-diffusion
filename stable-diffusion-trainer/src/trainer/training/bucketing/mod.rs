//! Bucketing module.

use crate::prelude::*;

fn default_min_resolution() -> usize { 512 }
fn default_max_resolution() -> usize { 1024 }
fn default_step() -> usize { 64 }

/// Bucketing.
#[derive(Debug, Serialize, Deserialize)]
pub struct Bucketing {
    /// The minimum resolution.
    #[serde(default = "default_min_resolution")]
    pub min_resolution: usize,
    /// The maximum resolution.
    #[serde(default = "default_max_resolution")]
    pub max_resolution: usize,
    /// The number of steps for the bucket resolution.
    #[serde(default = "default_step")]
    pub resolution_steps: usize
}
