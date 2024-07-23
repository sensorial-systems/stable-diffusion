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

impl Bucketing {
    /// Set parameters
    pub fn set_parameters(&self, command: &mut std::process::Command) {
        command
            .arg("--enable_bucket")
            // .arg("--bucket_no_upscale")
            .args(["--min_bucket_reso", &self.min_resolution.to_string()])
            .args(["--max_bucket_reso", &self.max_resolution.to_string()])
            .args(["--bucket_reso_steps", &self.resolution_steps.to_string()]);
    }
}