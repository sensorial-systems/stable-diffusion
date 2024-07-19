//! Network module.

use crate::prelude::*;

/// The network structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    /// The dimension of the network.
    pub dimension: usize,
    /// The alpha value of the network.
    pub alpha: f32
}

impl Default for Network {
    fn default() -> Self {
        Network {
            dimension: 8,
            alpha: 1.0
        }
    }
}

impl Network {
    /// Create a new network structure.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the dimension of the network.
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension.into();
        self
    }

    /// Set the alpha value of the network.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.into();
        self
    }
}
