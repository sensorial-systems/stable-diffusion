//! Optimizer module for the trainer.

use crate::prelude::*;
use std::fmt::Display;

/// The optimizer to use for the training process.
#[derive(Debug, Serialize, Deserialize)]
pub enum Optimizer {
    /// AdamW optimizer.
    AdamW,
    /// AdamW 8-bit optimizer.
    AdamW8bit,
    /// Adafactor optimizer.
    Adafactor,
    /// DAdaptation optimizer.
    DAdaptation,
    /// DAdaptationGrad optimizer.
    DAdaptationGrad,
    /// DAdaptAdam optimizer.
    DAdaptAdam,
    /// DAdaptAdan optimizer.
    DAdaptAdan,
    /// DAdaptAdamIP optimizer.
    DAdaptAdamIP,
    /// DAdaptAdamReprint optimizer.
    DAdaptAdamReprint,
    /// DAdaptLion optimizer.
    DAdaptLion,
    /// DAdaptSGD optimizer.
    DAdaptSGD,
    /// Lion optimizer.
    Lion,
    /// Lion 8-bit optimizer.
    Lion8bit,
    /// PagedAdamW 8-bit optimizer.
    PagedAdamW8bit,
    /// PagedAdamW 32-bit optimizer.
    PagedAdamW32bit,
    /// PagedLion 8-bit optimizer.
    PagedLion8bit,
    /// Prodigy optimizer.
    Prodigy,
    /// SGDNesterov optimizer.
    SGDNesterov,
    /// SGDNesterov 8-bit optimizer.
    SGDNesterov8bit
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::AdamW
    }
}

impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimizer::AdamW => write!(f, "AdamW"),
            Optimizer::AdamW8bit => write!(f, "AdamW8bit"),
            Optimizer::Adafactor => write!(f, "Adafactor"),
            Optimizer::DAdaptation => write!(f, "DAdaptation"),
            Optimizer::DAdaptationGrad => write!(f, "DAdaptationGrad"),
            Optimizer::DAdaptAdam => write!(f, "DAdaptAdam"),
            Optimizer::DAdaptAdan => write!(f, "DAdaptAdan"),
            Optimizer::DAdaptAdamIP => write!(f, "DAdaptAdamIP"),
            Optimizer::DAdaptAdamReprint => write!(f, "DAdaptAdamReprint"),
            Optimizer::DAdaptLion => write!(f, "DAdaptLion"),
            Optimizer::DAdaptSGD => write!(f, "DAdaptSGD"),
            Optimizer::Lion => write!(f, "Lion"),
            Optimizer::Lion8bit => write!(f, "Lion8bit"),
            Optimizer::PagedAdamW8bit => write!(f, "PagedAdamW8bit"),
            Optimizer::PagedAdamW32bit => write!(f, "PagedAdamW32bit"),
            Optimizer::PagedLion8bit => write!(f, "PagedLion8bit"),
            Optimizer::Prodigy => write!(f, "Prodigy"),
            Optimizer::SGDNesterov => write!(f, "SGDNesterov"),
            Optimizer::SGDNesterov8bit => write!(f, "SGDNesterov8bit")
        }
    }
}