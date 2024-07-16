//! The variable module.

use serde::de::DeserializeOwned;

use crate::prelude::*;

use super::{ReferenceResolver, Update};

/// The variable type.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Variable<T> {
    /// The value.
    Resolved(T),
    /// The reference.
    Reference(String),
}

impl<T> From<T> for Variable<T> {
    fn from(value: T) -> Self {
        Variable::Resolved(value)
    }
}

impl<T: Default> Default for Variable<T> {
    fn default() -> Self {
        Variable::Resolved(T::default())
    }
}

impl<T> Variable<T> {
    /// Get a reference to the resolved value.
    pub fn resolved(&self) -> Option<&T> {
        match self {
            Variable::Resolved(value) => Some(value),
            _ => None
        }
    }

    /// Get a mutable reference to the resolved value.
    pub fn resolved_mut(&mut self) -> Option<&mut T> {
        match self {
            Variable::Resolved(value) => Some(value),
            _ => None
        }
    }
}

// Implement display for Variable
impl<T: std::fmt::Display> std::fmt::Display for Variable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Variable::Reference(reference) => write!(f, "{}", reference),
            Variable::Resolved(value) => write!(f, "{}", value)
        }
    }
}

impl<T: DeserializeOwned> ReferenceResolver for Variable<T> {
    fn resolve_reference(&mut self, reference: &str, value: &serde_json::Value) {
        if let Variable::Reference(self_reference) = self {
            let self_reference = &self_reference[1..self_reference.len() - 1]; // remove {}
            if self_reference == reference {
                *self = serde_json::from_value(value.clone()).unwrap();
            }
        }
    }
}

impl<T> Update for Variable<T> {
    fn update(&mut self, base: Self) {
        *self = base;
    }
}