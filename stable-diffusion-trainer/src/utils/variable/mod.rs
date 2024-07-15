//! The variable module.

use crate::prelude::*;

use super::Update;

/// The variable type.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Variable<T> {
    /// The reference.
    Reference(String),
    /// The value.
    Value(T)
}

impl<T> From<T> for Variable<T> {
    fn from(value: T) -> Self {
        Variable::Value(value)
    }
}

impl<T: Default> Default for Variable<T> {
    fn default() -> Self {
        Variable::Value(T::default())
    }
}

// Implement display for Variable
impl<T: std::fmt::Display> std::fmt::Display for Variable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Variable::Reference(reference) => write!(f, "{}", reference),
            Variable::Value(value) => write!(f, "{}", value)
        }
    }
}

impl<T> Update for Variable<T> {
    fn update(&mut self, base: Self) {
        *self = base;
    }
}