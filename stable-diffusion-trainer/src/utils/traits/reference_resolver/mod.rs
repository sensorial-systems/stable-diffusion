//! The reference resolver trait.

use std::{collections::HashMap, path::PathBuf};

/// The reference resolver trait.
pub trait ReferenceResolver {
    /// Resolve multiple references.
    fn resolve_references(&mut self, variables: &HashMap<String, serde_json::Value>) {
        for (name, value) in variables {
            self.resolve_reference(name, value);
        }
    }

    /// Resolve a single reference.
    fn resolve_reference(&mut self, _name: &str, _value: &serde_json::Value) {
        unimplemented!("Unimplemented resolve_reference should not happen.")
    }
}

impl ReferenceResolver for String {
    fn resolve_reference(&mut self, name: &str, value: &serde_json::Value) {
        let value = match value {
            serde_json::Value::String(string) => string.clone(),
            _ => value.to_string()
        };
        *self = self.replace(&format!("{{{name}}}"), &value);
    }
}

impl ReferenceResolver for serde_json::Value {
    fn resolve_reference(&mut self, name: &str, value: &serde_json::Value) {
        if let serde_json::Value::String(string) = self {
            string.resolve_reference(name, value);
        }
    }
}

impl<T: ReferenceResolver> ReferenceResolver for Option<T> {
    fn resolve_references(&mut self, variables: &HashMap<String, serde_json::Value>) {
        if let Some(self_) = self {
            self_.resolve_references(variables);
        }
    }
}

impl ReferenceResolver for PathBuf {
    fn resolve_references(&mut self, variables: &HashMap<String, serde_json::Value>) {
        let mut string = self.display().to_string();
        string.resolve_references(variables);
        *self = std::path::PathBuf::from(string);
    }
}
