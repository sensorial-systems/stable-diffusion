//! The traits module.

/// The replace trait.
pub trait Replace {
    /// Replace the value.
    fn replace_value(&mut self, name: &String, value: &serde_json::Value);
}

impl Replace for String {
    fn replace_value(&mut self, name: &String, value: &serde_json::Value) {
        let value = match value {
            serde_json::Value::String(string) => string.clone(),
            _ => value.to_string()
        };
        *self = self.replace(name, &value);
    }
}

/// The update trait.
pub trait Update {
    /// Update the value.
    fn update(&mut self, base: Self);
}

impl<T: Update> Update for Option<T> {
    fn update(&mut self, base: Self) {
        if let Some(base) = base {
            if let Some(self_) = self {
                self_.update(base);
            } else {
                *self = Some(base);
            }
        }
    }
}

impl<T> Update for Vec<T> {
    fn update(&mut self, base: Self) {
        self.extend(base);
    }
}