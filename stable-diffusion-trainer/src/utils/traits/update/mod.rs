//! The update trait module.

use std::collections::HashMap;

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

impl<K, V> Update for HashMap<K, V>
where K: Eq + std::hash::Hash
{
    fn update(&mut self, base: Self) {
        self.extend(base);
    }
}
