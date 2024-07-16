#![allow(missing_docs)]

pub mod replacing {
    use serde_json::*;

    fn get_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
        path.split(".").fold(Some(value), |acc, segment| acc?.get(segment))
    }

    pub fn from_str(value: &str) -> Result<Value> {
        let mut json: Value = serde_json::from_str(value)?;
        let clone = json.clone();
        if let Some(object) = json.as_object_mut() {
            for (_, value) in object.iter_mut() {
                if let Some(string) = value.as_str() {
                    if string.starts_with('{') && string.ends_with('}') {
                        let path = &string[1 .. string.len() - 1];
                        if let Some(new_value) = get_path(&clone, path).cloned() {
                            *value = new_value
                        }
                    }
                }
            }
        }
        Ok(json)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn json() {
        let file = include_str!("test.json");
        let json: serde_json::Value = replacing::from_str(file).unwrap();
        println!("{}", json);
    }
}