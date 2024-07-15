//! ImageDataSet is a simple struct that holds the path to a directory containing images. It is used to represent a dataset of images that can be used for training a model.

use crate::{prelude::*, utils::Update};
use std::path::{Path, PathBuf};

/// A data set of images.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ImageDataSet(PathBuf);

impl ImageDataSet {
    /// Create a new image data set from a directory.
    pub fn from_dir<Path: AsRef<std::path::Path>>(path: Path) -> Self {
        let path = path.as_ref().to_path_buf();
        ImageDataSet(path)
    }

    /// Set the path to the image data set.
    pub fn set_path(&mut self, path: PathBuf) {
        self.0 = path;
    }

    /// Get the path to the image data set.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl From<String> for ImageDataSet {
    fn from(path: String) -> Self {
        ImageDataSet(PathBuf::from(path))
    }
}

impl Update for ImageDataSet {
    fn update(&mut self, base: Self) {
        if base.path().exists() {
            self.0 = base.0;
        }
    }
}