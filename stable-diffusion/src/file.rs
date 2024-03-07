//! A module for handling local files and files in a repository.

use std::path::PathBuf;


/// A repository containing a file.
pub struct Repository {
    /// The repository containing the file.
    pub repository: PathBuf,
    /// The path to the file in the repository.
    pub path: PathBuf
}

impl Repository {
    /// Create a new `Repository` instance from a repository and a path.
    pub fn new(repository: impl AsRef<std::path::Path>, path: impl AsRef<std::path::Path>) -> Self {
        let repository = repository.as_ref().into();
        let path = path.as_ref().into();
        Self { repository, path }
    }

    /// Fetch the file from the repository.
    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        if self.repository.exists() {
            Ok(self.repository.join(&self.path).into())
        } else {
            let api = hf_hub::api::sync::Api::new()?;
            Ok(api.model(self.repository.display().to_string()).get(&self.path.display().to_string())?)    
        }
    }
}

/// A file that can be fetched.
pub enum File {
    /// A local file.
    Path(std::path::PathBuf),
    /// A file in a repository.
    Repository(Repository),
}

impl From<std::path::PathBuf> for File {
    fn from(path: std::path::PathBuf) -> Self {
        Self::Path(path)
    }
}

impl From<&str> for File {
    fn from(path: &str) -> Self {
        Self::Path(path.into())
    }
}

impl File {
    /// Fetch the file.
    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        match self {
            Self::Path(path) => Ok(path.clone()),
            Self::Repository(repository) => repository.fetch()
        }
    }
}
