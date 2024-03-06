use std::path::PathBuf;

pub struct Repository {
    pub repository: PathBuf,
    pub path: PathBuf
}

impl Repository {
    pub fn new(repository: impl AsRef<std::path::Path>, path: impl AsRef<std::path::Path>) -> Self {
        let repository = repository.as_ref().into();
        let path = path.as_ref().into();
        Self { repository, path }
    }

    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        if self.repository.exists() {
            Ok(self.repository.join(&self.path).into())
        } else {
            let api = hf_hub::api::sync::Api::new()?;
            Ok(api.model(self.repository.display().to_string()).get(&self.path.display().to_string())?)    
        }
    }
}

pub enum File {
    Path(std::path::PathBuf),
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
    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        match self {
            Self::Path(path) => Ok(path.clone()),
            Self::Repository(repository) => repository.fetch()
        }
    }
}
