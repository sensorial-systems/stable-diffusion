use std::path::PathBuf;

pub struct Repository {
    pub repository: String,
    pub path: PathBuf
}

impl Repository {
    pub fn new(repository: impl Into<String>, path: impl AsRef<std::path::Path>) -> Self {
        let repository = repository.into();
        let path = path.as_ref().into();
        Self { repository, path }
    }

    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        Ok(api.model(self.repository.to_string()).get(&self.path.display().to_string())?)
    }
}

pub enum File {
    Path(std::path::PathBuf),
    Repository(Repository),
}

impl File {
    pub fn fetch(&self) -> anyhow::Result<PathBuf> {
        match self {
            Self::Path(path) => Ok(path.clone()),
            Self::Repository(repository) => repository.fetch()
        }
    }
}
