//! Environment module.
use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use serde::{Serialize, Deserialize};

/// The environment structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct Environment {
    kohya_ss: PathBuf
}

impl Default for Environment {
    fn default() -> Self {
        Self::load()
    }
}

impl Environment {
    /// Load the environment from the configuration file.
    pub fn load() -> Self {
        let kohya_ss = ProjectDirs::from("com", "sensorial-systems", "stable-diffusion-trainer")
            .map(|dirs| dirs.config_dir().to_path_buf())
            .map(|config_dir| config_dir.join("config.json"))
            .and_then(|config_path| std::fs::read_to_string(config_path).ok())
            .and_then(|config| serde_json::from_str::<Environment>(&config).ok())
            .map(|env| env.kohya_ss().to_path_buf())
            .unwrap_or_default();
        Environment { kohya_ss }
    }

    /// Save the environment to the configuration file.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config_dir = ProjectDirs::from("com", "sensorial-systems", "stable-diffusion-trainer")
            .map(|dirs| dirs.config_dir().to_path_buf())
            .expect("Failed to get config_dir");
        let config_path = config_dir.join("config.json");
        std::fs::create_dir_all(&config_dir)?;
        let json = serde_json::to_string(self)?;
        std::fs::write(config_path, json)?;
        Ok(())
    }

    /// Create a new environment structure.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the kohya_ss path.
    pub fn with_kohya_ss(mut self, kohya_ss: PathBuf) -> Self {
        self.kohya_ss = kohya_ss;
        self
    }

    /// Get the kohya_ss path.
    pub fn kohya_ss(&self) -> &Path {
        &self.kohya_ss
    }

    /// Get the kohya_ss path.
    pub fn binary_path(&self) -> PathBuf {
        #[cfg(target_os = "windows")]
        let python_executable = self.kohya_ss.join("venv").join("Scripts");
        #[cfg(not(target_os = "windows"))]
        let python_executable = self.kohya_ss.join("venv").join("bin");
        python_executable
    }

    /// Get the kohya_ss path.
    pub fn python_executable_path(&self) -> PathBuf {
        #[cfg(target_os = "windows")]
        let python_executable = self.binary_path().join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let python_executable = self.binary_path().join("python");
        python_executable
    }

    /// Get the kohya_ss path.
    pub fn activate(&self) {
        std::env::set_var("PYTHONPATH", self.kohya_ss.join("venv").join("Lib").join("site-packages"));
        #[cfg(target_os = "windows")]
        std::env::set_var("PATH", format!("{};{}", std::env::var("PATH").unwrap(), self.binary_path().display()));
        #[cfg(not(target_os = "windows"))]
        std::env::set_var("PATH", format!("{}:{}", std::env::var("PATH").unwrap(), self.binary_path().display()));
        // FIXME: This is too invasive. It should be done in a more controlled way.
        std::env::set_current_dir(&self.kohya_ss).unwrap();
    }
}
