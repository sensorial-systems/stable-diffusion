use std::path::PathBuf;

use stable_diffusion_trainer::*;

use clap::{Args, Subcommand};

#[derive(Subcommand, Debug, Clone)]
pub enum Setup {
    /// Setup the trainer by specifying the kohya_ss's folder.
    Setup {
        /// kohya_ss's folder.
        #[arg(short, long)]
        kohya_ss: Option<PathBuf>,
    }
}

impl Setup {
    pub fn setup(&mut self) {
        self.setup_kohya_ss();
    }

    fn setup_kohya_ss(&mut self) {
        let mut exit = false;
        while !exit {
            let Setup::Setup { kohya_ss } = self;
            *kohya_ss = kohya_ss.take().or_else(|| rfd::FileDialog::new().set_title("Select kohya_ss's folder").pick_folder());
            if let Some(path) = &kohya_ss {
                let environment = Environment::new().with_kohya_ss(path.clone());
                if !environment.python_executable_path().exists() {
                    rfd::MessageDialog::new()
                        .set_title("Error")
                        .set_description("The selected folder is not a valid kohya_ss folder. Make sure its venv folder exists and that it's initialized properly.")
                        .show();
                    *kohya_ss = None;
                } else {
                    environment.save().expect("Failed to save config file.");
                    exit = true;
                }
            } else {
                exit = true;
            }
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct Arguments {
    /// Path to the training workflow JSON file.
    #[arg(short, long)]
    workflow: Option<PathBuf>,

    #[arg(short, long)]
    prepare: bool,

    #[command(subcommand)]
    setup: Option<Setup>
}

impl Arguments {
    pub fn execute(self) -> anyhow::Result<()> {
        if let Some(mut setup) = self.setup {
            setup.setup();
        } else {
            let config = self.workflow.as_ref().ok_or_else(|| anyhow::anyhow!("No config file provided."))?;
            let parameters = Workflow::from_file(config)?;
            Trainer::new().start(&parameters);
        }
        Ok(())
    }
}
