#![doc=include_str!("../README.md")]

use std::path::PathBuf;

use stable_diffusion_trainer::*;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Debug)]
pub struct Setup {
    /// kohya_ss path.
    #[arg(short, long)]
    pub kohya_ss: Option<PathBuf>,
}

impl Setup {
    pub fn setup(&mut self) {
        self.setup_kohya_ss();
    }

    fn setup_kohya_ss(&mut self) {
        let mut exit = false;
        while !exit {
            self.kohya_ss = self.kohya_ss.take().or_else(|| rfd::FileDialog::new().set_title("Select kohya_ss's folder").pick_folder());
            if let Some(kohya_ss) = &self.kohya_ss {
                let environment = Environment::new().with_kohya_ss(kohya_ss.clone());
                if !environment.python_executable_path().exists() {
                    rfd::MessageDialog::new()
                        .set_title("Error")
                        .set_description("The selected folder is not a valid kohya_ss folder. Make sure its venv folder exists and that it's initialized properly.")
                        .show();
                    self.kohya_ss = None;
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

#[derive(Args, Debug)]
pub struct Train {
    /// Path to the training config JSON file.
    #[arg(short, long)]
    config: PathBuf
}

#[derive(Subcommand)]
pub enum Command {
    /// Setup the trainer.
    Setup(Setup),
    /// Train the model.
    Train(Train),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Setup(mut setup) => {
            setup.setup();
        },
        Command::Train(train) => {
            let parameters = Parameters::from_file(train.config)?;
            Trainer::new().start(&parameters);
        }
    }
    Ok(())
}
