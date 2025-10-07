#![doc=include_str!("../README.md")]

use clap::{Parser, Subcommand};

mod train;

#[derive(Debug, Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Arguments {
    #[command(subcommand)]
    command: Command
}

#[derive(Debug, Subcommand, Clone)]
#[command(author, version, about, long_about = None)]
pub enum Command {
    /// Train a Stable Diffusion model
    Train(train::Arguments),
}

impl Arguments {
    pub fn execute(self) -> anyhow::Result<()> {
        match self.command {
            Command::Train(args) => args.execute(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    Arguments::parse().execute()
}
