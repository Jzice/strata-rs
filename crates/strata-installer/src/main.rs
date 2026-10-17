//! # strata-installer
//!
//! Hardware detection, model download with resume, config generation, service
//! control and self-update. Windows-first, cross-platform capable.

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "strata-installer", version, about = "Strata installer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Detect the hardware (GPU, VRAM, RAM, disk) and print a JSON summary.
    Detect,
    /// Download a model file with resume and checksum verification.
    Download {
        /// Model name or direct URL.
        model: String,
        /// Destination directory.
        #[arg(long, default_value = "models")]
        out: String,
    },
    /// Write the run configuration for the chosen model and hardware.
    Config {
        /// Model name.
        model: String,
    },
    /// Start or stop the Strata server as a background process.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Debug, Subcommand)]
enum ServiceAction {
    Start,
    Stop,
    Status,
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Detect => println!("{{}}"),
        Command::Download { model, out } => println!("download {model} to {out}"),
        Command::Config { model } => println!("config for {model}"),
        Command::Service { action } => match action {
            ServiceAction::Start => println!("start"),
            ServiceAction::Stop => println!("stop"),
            ServiceAction::Status => println!("status"),
        },
    }
}
