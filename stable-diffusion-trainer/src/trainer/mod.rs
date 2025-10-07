//! The Trainer module contains the training configuration and the training process.

use std::{path::PathBuf, process::Command};

pub mod training;
pub mod workflow;

use rand::random;
pub use training::*;
pub use workflow::*;

use crate::{environment::Environment, Captioning};
use walkdir::WalkDir;

/// The Trainer structure.
#[derive(Default)]
pub struct Trainer {
    /// The environment to use for the training process.
    pub environment: Environment
}

impl Trainer {
    /// Create a new Trainer.
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the environment for the training process.
    pub fn with_environment(mut self, environment: Environment) -> Self {
        self.environment = environment;
        self
    }

    fn training_dir() -> PathBuf {
        std::env::temp_dir().join(uuid::Uuid::new_v4().to_string())
    }

    /// Start the training process.
    pub fn start(&mut self, parameters: &Workflow) {
        let training_dir = Self::training_dir();
        self.activate();
        if let Some(captioning) = parameters.captioning.as_ref() {
            self.caption(parameters, captioning);
        }
        if let Some(training) = parameters.training.as_ref() {
            self.prepare(training, parameters, &training_dir);
            self.train(training, &training_dir);
        }
        self.deactivate();
    }

    fn image_dir(training_dir: &PathBuf) -> PathBuf {
        training_dir.join("img")
    }

    fn reg_dir(training_dir: &PathBuf) -> PathBuf {
        training_dir.join("reg")
    }

    fn subject_dir(&self, training: &Training, training_dir: &PathBuf) -> PathBuf {
        Self::image_dir(training_dir).join(format!("{}_{} {}", training.images_repeat, training.prompt.instance, training.prompt.class))
    }

    fn activate(&mut self) {
        self.environment.activate();
    }

    fn deactivate(&mut self) {
        self.environment.deactivate();
    }

    fn prepare(&self, training: &Training, parameters: &Workflow, training_dir: &PathBuf) {
        let image_dir = self.subject_dir(training, training_dir);
        let class_dir = Self::reg_dir(training_dir).join(format!("{}_{}", training.regularization_images_repeat, training.prompt.class));
        std::fs::create_dir_all(training_dir.join("log")).unwrap();
        std::fs::create_dir_all(training_dir.join("model")).unwrap();
        std::fs::create_dir_all(&image_dir).unwrap();
        std::fs::create_dir_all(&class_dir).unwrap();
        for entry in WalkDir::new(parameters.dataset.training.path()).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension().unwrap();
                if ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff" && ext != "webp" && ext != "txt" {
                    continue;
                }
                let file_name = path.file_name().unwrap();
                std::fs::copy(&path, image_dir.join(file_name)).unwrap();    
            }
        }

        if let Some(regularization) = &parameters.dataset.regularization {
            for file in regularization.path().read_dir().unwrap() {
                let file = file.unwrap().path();
                let file_name = file.file_name().unwrap();
                std::fs::copy(&file, class_dir.join(file_name)).unwrap();
            }
        }
    }

    fn caption(&self, parameters: &Workflow, captioning: &Captioning) {
        let image_dir = parameters.dataset.training.path();
        let python_executable = self.environment.python_executable_path();
        std::env::set_current_dir(&self.environment.sd_scripts()).expect("Failed to set current directory");
        Command::new(python_executable)
            .arg(self.environment.sd_scripts().join("finetune").join("make_captions.py"))
            .args(["--batch_size", &captioning.batch_size.to_string()])
            .args(["--num_beams", &captioning.num_beams.to_string()])
            .args(["--top_p", &captioning.top_p.to_string()])
            .args(["--min_length", &captioning.min_length.to_string()])
            .args(["--max_length", &captioning.max_length.to_string()])
            .arg("--beam_search")
            .args(["--caption_extension", ".txt"])
            .arg(&image_dir)
            .args(["--caption_weights", "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"])
            .status()
            .expect("Failed to execute command");    
        for txt in image_dir.read_dir().expect(format!("Failed to read directory: {}", image_dir.display()).as_str()) {
            let txt = txt.unwrap().path();
            match txt.extension() {
                Some(extension) => {
                    if extension == "txt" {
                        let content = std::fs::read_to_string(&txt).unwrap();
                        let content = captioning.replace.iter().fold(content, |acc, (from, to)| acc.replace(from, to));
                        std::fs::write(txt, content).expect("Failed to update txt file");
                    }
                },
                None => {
                    println!("Failed to get extension of: {}", txt.display())
                }
            }
        }
    }

    fn train(&self, training: &Training, training_dir: &PathBuf) {
        let (width, height) = training.resolution.unwrap_or(training.model.resolution());
        let script = self.environment.sd_scripts().join(training.model.training_script(&training));
        let mut command = Command::new("accelerate");
        command
            .arg("launch")
            .args(["--gpu_ids","1"])
            // .arg("--num_cpu_threads_per_process=16")
            .arg(script)
            .args(["--seed", &training.seed.unwrap_or_else(|| random()).to_string()])
            .args(["--train_data_dir", &Self::image_dir(training_dir).display().to_string()])
            .args(["--reg_data_dir", &Self::reg_dir(training_dir).display().to_string()])
            .args(["--output_dir", &training.output.directory.display().to_string()])
            .args(["--output_name", &training.output.name])
            .args(["--pretrained_model_name_or_path", &training.model.checkpoint().as_str()])
            .args(["--resolution", &format!("{},{}", width, height)])
            .args(["--save_model_as", &training.output.save_model_as.to_string()])
            .args(["--lr_scheduler_num_cycles", &training.learning_rate.scheduler_num_cycles.to_string()])
            // .arg("--no_half_vae")
            .args(["--learning_rate", &training.learning_rate.amount.to_string()])
            .args(["--lr_scheduler", &training.learning_rate.scheduler.to_string()])
            .args(["--train_batch_size", &training.batch_size.to_string()])
            .args(["--max_train_steps", &training.max_train_steps.to_string()])
            .args(["--save_every_n_epochs", &training.output.save_every_n_epochs.to_string()])
            .args(["--mixed_precision", &training.mixed_precision.to_string()])
            .args(["--save_precision", &training.output.save_precision.to_string().replace("fp32", "float")])
            .args(["--optimizer_type", &training.optimizer.to_string()])
            .args(["--max_grad_norm", &training.max_grad_norm.to_string()])
            // .args(["--max_data_loader_n_workers", &training.max_data_loader_n_workers.to_string()])
            .args(["--noise_offset", &training.noise_offset.to_string()])
            // .arg("--cache_latents")
            .arg("--xformers");

    training.target.set_parameters(&mut command, training);
        // Move it to Adafactor
            // .args(["--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False"])

    if let Some(bucketing) = &training.bucketing {
        bucketing.set_parameters(&mut command);
    }

    println!("{:?}", command);

    command
        .status()
        .expect("Failed to execute command");
    }
}
