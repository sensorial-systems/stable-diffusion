//! The Trainer module contains the training configuration and the training process.

use std::{path::PathBuf, process::Command};

pub mod training;
pub mod output;
pub mod optimizer;
pub mod scheduler;
pub mod parameters;

pub use training::*;
pub use output::*;
pub use optimizer::*;
pub use scheduler::*;
pub use parameters::*;

use crate::{environment::Environment, model_file_format::ModelFileFormat, precision::FloatPrecision};

/// The Trainer structure.
pub struct Trainer {
    /// The environment to use for the training process.
    pub environment: Environment,
    /// The number of times to repeat the training images.
    pub training_images_repeat: usize,
    /// The number of times to repeat the regularization images.
    pub regularization_images_repeat: usize,
    /// The maximum resolution of the images to use for the training process.
    pub resolution: (usize, usize),
    /// The format to save the model as.
    pub save_model_as: ModelFileFormat,
    /// The module to use for the network.
    pub network_module: String,
    /// The learning rate for the text encoder.
    pub text_encoder_lr: f32,
    /// The learning rate for the unet.
    pub unet_lr: f32,
    /// The number of cycles for the learning rate scheduler.
    pub lr_scheduler_num_cycles: usize,
    /// The learning rate for the training process.
    pub learning_rate: f32,
    /// The number of warmup steps for the learning rate.
    pub lr_warmup_steps: usize,
    /// The batch size for the training process.
    pub train_batch_size: usize,
    /// The maximum number of training steps.
    pub max_train_steps: usize,
    /// The frequency to 
    pub save_every_n_epochs: usize,
    /// The precision to use for mixed precision training.
    pub mixed_precision: FloatPrecision,
    /// The precision to use for saving the model.
    pub save_precision: FloatPrecision,
    /// The maximum gradient norm.
    pub max_grad_norm: f32,
    /// The maximum number of data loader workers.
    pub max_data_loader_n_workers: usize,
    /// The number of steps for the bucket resolution.
    pub bucket_reso_steps: usize,
    /// The noise offset.
    pub noise_offset: f32,
}

impl Default for Trainer {
    fn default() -> Self {
        Trainer {
            environment: Default::default(),
            training_images_repeat: 40,
            regularization_images_repeat: 1,
            resolution: (1024,1024),
            save_model_as: ModelFileFormat::Safetensors,
            network_module: "networks.lora".to_string(),
            text_encoder_lr: 5e-05,
            unet_lr: 0.0001,
            lr_scheduler_num_cycles: 1,
            learning_rate: 0.0001,
            lr_warmup_steps: 48,
            train_batch_size: 1,
            max_train_steps: 480,
            save_every_n_epochs: 1,
            mixed_precision: FloatPrecision::F16,
            save_precision: FloatPrecision::F16,
            max_grad_norm: 1.0,
            max_data_loader_n_workers: 0,
            bucket_reso_steps: 64,
            noise_offset: 0.0,
        }
    }
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
        if let Some(path) = std::env::var_os("TRAINING_DIR") {
            PathBuf::from(path)
        } else {
            std::env::temp_dir().join(uuid::Uuid::new_v4().to_string())
        }
    }

    /// Start the training process.
    pub fn start(&mut self, parameters: &Parameters) {
        let training_dir = Self::training_dir();
        self.prepare(parameters, &training_dir);
        self.activate();
        self.caption(parameters, &training_dir);
        self.train(parameters, &training_dir);
        self.deactivate();
    }

    fn image_dir(training_dir: &PathBuf) -> PathBuf {
        training_dir.join("img")
    }

    fn reg_dir(training_dir: &PathBuf) -> PathBuf {
        training_dir.join("reg")
    }

    fn subject_dir(&self, parameters: &Parameters, training_dir: &PathBuf) -> PathBuf {
        Self::image_dir(training_dir).join(format!("{}_{} {}", self.training_images_repeat, parameters.prompt.instance, parameters.prompt.class))
    }

    fn activate(&mut self) {
        self.environment.activate();
    }

    fn deactivate(&mut self) {
        self.environment.deactivate();
    }

    fn prepare(&self, parameters: &Parameters, training_dir: &PathBuf) {
        let image_dir = self.subject_dir(parameters, training_dir);
        let class_dir = Self::reg_dir(training_dir).join(format!("{}_{}", self.regularization_images_repeat, parameters.prompt.class));
        std::fs::create_dir_all(training_dir.join("log")).unwrap();
        std::fs::create_dir_all(training_dir.join("model")).unwrap();
        std::fs::create_dir_all(&image_dir).unwrap();
        std::fs::create_dir_all(&class_dir).unwrap();
        println!("{}", parameters.dataset.training.path().display());
        for file in parameters.dataset.training.path().read_dir().unwrap() {
            let file = file.unwrap().path();
            let file_name = file.file_name().unwrap();
            std::fs::copy(&file, image_dir.join(file_name)).unwrap();
        }

        if let Some(regularization) = &parameters.dataset.regularization {
            for file in regularization.path().read_dir().unwrap() {
                let file = file.unwrap().path();
                let file_name = file.file_name().unwrap();
                std::fs::copy(&file, class_dir.join(file_name)).unwrap();
            }
        }
    }

    fn caption(&self, parameters: &Parameters, training_dir: &PathBuf) {
        let image_dir = self.subject_dir(parameters, training_dir);
        let python_executable = self.environment.python_executable_path();
        Command::new(python_executable)
        .arg(self.environment.kohya_ss().join("finetune").join("make_captions.py"))
        .args(["--batch_size", "1"])
        .args(["--num_beams", "1"])
        .args(["--top_p", "0.9"])
        .args(["--max_length", "75"])
        .args(["--min_length", "5"])
        .arg("--beam_search")
        .args(["--caption_extension", ".txt"])
        .arg(&image_dir)
        .args(["--caption_weights", "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"])
        .status()
        .expect("Failed to execute command");
        for txt in image_dir.read_dir().unwrap() {
            let txt = txt.unwrap().path();
            match txt.extension() {
                Some(extension) => {
                    if extension == "txt" {
                        let content = format!("{} {} {}", parameters.prompt.instance, parameters.prompt.class, std::fs::read_to_string(&txt).unwrap());
                        std::fs::write(txt, content).expect("Failed to update txt file");        
                    }
                },
                None => {
                    println!("Failed to get extension of: {}", txt.display())
                }
            }
        }
    }

    fn train(&self, parameters: &Parameters, training_dir: &PathBuf) {
        Command::new("accelerate")
        .arg("launch")
        .arg("--num_cpu_threads_per_process=2")
        .arg(self.environment.kohya_ss().join("sdxl_train_network.py"))
        .args(["--train_data_dir", &Self::image_dir(training_dir).display().to_string()])
        .args(["--reg_data_dir", &Self::reg_dir(training_dir).display().to_string()])
        .args(["--output_dir", &parameters.output.directory.display().to_string()])
        .args(["--output_name", &parameters.output.name])
        .args(["--pretrained_model_name_or_path", &parameters.training.pretrained_model])
        .args(["--resolution", &format!("{},{}", self.resolution.0, self.resolution.1)])
        .args(["--save_model_as", &self.save_model_as.to_string()])
        .args(["--network_alpha", &parameters.network.alpha.to_string()])
        .args(["--network_module", &self.network_module])
        .args(["--network_dim", &parameters.network.dimension.to_string()])
        .args(["--text_encoder_lr", &self.text_encoder_lr.to_string()])
        .args(["--unet_lr", &self.unet_lr.to_string()])
        .args(["--lr_scheduler_num_cycles", &self.lr_scheduler_num_cycles.to_string()])
        .arg("--no_half_vae")
        .args(["--learning_rate", &self.learning_rate.to_string()])
        .args(["--lr_scheduler", &parameters.training.learning_rate.scheduler.to_string()])
        // .args(["--lr_warmup_steps", &self.lr_warmup_steps.to_string()])
        .args(["--train_batch_size", &self.train_batch_size.to_string()])
        // .args(["--max_train_steps", &self.max_train_steps.to_string()])
        .args(["--save_every_n_epochs", &self.save_every_n_epochs.to_string()])
        .args(["--mixed_precision", &self.mixed_precision.to_string()])
        .args(["--save_precision", &self.save_precision.to_string()])
        .args(["--optimizer_type", &parameters.training.optimizer.to_string()])
        .args(["--max_grad_norm", &self.max_grad_norm.to_string()])
        .args(["--max_data_loader_n_workers", &self.max_data_loader_n_workers.to_string()])
        
        // Move it to Adafactor
        // .args(["--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False"])
        
        .arg("--xformers")
        .arg("--enable_bucket")
        .args(["--min_bucket_reso", "256"])
        .args(["--max_bucket_reso", "2048"])
        .args(["--bucket_reso_steps", &self.bucket_reso_steps.to_string()])
        .arg("--bucket_no_upscale")
        .args(["--noise_offset", &self.noise_offset.to_string()])
        .status()
        .expect("Failed to execute command");
    }
}
