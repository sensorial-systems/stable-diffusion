#![doc = include_str!("../README.md")]

mod dtype;
mod vae;
mod clip;
mod tokenizer;
mod unet;
mod file;

pub use vae::*;
pub use tokenizer::*;
pub use clip::*;
pub use dtype::*;
pub use unet::*;
pub use file::*;

use candle_transformers::models::stable_diffusion::{self, StableDiffusionConfig};

pub use anyhow::{Error, Result};
use candle::{Device, Tensor, D};

/// The `StableDiffusionParameters` struct is used to specify the parameters of the Stable Diffusion model.
pub struct StableDiffusionParameters {
    pub weights: StableDiffusionWeights,
    pub dtype: DType,
    pub config: StableDiffusionConfig,
    pub device: Device
}

impl StableDiffusionParameters {
    /// Create a new `StableDiffusionParameters` instance.
    pub fn new(weights: StableDiffusionWeights, device: Device, dtype: DType) -> anyhow::Result<Self> {
        let config = match weights.version {
            StableDiffusionVersion::V1_5 => stable_diffusion::StableDiffusionConfig::v1_5(None, None, None),
            StableDiffusionVersion::V2_1 => stable_diffusion::StableDiffusionConfig::v2_1(None, None, None),
            StableDiffusionVersion::XL => stable_diffusion::StableDiffusionConfig::sdxl(None, None, None),
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, None, None),
        };
        Ok(Self { device, weights, dtype, config })
    }
}

/// The `StableDiffusionWeights` struct is used to specify the weights of the Stable Diffusion model.
pub struct StableDiffusionWeights {
    pub version: StableDiffusionVersion,
    pub dtype: DType,
    pub unet: UNetWeights,
    pub vae: VAEWeights,
    pub clip: CLIPWeights,
    pub tokenizer: TokenizerWeights,
}

impl StableDiffusionWeights {
    /// Create a new `StableDiffusionWeights` instance from a version and dtype.
    pub fn new(version: StableDiffusionVersion, dtype: DType) -> Self {
        Self::from_repository(version, Some(version.repo().into()), dtype)
    }

    /// Create a new `StableDiffusionWeights` instance from a version, repository, and dtype.
    pub fn from_repository(version: StableDiffusionVersion, repository: Option<String>, dtype: DType) -> Self {
        let repository = repository.unwrap_or_else(|| version.repo().to_string());
        let unet = UNetWeights::from_repository(&repository, dtype);
        let vae = VAEWeights::from_repository(&repository, version, dtype);
        let clip = CLIPWeights::from_repository(&repository, version, dtype);
        let tokenizer = TokenizerWeights::from_repository(version);
        Self { version, dtype, unet, vae, clip, tokenizer }
    }

    /// Sets the weights of the UNet model.
    pub fn with_unet(self, unet: UNetWeights) -> Self {
        Self { unet, ..self }
    }

    /// Sets the weights of the VAE model.
    pub fn with_vae(self, vae: VAEWeights) -> Self {
        Self { vae, ..self }
    }

    /// Sets the weights of the CLIP model.
    pub fn with_clip(self, clip: CLIPWeights) -> Self {
        Self { clip, ..self }
    }

    /// Sets the weights of the Tokenizer model.
    pub fn with_tokenizer(self, tokenizer: TokenizerWeights) -> Self {
        Self { tokenizer, ..self }
    }
}

/// The `StableDiffusion` struct is used to specify the Stable Diffusion model.
pub struct StableDiffusion {
    version: StableDiffusionVersion,
    device: Device,
    dtype: DType,
    config: StableDiffusionConfig,
    unet: UNet,
    vae: VAE,
    tokenizer: Tokenizer,
    tokenizer_2: Option<Tokenizer>,
    clip: CLIP,
    clip_2: Option<CLIP>,
}

/// The `GenerationParameters` struct is used to specify the parameters of the generation process.
pub struct GenerationParameters {
    pub prompt: String,
    pub uncond_prompt: String,
    pub style_prompt: Option<String>,
    pub uncond_style_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub n_steps: Option<usize>,
    pub guidance_scale: Option<f64>,
    pub img2img: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>,
    pub img2img_strength: f64,
}

impl From<String> for GenerationParameters {
    fn from(prompt: String) -> Self {
        Self::new(prompt)
    }
}

impl GenerationParameters {
    /// Create a new `GenerationParameters` instance from a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        let uncond_prompt = Default::default();
        let width = Default::default();
        let height = Default::default();
        let n_steps = Default::default();
        let guidance_scale = Default::default();
        let style_prompt = Default::default();
        let uncond_style_prompt = Default::default();
        let img2img = Default::default();
        let img2img_strength = 0.5;
        Self { prompt, uncond_prompt, style_prompt, uncond_style_prompt, width, height, n_steps, guidance_scale, img2img, img2img_strength }
    }

    /// Sets the unconditional prompt.
    pub fn with_uncond_prompt(self, uncond_prompt: String) -> Self {
        Self { uncond_prompt, ..self }
    }

    /// Sets the style prompt.
    pub fn with_style_prompt(self, style_prompt: Option<String>) -> Self {
        Self { style_prompt, ..self }
    }

    /// Sets the unconditional style prompt.
    pub fn with_uncond_style_prompt(self, uncond_style_prompt: Option<String>) -> Self {
        Self { uncond_style_prompt, ..self }
    }

    /// Sets the width.
    pub fn with_width(self, width: Option<usize>) -> Self {
        Self { width, ..self }
    }

    /// Sets the height.
    pub fn with_height(self, height: Option<usize>) -> Self {
        Self { height, ..self }
    }

    /// Sets the number of steps.
    pub fn with_n_steps(self, n_steps: Option<usize>) -> Self {
        Self { n_steps, ..self }
    }

    /// Sets the guidance scale.
    pub fn with_guidance_scale(self, guidance_scale: Option<f64>) -> Self {
        Self { guidance_scale, ..self }
    }

    /// Sets the image to image.
    pub fn with_img2img(self, img2img: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>) -> Self {
        Self { img2img, ..self }
    }

    /// Sets the image to image strength.
    pub fn with_img2img_strength(self, img2img_strength: f64) -> Self {
        Self { img2img_strength, ..self }
    }
}

impl StableDiffusion {
    /// Create a new `StableDiffusion` instance from parameters.
    pub fn new(parameters: StableDiffusionParameters) -> Result<Self> {
        let device = parameters.device;
        let config = parameters.config;
        let dtype = parameters.dtype;
        let version = parameters.weights.version;
        let weights = parameters.weights;

        println!("Building the unet.");
        let unet = UNet::new(weights.unet.file.fetch()?, &config, &device, dtype)?;

        println!("Building the autoencoder.");
        let vae = VAE::new(&config, weights.vae.file.fetch()?, &device, dtype)?;
        let tokenizer = Tokenizer::new(&config, &weights.tokenizer.tokenizer.fetch()?)?;
        let clip = CLIP::new(&config.clip, weights.clip.clip.fetch()?, &device, dtype)?;
        let tokenizer_2 = if let Some(weights) = &weights.tokenizer.tokenizer2 {
            Some(Tokenizer::new(&config, weights.fetch()?)?)
        } else {
            None
        };
        let clip_2 = if let (Some(config), Some(weights)) = (&config.clip2, weights.clip.clip2) {
            Some(CLIP::new(config, weights.fetch()?, &device, dtype)?)
        } else {
            None
        };

        Ok(Self { version, device, dtype, config, unet, vae, tokenizer, clip, tokenizer_2, clip_2 })
    }

    /// Generate an image from the model.
    pub fn generate(&self, args: impl Into<GenerationParameters>) -> Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        let GenerationParameters {
            prompt,
            uncond_prompt,
            n_steps,
            guidance_scale,
            img2img,
            img2img_strength,
            width,
            height,
            style_prompt,
            uncond_style_prompt
        } = args.into();
        let width = width.unwrap_or(self.config.width);
        let height = height.unwrap_or(self.config.height);
    
        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match self.version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::XL => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match n_steps {
            Some(n_steps) => n_steps,
            None => match self.version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::XL => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };
    
        let scheduler = self.config.build_scheduler(n_steps)?;
        let use_guide_scale = guidance_scale > 1.0;
   

        let (t_start, init_latent_dist) = match img2img {
            None => (0, None),
            Some(image) => {
                let t_start = n_steps - (n_steps as f64 * img2img_strength) as usize;
                (t_start, Some(self.vae.image_to_latent(image, &self.device, self.dtype)?))
            }
        };

        let uncond_prompt = if use_guide_scale { Some(uncond_prompt.as_str()) } else { None };
        let mut text_embeddings = Vec::new();
        {
            let (prompt, uncond_prompt) = self.tokenizer.tokenize_pair(&prompt, uncond_prompt)?;
            text_embeddings.push(self.clip.text_embeddings_pair(
                prompt,
                uncond_prompt,
                &self.device,
                self.dtype
            )?);
        }
        if matches!(self.version, StableDiffusionVersion::XL | StableDiffusionVersion::Turbo) {
            let style_prompt = style_prompt.unwrap_or_default();
            let uncond_style_prompt = Some(
                uncond_style_prompt
                .as_ref().map(|s| s.as_str())
                .unwrap_or(""));
            let (prompt, uncond_prompt) = self.tokenizer_2.as_ref().unwrap().tokenize_pair(&style_prompt, uncond_style_prompt)?;
            text_embeddings.push(self.clip_2.as_ref().unwrap().text_embeddings_pair(
                prompt,
                uncond_prompt,
                &self.device,
                self.dtype
            )?);
        }

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        println!("{text_embeddings:?}");
    
        let bsize = 1;
    
        let vae_scale = match self.version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::XL => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };
    
        let timesteps = scheduler.timesteps();
        let latents = match &init_latent_dist {
            Some(init_latent_dist) => {
                let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&self.device)?;
                if t_start < timesteps.len() {
                    let noise = latents.randn_like(0f64, 1f64)?;
                    scheduler.add_noise(&latents, noise, timesteps[t_start])?
                } else {
                    latents
                }
            }
            None => {
                let latents = Tensor::randn(
                    0f32,
                    1f32,
                    (bsize, 4, height / 8, width / 8),
                    &self.device,
                )?;
                // scale the initial noise by the standard deviation required by the scheduler
                (latents * scheduler.init_noise_sigma())?
            }
        };
        let mut latents = latents.to_dtype(self.dtype)?;

        println!("starting sampling");
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            if timestep_index < t_start {
                continue;
            }
            let start_time = std::time::Instant::now();
            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                self.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guide_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);
        }
        Ok(self.vae.latent_to_image(&latents, vae_scale)?)
    }
}

/// The `StableDiffusion` struct is used to specify the Stable Diffusion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
    XL,
    Turbo,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::XL => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }
}
