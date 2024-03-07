//! Variational Autoencoder (VAE) for Stable Diffusion models.

use candle_transformers::models::stable_diffusion::{
    vae::{AutoEncoderKL, DiagonalGaussianDistribution}, StableDiffusionConfig
};
use candle::{DType, Device, Tensor, IndexOp};

use crate::{File, StableDiffusionVersion};

/// The `VAEWeights` struct is used to specify the weights of the Variational Autoencoder (VAE) model.
pub struct VAEWeights {
    pub file: File,
}

impl VAEWeights {
    /// Create a new `VAEWeights` instance from a file.
    pub fn from_file(file: File) -> Self {
        Self { file }
    }

    /// Create a new `VAEWeights` instance from a repository.
    pub fn from_repository(repository: impl Into<String>, version: StableDiffusionVersion, dtype: DType) -> Self {
        let use_f16 = dtype == DType::F16;
        let repository = repository.into();
        let (repo, filename) = if use_f16 {
            if matches!(version, StableDiffusionVersion::XL | StableDiffusionVersion::Turbo) {
                let repo = "madebyollin/sdxl-vae-fp16-fix";
                let filename = "diffusion_pytorch_model.safetensors";
                (repo, filename)
            } else {
                let repo = repository.as_str();
                let filename = "vae/diffusion_pytorch_model.fp16.safetensors";
                (repo, filename)
            }
        } else {
            let repo = repository.as_str();
            let filename = "vae/diffusion_pytorch_model.safetensors";
            (repo, filename)
        };
        let file = File::Repository(crate::Repository::new(repo, filename));
        Self::from_file(file)
    }

}

/// The `VAE` struct is used to specify the Variational Autoencoder (VAE) model.
pub struct VAE {
    vae: AutoEncoderKL
}

impl VAE {
    /// Create a new `VAE` instance from a configuration, weights, device, and data type.
    pub fn new(config: &StableDiffusionConfig, vae_weights: impl AsRef<std::path::Path>, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        
        let vae = config.build_vae(vae_weights, &device, dtype)?;
        Ok(Self { vae })
    }

    /// Encode an image into a latent distribution.
    pub fn image_to_latent(&self, image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>, device: &Device, dtype: DType) -> candle::Result<DiagonalGaussianDistribution> {
        let (height, width) = (image.height() as usize, image.width() as usize);
        let image = image.into_raw();
        let tensor = Tensor::from_vec(image, (height, width, 3), device)?
            .permute((2, 0, 1))?
            .to_dtype(dtype)?
            .affine(2. / 255., -1.)?
            .unsqueeze(0)?;
        Ok(self.encode(&tensor)?)
    }

    /// Decode a latent distribution into an image.
    pub fn latent_to_image(&self, latents: &Tensor, vae_scale: f64) -> candle::Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        let image = self.vae.decode(&(latents / vae_scale)?)?;
        let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
        let (channel, height, width) = image.dims3()?;
        if channel != 3 {
            candle::bail!("save_image expects an input of shape (3, height, width)")
        }
        let image = image.permute((1, 2, 0))?.flatten_all()?;
        let pixels = image.to_vec1::<u8>()?;
        let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                Some(image) => image,
                None => candle::bail!("error saving image"),
            };
        Ok(image)
    }

    /// Encode a tensor into a latent distribution.
    pub fn encode(&self, tensor: &Tensor) -> candle::Result<DiagonalGaussianDistribution> {
        Ok(self.vae.encode(tensor)?)
    }

    /// Decode a latent distribution into a tensor.
    pub fn decode(&self, tensor: &Tensor) -> candle::Result<Tensor> {
        Ok(self.vae.decode(tensor)?)
    }
}