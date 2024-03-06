use std::path::Path;

use candle::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::{unet_2d::UNet2DConditionModel, StableDiffusionConfig};

use crate::File;

pub struct UNetWeights {
    pub file: File,
}

impl UNetWeights {
    pub fn from_file(file: impl Into<File>) -> Self {
        let file = file.into();
        Self { file }
    }

    fn default_path(dtype: DType) -> &'static str {
        if dtype == DType::F16 {
            "unet/diffusion_pytorch_model.fp16.safetensors"
        } else {
            "unet/diffusion_pytorch_model.safetensors"
        }
    }

    pub fn from_repository(repository: impl Into<String>, dtype: DType) -> Self {
        let path = Self::default_path(dtype);
        let file = File::Repository(crate::Repository::new(repository.into(), path));
        Self::from_file(file)
    }
}


pub struct UNet {
    unet: UNet2DConditionModel
}

impl UNet {
    pub fn new(weights: impl AsRef<Path>, config: &StableDiffusionConfig, device: &Device, dtype: DType) -> candle::Result<Self> {
        let use_flash_attention = false;
        let unet = config.build_unet(weights, &device, 4, use_flash_attention, dtype)?;
        Ok(Self { unet })
    }

    pub fn forward(&self, latent: &Tensor, timestep: f64, text_embeddings: &Tensor) -> candle::Result<Tensor> {
        self.unet.forward(latent, timestep, &text_embeddings)
    }
}