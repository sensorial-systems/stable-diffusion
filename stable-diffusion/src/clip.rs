use candle::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion::{self, clip::{self, ClipTextTransformer}};

use crate::{File, StableDiffusionVersion};

pub struct CLIPWeights {
    pub clip: File,
    pub clip2: Option<File>
}

impl CLIPWeights {
    fn clip_file(dtype: DType) -> &'static str {
        if dtype == DType::F16 {
            "text_encoder/model.fp16.safetensors"
        } else {
            "text_encoder/model.safetensors"
        }
    }

    fn clip2_file(dtype: DType) -> &'static str {
        if dtype == DType::F16 {
            "text_encoder_2/model.fp16.safetensors"
        } else {
            "text_encoder_2/model.safetensors"
        }
    }

    pub fn new(repository: impl Into<String>, version: StableDiffusionVersion, dtype: DType) -> anyhow::Result<Self> {
        let repo = repository.into();
        let filename = Self::clip_file(dtype);
        let clip = File::Repository(crate::Repository::new(repo.clone(), filename));
        let clip2 = if matches!(version, StableDiffusionVersion::XL | StableDiffusionVersion::Turbo) {
            let filename = Self::clip2_file(dtype);
            let clip2 = File::Repository(crate::Repository::new(repo, filename));
            Some(clip2)
        } else {
            None
        };
        Ok(Self { clip, clip2 })
    }
}


pub struct CLIP {
    clip: ClipTextTransformer
}

impl CLIP {
    pub fn new(config: &clip::Config, weights: impl AsRef<std::path::Path>, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        let clip = stable_diffusion::build_clip_transformer(config, weights, device, dtype)?;
        Ok(Self { clip })
    }

    pub fn text_embeddings(&self, prompt_tokens: impl AsRef<[u32]>, device: &Device, dtype: DType) -> anyhow::Result<Tensor> {
        let tokens = Tensor::new(prompt_tokens.as_ref(), device)?.unsqueeze(0)?;
        let text_embeddings = self.clip.forward(&tokens)?;
        Ok(text_embeddings.to_dtype(dtype)?)
    }

    pub fn text_embeddings_pair(&self, prompt_tokens: impl AsRef<[u32]>, uncond_prompt: Option<impl AsRef<[u32]>>, device: &Device, dtype: DType) -> anyhow::Result<Tensor> {
        let tokens = Tensor::new(prompt_tokens.as_ref(), device)?.unsqueeze(0)?;
        let text_embeddings = self.clip.forward(&tokens)?;

        let text_embeddings = if let Some(uncond_tokens) = uncond_prompt {
            let uncond_tokens = Tensor::new(uncond_tokens.as_ref(), device)?.unsqueeze(0)?;
            let uncond_embeddings = self.clip.forward(&uncond_tokens)?;
            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
        } else {
            text_embeddings.to_dtype(dtype)?
        };
        Ok(text_embeddings)
    }
}