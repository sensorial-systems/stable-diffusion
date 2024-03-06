use candle_transformers::models::stable_diffusion::StableDiffusionConfig;

use crate::{File, StableDiffusionVersion};

pub struct TokenizerWeights {
    pub tokenizer: File,
    pub tokenizer2: Option<File>,
}

impl TokenizerWeights {
    fn tokenizer1(version: StableDiffusionVersion) -> File {
        let tokenizer_repo = match version {
            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                "openai/clip-vit-base-patch32"
            }
            StableDiffusionVersion::XL | StableDiffusionVersion::Turbo => {
                // This seems similar to the patch32 version except some very small
                // difference in the split regex.
                "openai/clip-vit-large-patch14"
            }
        };
        File::Repository(crate::Repository::new(tokenizer_repo, "tokenizer.json"))
    }

    fn tokenizer2() -> File {
        File::Repository(crate::Repository::new("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json"))
    }

    pub fn from_file(tokenizer: File, tokenizer2: Option<File>) -> Self {
        Self { tokenizer, tokenizer2 }
    }

    pub fn from_repository(version: StableDiffusionVersion) -> Self {
        let tokenizer = Self::tokenizer1(version);
        let tokenizer2 = if matches!(version, StableDiffusionVersion::XL | StableDiffusionVersion::Turbo) {
            Some(Self::tokenizer2())
        } else {
            None
        };
        Self::from_file(tokenizer, tokenizer2)
    }
}

pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    pad_id: u32,
    max_position_embeddings: usize,
}

impl Tokenizer {
    pub fn new(config: &StableDiffusionConfig, file: impl AsRef<std::path::Path>) -> anyhow::Result<Tokenizer> {
        let tokenizer = tokenizers::Tokenizer::from_file(file).map_err(anyhow::Error::msg)?;
        let pad_id = match &config.clip.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };
        let max_position_embeddings = config.clip.max_position_embeddings;
        Ok(Tokenizer { pad_id, tokenizer, max_position_embeddings })
    }

    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut tokens = self.tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        while tokens.len() < self.max_position_embeddings {
            tokens.push(self.pad_id)
        }
        Ok(tokens)
    }

    pub fn tokenize_pair(&self, prompt: &str, cond_prompt: Option<&str>) -> anyhow::Result<(Vec<u32>, Option<Vec<u32>>)> {
        let prompt = self.tokenize(prompt)?;
        let cond_prompt = match cond_prompt {
            Some(cond_prompt) => Some(self.tokenize(cond_prompt)?),
            None => None,
        };
        Ok((prompt, cond_prompt))
    }
}
