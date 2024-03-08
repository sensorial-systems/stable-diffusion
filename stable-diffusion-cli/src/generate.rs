use clap::Parser;
use stable_diffusion::*;
use candle::{Device, Result};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}


#[derive(Debug, Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Arguments {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// The output file to save the generated image to.
    #[arg(long)]
    output: Option<String>,
    
    /// The repository to download the weights from.
    #[arg(long)]
    repository: Option<String>,

    /// The unconditional prompt to be used for image generation.
    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// The style prompt to be used for image generation.
    #[arg(long)]
    style_prompt: Option<String>,

    /// The unconditional style prompt to be used for image generation.
    #[arg(long)]
    uncond_style_prompt: Option<String>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long)]
    n_steps: Option<usize>,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    /// The version of the stable diffusion model to use.
    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// The scale of the guidance image.
    #[arg(long)]
    guidance_scale: Option<f64>,

    /// The image to image transformation to apply.
    #[arg(long, value_name = "FILE")]
    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[arg(long, default_value_t = 0.8)]
    img2img_strength: f64,
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    Ok(img.to_rgb8())
}

fn view(output: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer").arg(output).output()?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(output).output()?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(output).output()?;
    Ok(())
}

/// The `StableDiffusion` struct is used to specify the Stable Diffusion model.
#[derive(Debug, Clone, clap::ValueEnum, Copy, PartialEq, Eq)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
    XL,
    Turbo,
}

impl From<StableDiffusionVersion> for stable_diffusion::StableDiffusionVersion {
    fn from(version: StableDiffusionVersion) -> Self {
        match version {
            StableDiffusionVersion::V1_5 => stable_diffusion::StableDiffusionVersion::V1_5,
            StableDiffusionVersion::V2_1 => stable_diffusion::StableDiffusionVersion::V2_1,
            StableDiffusionVersion::XL => stable_diffusion::StableDiffusionVersion::XL,
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionVersion::Turbo,
        }
    }

}

impl Arguments {
    pub fn execute(self) -> anyhow::Result<()> {
        let args = self;
        let device = device(args.cpu)?;
        let output = args.output.as_ref().map(String::from).unwrap_or(String::from("output.png"));
        let weights = StableDiffusionWeights::from_repository(args.sd_version.into(), args.repository, DType::F32);
        let parameters = StableDiffusionParameters::new(weights, device, DType::F16)?;
        let stable_diffusion = StableDiffusion::new(parameters)?;
        let args = GenerationParameters::new(args.prompt.clone())
            .with_width(args.width)
            .with_height(args.height)
            .with_uncond_prompt(args.uncond_prompt)
            .with_style_prompt(Some(args.prompt))
            .with_uncond_style_prompt(args.uncond_style_prompt)
            .with_n_steps(args.n_steps)
            .with_guidance_scale(args.guidance_scale)
            .with_img2img(args.img2img.as_ref().and_then(|path| image_preprocess(path).ok()))
            .with_img2img_strength(args.img2img_strength);
        let image = stable_diffusion.generate(args)?;
        image.save(&output)?;
        view(&output)?;
        Ok(())
    }
}