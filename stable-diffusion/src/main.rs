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


#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long)]
    repository: Option<String>,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

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

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    use_f16: bool,

    #[arg(long)]
    guidance_scale: Option<f64>,

    #[arg(long, value_name = "FILE")]
    pub img2img: Option<String>,

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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;
    let parameters = StableDiffusionParameters::new(args.sd_version, args.repository, device)?;
    let stable_diffusion = StableDiffusion::new(parameters)?;
    let args = GenerationParameters::new(args.prompt)
        .with_width(args.width)
        .with_height(args.height)
        .with_uncond_prompt(args.uncond_prompt)
        .with_n_steps(args.n_steps)
        .with_img2img(args.img2img.as_ref().and_then(|path| image_preprocess(path).ok()))
        .with_img2img_strength(args.img2img_strength);
    let image = stable_diffusion.generate(args)?;
    image.save("output.png")?;
    std::process::Command::new("explorer").arg("output.png").output()?;
    Ok(())
}
