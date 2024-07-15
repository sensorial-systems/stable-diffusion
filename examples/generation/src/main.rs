use stable_diffusion::{DType, Device, GenerationParameters, StableDiffusion, StableDiffusionParameters, StableDiffusionVersion, StableDiffusionWeights};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let version = StableDiffusionVersion::XL;
    let dtype = DType::F16;
    let weights = StableDiffusionWeights::new(version, dtype);
    let device = Device::new_cuda(0)?;
    let parameters = StableDiffusionParameters::new(weights, device, dtype)?;
    let diffusion = StableDiffusion::new(parameters)?;
    let parameters = GenerationParameters::new("Dog walking in the park.")
        .with_guidance_scale(Some(5.0))
        .with_width(Some(1024))
        .with_height(Some(1024))
        .with_n_steps(Some(60));
    let output = diffusion.generate(parameters)?;
    output.save("output.png")?;
    Ok(())
}
