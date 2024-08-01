use stable_diffusion::{
    DType, Device, GenerationParameters, StableDiffusion, StableDiffusionParameters,
    StableDiffusionVersion, StableDiffusionWeights,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let version = StableDiffusionVersion::XL;
    let dtype = DType::F16;
    let weights = StableDiffusionWeights::new(version, dtype);
    let device = select_device()?;
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

fn select_device() -> Result<Device, Box<dyn std::error::Error>> {
    return if cfg!(feature = "metal") {
        Device::new_metal(0).map_err(|err| err.into())
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).map_err(|err| err.into())
    } else {
        Ok(Device::Cpu)
    };
}
