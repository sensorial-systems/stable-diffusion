[![](https://dcbadge.vercel.app/api/server/rzaesS82MT)](https://discord.gg/rzaesS82MT)

**Disclaimer** Stable Diffusion is a trademark owned by Stability AI. Original repos: [Stable Diffusion 1.5](https://github.com/runwayml/stable-diffusion), [Stable Diffusion 2.1](https://github.com/Stability-AI/stablediffusion), [Stable Diffusion XL and XL-Turbo](https://github.com/Stability-AI/stablediffusion)

# Stable Diffusion

Welcome to the official codebase for the Sensorial System's Stable Diffusion projects.

## Features

- **Inference**: Stable Diffusion 1.5, 2.1, XL and Turbo inferences.
- **Training**: Stable Diffusion XL LoRA training.

## Sub-projects

- [Stable Diffusion](stable-diffusion/README.md): Library core.
- [Stable Diffusion Trainer](stable-diffusion-trainer/README.md): Trainer library.
- [Stable Diffusion CLI](cli/README.md): CLI for image generation and model training.

## Examples

#### Image generation

```rust
use candle::Device;
use stable_diffusion::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let weights = StableDiffusionWeights::new(StableDiffusionVersion::XL, DType::F32);
    let parameters = StableDiffusionParameters::new(weights, device, DType::F16)?;
    let stable_diffusion = StableDiffusion::new(parameters)?;
    let args = GenerationParameters::new("A green apple");
    let image = stable_diffusion.generate(args)?;
    image.save("output.png")?;
    Ok(())
}
```

#### XL LoRA training

```rust
use stable_diffusion_trainer::*;

fn main() {
    let kohya_ss = std::env::var("KOHYA_SS_PATH").expect("KOHYA_SS_PATH not set");
    let environment = Environment::new().with_kohya_ss(kohya_ss);

    let prompt = Prompt::new("bacana", "white dog");
    let image_data_set = ImageDataSet::from_dir("examples/training/lora/bacana/images");
    let data_set = TrainingDataSet::new(image_data_set);
    let output = Output::new("{prompt.instance}({prompt.class})d{network.dimension}a{network.alpha}", "examples/training/lora/bacana/output");
    let parameters = Parameters::new(prompt, data_set, output);

    Trainer::new()
        .with_environment(environment)
        .start(&parameters);
}
```
