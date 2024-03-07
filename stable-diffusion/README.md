[![](https://dcbadge.vercel.app/api/server/rzaesS82MT)](https://discord.gg/rzaesS82MT)

# Stable Diffusion

This is the Stable Diffusion core library.

## Supported Versions

* Stable Diffusion 1.5
* Stable Diffusion 2.0
* Stable Diffusion XL
* Stable Diffusion Turbo

## Backends

* Cuda (default rust-feature)
* Metal
* Onnx
* CPU

## Examples

#### Image generation

```rust
use candle::Device;
use stable_diffusion::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;
    let weights = StableDiffusionWeights::new(StableDiffusionVersion::XL, DType::F32);
    let parameters = StableDiffusionParameters::new(StableDiffusionVersion::XL, weights, device, DType::BF16)?;
    let stable_diffusion = StableDiffusion::new(parameters)?;
    let args = GenerationParameters::new("A green apple");
    let image = stable_diffusion.generate(args)?;
    image.save("output.png")?;
    Ok(())
}
```
