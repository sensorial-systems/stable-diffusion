[![](https://dcbadge.vercel.app/api/server/rzaesS82MT)](https://discord.gg/rzaesS82MT)

# Stable Diffusion XL LoRA Trainer

Welcome to the official codebase for the Sensorial System's Stable Diffusion projects. For now, this only hosts the codebase for our Stable Diffusion XL LoRA Trainer, designed to make it easier to automate all the steps of finetuning Stable Diffusion models.

## Requirements

- **kohya_ss**: Follow the installation guidelines here [https://github.com/bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss).

## Examples

We have a [dataset with photos of Bacana](examples/training/lora/bacana/images), a Coton de Tuléar, conceptualized as `bacana white dog` to not mix with the existing `Coton de Tuléar` concept in the `Stable Diffusion XL` model.

Some of the training images in [examples/training/lora/bacana/images](examples/training/lora/bacana/images):
<p>
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5175.PNG" width="128">
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5176.PNG" width="128">
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5180.PNG" width="128">
</p>

The training code example looks like this:

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

Note that the `Output::name` is a format string that captures the parameters values. This is useful for experimenting with different parameters and keeping track of them in the model file name.

Train the example with:
```bash
KOHYA_SS_PATH=<your kohya_ss path here> cargo run --example train-lora
```

The LoRA safetensor file will be generated as
```bash
examples/training/lora/bacana/output/bacana(white dog)d8a1-000001.safetensors
examples/training/lora/bacana/output/bacana(white dog)d8a1.safetensors
```

Where, in this case, `bacana(white dog)d8a1-000001.safetensors` is the first epoch and `bacana(white dog)d8a1.safetensors` is the final epoch.

You can then
```bash
cd examples/training/lora/bacana/generation
```
and run
```bash
python generate.py
```
to test image generation with the LoRA model. The generated images will be present in [examples/training/lora/bacana/generation](examples/training/lora/bacana/generation).

Some of the generated images:
<p>
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as a fireman.png" width="128" />
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as a scientist.png" width="128" />
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as an astronaut.png" width="128" />
</p>
