[![](https://dcbadge.vercel.app/api/server/rzaesS82MT)](https://discord.gg/rzaesS82MT)

# Stable Diffusion XL LoRA Trainer

Welcome to the official codebase for the Sensorial System's Stable Diffusion projects. For now, this only hosts the codebase for our Stable Diffusion XL LoRA Trainer, designed to make it easier to automate all the steps of finetuning Stable Diffusion models.

## Features

- [**Rust Library**: For programmability, a Rust library is made available for developers.](trainer/lib/README.md)
- [**Command Line Interface (CLI)**: For ease of use, the trainer can be accessed via a CLI, making it accessible for various use cases.](trainer/cli/README.md)

## Requirements

- **kohya_ss**: Follow the installation guidelines here [https://github.com/bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss).

## Examples

We have a [dataset with photos of Bacana](examples/training/lora/bacana/images), a Coton de Tuléar, conceptualized as `bacana white dog` to not mix with the existing `Coton de Tuléar` concept in the `Stable Diffusion XL` model.

It's organized as follows:
* [Training images examples directory](examples/training/lora/bacana/images)
<p>
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5175.PNG" width="128">
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5176.PNG" width="128">
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/images/IMG_5180.PNG" width="128">
</p>

* [Generated images examples directory](examples/training/lora/bacana/generation)
<p>
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as a fireman.png" width="128" />
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as a scientist.png" width="128" />
<img src="https://raw.githubusercontent.com/sensorial-systems/stable-diffusion/main/examples/training/lora/bacana/generation/bacana as an astronaut.png" width="128" />
</p>

<details>
<summary>Rust Library</summary>

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

</details>
<details>
<summary>Stable Diffusion CLI</summary>

Install the CLI tool:
```bash
cargo install stable-diffusion-cli
```

Setup the environment:
```bash
stable-diffusion-cli setup
```

Train the example with:
```bash
stable-diffusion-cli train --config examples/training/lora/bacana/parameters.json
```


The training parameters looks like this:

```json
{
    "prompt": {
        "instance": "bacana",
        "class": "white dog"
    },
    "dataset": {
        "training": "images"
    },
    "network": {
        "dimension": 8,
        "alpha": 1.0
    },
    "output": {
        "name": "{prompt.instance}({prompt.class})d{network.dimension}a{network.alpha}",
        "directory": "./output"
    },
    "training": {
        "optimizer": "Adafactor",
        "learning_rate": {
            "scheduler": "Constant"
        }    
    }
}
```
</details>

### Understanding Stable Diffusion LoRA training

Maybe you want to understand

* [Learning rate schedulers, network dimension and alpha](https://medium.com/@dreamsarereal/understanding-lora-training-part-1-learning-rate-schedulers-network-dimension-and-alpha-c88a8658beb7)
* [Offset noise, epochs and repeats](https://medium.com/@dreamsarereal/understanding-lora-training-part-2-offset-noise-epochs-and-repeats-c68b86c69da8)
* [Block weights](https://medium.com/@dreamsarereal/understanding-lora-training-part-3-block-weights-967711816280)