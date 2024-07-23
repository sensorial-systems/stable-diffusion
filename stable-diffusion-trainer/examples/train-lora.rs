use std::collections::HashMap;

use stable_diffusion_trainer::*;

fn main() {
    let kohya_ss = std::env::var("KOHYA_SS_PATH").expect("KOHYA_SS_PATH not set");
    let environment = Environment::new().with_kohya_ss(kohya_ss);

    let prompt = Prompt::new("whitedogbacana", "white dog");
    let image_data_set = ImageDataSet::from_dir("examples/training/lora/bacana/images");
    let data_set = TrainingDataSet::new(image_data_set);
    let output = Output::new("{prompt.instance}({prompt.class})d{network.dimension}a{network.alpha}", "examples/training/lora/bacana/output");
    let mut replace = HashMap::new();
    replace.insert("dog".to_string(), "whitedogbacana".to_string());
    let captioning = Captioning { replace, ..Default::default() };
    let workflow = Workflow::new(data_set)
        .with_training(Some(Training::new(prompt, output)))
        .with_captioning(Some(captioning));

    Trainer::new()
        .with_environment(environment)
        .start(&workflow);
}