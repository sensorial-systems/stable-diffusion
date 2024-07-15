use stable_diffusion_trainer::*;

fn main() {
    let kohya_ss = std::env::var("KOHYA_SS_PATH").expect("KOHYA_SS_PATH not set");
    let environment = Environment::new().with_kohya_ss(kohya_ss);

    let prompt = Prompt::new("bacana", "white dog");
    let image_data_set = ImageDataSet::from_dir("examples/training/lora/bacana/images");
    let data_set = TrainingDataSet::new(image_data_set);
    let output = Output::new("{prompt.instance}({prompt.class})d{network.dimension}a{network.alpha}", "examples/training/lora/bacana/output");
    let parameters = Workflow::new(prompt, data_set, output);

    Trainer::new()
        .with_environment(environment)
        .start(&parameters);
}