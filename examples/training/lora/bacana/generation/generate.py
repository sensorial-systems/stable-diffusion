import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, DPMSolverMultistepScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("../output", weight_name="bacana(white dog)d8a1.safetensors")
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

pipe = pipe.to(device)

professions = ["police officer", "fireman", "astronaut", "doctor", "scientist", "chef", "sailor", "aviator"]

for profession in professions:
    generator = torch.Generator(device).manual_seed(32)
    if profession.startswith("a"):
        profession = f"an {profession}"
    else: 
        profession = f"a {profession}"
    prompt = f"bacana white dog as {profession}"
    image = pipe(prompt, generator=generator, num_inference_steps=20, cross_attention_kwargs={"scale":0.5}).images[0]
    output = f"bacana as {profession}.png"
    print("Saving", output)
    image.save(output)