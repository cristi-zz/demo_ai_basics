import torch
from diffusers import DiffusionPipeline

# Model card: https://huggingface.co/runwayml/stable-diffusion-v1-5
# More info: https://huggingface.co/blog/stable_diffusion

# This code runs on 6GB GPU.
# float16 is not available for CPU

prompt = "A red flower in a pot. The flower has bright green leafs. The pot is from glass."
device = "cuda"     # for FP16, it fits in < 6G ram
# device = "cpu"    # Uncomment to run on cpu. WARNING! It can take a looong time!!!

if device == "cuda":
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                 revision="fp16", torch_dtype=torch.float16)
else:
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

pipeline.to(device)
generator = torch.Generator(device).manual_seed(3213)

generator_output = pipeline(prompt, guidance_scale=4.5, num_inference_steps=5, generator=generator)
image = generator_output.images[0]
image.save("../test/images/local_image_gen1.jpg")

generator_output = pipeline(prompt, guidance_scale=4.5, num_inference_steps=22, generator=generator)
image = generator_output.images[0]
image.save("../test/images/local_image_gen2.jpg")

generator_output = pipeline(prompt, guidance_scale=4.5, num_inference_steps=50, generator=generator)
image = generator_output.images[0]
image.save("../test/images/local_image_gen3.jpg")
