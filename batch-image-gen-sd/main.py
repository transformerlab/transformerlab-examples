import os
from time import perf_counter
from datetime import datetime

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login

from lab import lab

if os.getenv("HF_API_TOKEN"):
    token = os.getenv("HF_API_TOKEN")
    login(token=token)

def slugify(s: str, maxlen: int = 64) -> str:
    keep = []
    for ch in s.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep)[:maxlen].rstrip("_")


def main():
    lab.init()
    config = lab.get_config()

    model_name = config.get("model_name", "stabilityai/stable-diffusion-3.5-medium")
    prompts = config.get(
        "prompts",
        [
            "A fluffy orange cat sitting on a windowsill",
            "A serene mountain lake at sunrise",
            "A futuristic city skyline at night with neon lights",
        ],
    )
    output_dir = os.path.expanduser(config.get("output_dir", "./generated-images"))
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab.log(f"Loading pipeline: {model_name} (device={device})")
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        # let accelerate/device_map place weights automatically
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=load_dtype, device_map="cuda")
    else:
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=load_dtype).to(device)

    num_inference_steps = int(config.get("num_inference_steps", 25))
    guidance_scale = float(config.get("guidance_scale", 7.5))
    height = int(config.get("height", 512))
    width = int(config.get("width", 512))

    results = []
    total = len(prompts)
    for i, prompt in enumerate(prompts, start=1):
        safe_name = f"{i:02d}_{slugify(prompt)}.png"
        out_path = os.path.join(output_dir, safe_name)

        lab.log(f"[{i}/{total}] Generating image for prompt: {prompt}")
        start = perf_counter()
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            ).images[0]
        duration = perf_counter() - start

        image.save(out_path)
        artifact = lab.save_artifact(out_path, name=os.path.basename(out_path), type="image")
        lab.log(f"Saved image artifact: {artifact}")

        results.append({"prompt": prompt, "output": out_path, "artifact": artifact, "duration_s": duration})
        lab.update_progress(int((i / total) * 100))

    end_time = datetime.now()
    lab.log(f"Batch generation completed at {end_time}")
    lab.finish("Batch Stable Diffusion generation finished")

    return {"status": "success", "generated": results, "output_dir": output_dir}


if __name__ == "__main__":
    res = main()
    print(res)