#!/usr/bin/env python3
"""
Image Generation script using Stable Diffusion XL.
Generates images based on a text prompt and saves them as artifacts.
"""

import os
import json
import torch
from datetime import datetime
from diffusers import DiffusionPipeline, AutoencoderKL

# TransformerLab import
from lab import lab

# Hugging Face login
from huggingface_hub import login

def setup_environment():
    """Handle logins and configuration"""
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    return {
        "model_name": os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0"),
        "prompt": os.getenv("PROMPT", "A black cat"),
        "negative_prompt": os.getenv("NEGATIVE_PROMPT", "blurry, low quality"),
        "num_images": int(os.getenv("NUM_IMAGES", "4")),
        "steps": int(os.getenv("STEPS", "30")),
        "output_dir": "./output",
        "seed": 42
    }

def generate_images():
    config = setup_environment()
    
    # Initialize Lab
    lab.init()
    lab.set_config(config)
    
    lab.log("🚀 Starting SDXL Image Generation")
    lab.log(f"Prompt: {config['prompt']}")
    
    start_time = datetime.now()
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 1. Hardware Detection
    dtype = torch.float16
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32 # MPS often prefers fp32
        lab.log("🍎 Mac (MPS) detected. Using fp32.")
    elif torch.cuda.is_available():
        device = "cuda"
        lab.log("⚡ NVIDIA GPU detected. Using fp16.")
    else:
        device = "cpu"
        dtype = torch.float32
        lab.log("⚠️ No GPU detected. Using CPU (this will be slow).")

    try:
        # 2. Load Model
        lab.log(f"Loading model: {config['model_name']}...")
        
        # Load VAE separately to force fp32 if needed (prevents black image bugs in SDXL)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        pipe = DiffusionPipeline.from_pretrained(
            config["model_name"],
            vae=vae,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        )
        pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            # pipe.enable_model_cpu_offload() # Uncomment if VRAM < 16GB
            pass
            
        lab.log("✅ Model loaded.")
        lab.update_progress(10)

        # 3. Generate Images
        generated_paths = []
        
        lab.log(f"Generating {config['num_images']} images...")
        
        for i in range(config["num_images"]):
            # Set a different seed for each image
            generator = torch.Generator(device=device).manual_seed(config["seed"] + i)
            
            lab.log(f"🎨 Generating image {i+1}/{config['num_images']}...")
            
            image = pipe(
                prompt=config["prompt"],
                negative_prompt=config["negative_prompt"],
                num_inference_steps=config["steps"],
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            # Save Image Locally
            filename = f"black_cat_{i+1}_{int(datetime.now().timestamp())}.png"
            filepath = os.path.join(config["output_dir"], filename)
            image.save(filepath)
            
            # Save as Artifact in Lab
            artifact_path = lab.save_artifact(filepath, filename)
            generated_paths.append(artifact_path)
            
            lab.log(f"✅ Saved: {filename}")
            
            # Update progress based on images done
            progress = 10 + int((i + 1) / config["num_images"] * 80)
            lab.update_progress(progress)

        # 4. Create Summary Gallery (HTML)
        # This helps view all images at once in the artifact viewer
        html_content = f"""
        <html>
        <head><title>Generation Results</title></head>
        <body style="font-family: sans-serif; background: #111; color: #fff; padding: 20px;">
            <h1>Generated Images</h1>
            <p><strong>Prompt:</strong> {config['prompt']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """
        
        for p in generated_paths:
            # We reference the filename, assuming relative paths in artifact storage
            # (Note: In some viewers, this might strictly need the artifact URL, 
            # but usually saving the summary in the same folder works).
            fname = os.path.basename(p)
            html_content += f'<div style="margin: 10px;"><img src="{fname}" width="512" style="border-radius: 8px;"><br>{fname}</div>'
            
        html_content += "</div></body></html>"
        
        summary_path = os.path.join(config["output_dir"], "gallery.html")
        with open(summary_path, "w") as f:
            f.write(html_content)
            
        lab.save_artifact(summary_path, "gallery.html")
        lab.update_progress(100)
        
        duration = datetime.now() - start_time
        lab.finish(f"Generated {config['num_images']} images in {duration}")
        
        return {
            "status": "success",
            "count": config["num_images"],
            "artifacts": generated_paths
        }

    except Exception as e:
        lab.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    generate_images()