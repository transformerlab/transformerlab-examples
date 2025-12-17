#!/usr/bin/env python3
"""
Training script for Stable Diffusion using Diffusers and LoRA.
Demonstrates setting up a custom training loop with Accelerator and TransformerLab integration.
"""

import os
import json
import random
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# TransformerLab import
from lab import lab

# Hugging Face imports
from huggingface_hub import login
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

# --- Configuration & Setup ---

def setup_environment():
    """Handle logins and env vars"""
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    # Default config
    return {
        "model_name": os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5"),
        "output_dir": "./output",
        "instance_prompt": "a photo of sks dog",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_train_steps": 100,
        "lora_rank": 4,
        "seed": 42,
    }

# --- Dummy Dataset Generation ---

class DummyDataset(Dataset):
    """
    Generates random noise images to allow the script to run 
    immediately without external data dependencies.
    """
    def __init__(self, size=10, resolution=512, tokenizer=None, prompt=""):
        self.size = size
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Generate fake images
        self.images = []
        for _ in range(size):
            # Create a random colored image
            img = Image.fromarray(np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8))
            self.images.append(img)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.images[idx]
        example = {}
        example["pixel_values"] = self.transforms(image)
        
        # Tokenize the prompt
        if self.tokenizer:
            inputs = self.tokenizer(
                self.prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            example["input_ids"] = inputs.input_ids[0]
            
        return example

# --- Main Training Logic ---

def train_sd_lora():
    config = setup_environment()
    
    # Initialize Lab
    lab.init()
    lab.set_config(config)
    
    lab.log(f"🚀 Starting Stable Diffusion LoRA Training")
    lab.log(f"Model: {config['model_name']}")
    
    start_time = datetime.now()
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Initialize Accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=config["output_dir"], logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="fp16",
        project_config=accelerator_project_config,
        log_with="wandb" if os.getenv("WANDB_PROJECT") else None
    )
    
    set_seed(config["seed"])

    # 2. Load Models
    lab.log("Loading models (Tokenizer, Noise Scheduler, UNet, VAE)...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer")
        noise_scheduler = DDPMScheduler.from_pretrained(config["model_name"], subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(config["model_name"], subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
        lab.log("✅ Base models loaded.")
    except Exception as e:
        lab.error(f"Failed to load models: {e}")
        return {"status": "error", "error": str(e)}

    # 3. Freeze params & Add LoRA
    lab.log("Setting up LoRA configuration...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Configure LoRA for UNet
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # 4. Optimizer & Data
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Create dataset
    lab.log("Creating dataset...")
    train_dataset = DummyDataset(
        size=20, 
        resolution=config["resolution"], 
        tokenizer=tokenizer, 
        prompt=config["instance_prompt"]
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"], 
        shuffle=True, 
        num_workers=0
    )

    # 5. Prepare with Accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    # Move frozen models to GPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 6. Training Loop
    num_update_steps_per_epoch = len(train_dataloader) // config["gradient_accumulation_steps"]
    num_epochs = int(np.ceil(config["max_train_steps"] / num_update_steps_per_epoch))
    
    lab.log(f"Starting training for {config['max_train_steps']} steps ({num_epochs} epochs)...")
    lab.update_progress(0)

    global_step = 0
    
    for epoch in range(num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents (forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                
                # Update progress in Lab
                progress = int((global_step / config["max_train_steps"]) * 100)
                lab.update_progress(min(progress, 95))
                
                if global_step % 10 == 0:
                    lab.log(f"Step {global_step}/{config['max_train_steps']} - Loss: {loss.item():.4f}")

            if global_step >= config["max_train_steps"]:
                break
        
        lab.log(f"📊 Completed epoch {epoch + 1}")

    # 7. Save Model
    lab.log("Saving LoRA weights...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        # Save PEFT adapter
        unet.save_pretrained(config["output_dir"])
        
        # Save a summary
        summary_path = os.path.join(config["output_dir"], "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Model: {config['model_name']}\n")
            f.write(f"Steps: {global_step}\n")
            f.write(f"Resolution: {config['resolution']}\n")
        
        lab.save_artifact(summary_path, "training_summary.txt")
        
        # Save to lab model registry
        saved_path = lab.save_model(config["output_dir"], name="sd-lora-model")
        lab.log(f"✅ Model saved to TransformerLab: {saved_path}")

    # 8. Cleanup
    end_time = datetime.now()
    duration = end_time - start_time
    lab.log(f"Training finished in {duration}")
    lab.update_progress(100)
    
    # Try to clean up wandb
    if accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None: 
                lab.log(f"Wandb URL: {wandb.run.get_url()}")
                wandb.finish()
        except:
            pass

    lab.finish("SD LoRA Training Complete")
    
    return {
        "status": "success",
        "output_dir": config["output_dir"],
        "duration": str(duration)
    }

if __name__ == "__main__":
    try:
        train_sd_lora()
    except KeyboardInterrupt:
        lab.error("Training stopped by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        lab.error(f"Training failed: {e}")