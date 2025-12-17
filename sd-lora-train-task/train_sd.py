#!/usr/bin/env python3
"""
Training script for Stable Diffusion XL (SDXL) using LoRA.
Fixed: VAE is forced to float32 to prevent NaN losses.
"""

import os
import json
import shutil
import io
import random
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# TransformerLab import
from lab import lab

# Imports
from huggingface_hub import login
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

# --- Configuration ---

def setup_config():
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    return {
        "model_name": os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0"),
        "dataset_url": os.getenv("DATASET_URL", "https://huggingface.co/datasets/Norod78/simpsons-blip-captions/resolve/main/data/train-00000-of-00001.parquet"),
        "output_dir": "./output",
        "resolution": 1024,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5, # Conservative LR
        "max_train_steps": 500,
        "lora_rank": 8,
        "seed": 42,
    }

# --- Custom Dataset ---

class SimpsonsParquetDataset(Dataset):
    def __init__(self, parquet_url, resolution=1024):
        self.resolution = resolution
        print(f"📥 Downloading dataset from {parquet_url}...")
        try:
            self.df = pd.read_parquet(parquet_url)
            print(f"✅ Loaded {len(self.df)} rows.")
        except Exception as e:
            print(f"❌ Failed to load parquet: {e}")
            raise e
        
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']
        
        try:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                image_bytes = img_data['bytes']
            elif isinstance(img_data, bytes):
                image_bytes = img_data
            else:
                image_bytes = bytes(img_data)
                
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"⚠️ Error loading image at index {idx}: {e}")
            pixel_values = torch.zeros((3, self.resolution, self.resolution))
        
        text = str(row['text']) if 'text' in row else ""
        return {"pixel_values": pixel_values, "text": text}

# --- SDXL Helpers ---

def compute_embeddings(prompt_batch, tokenizers, text_encoders, device):
    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt_batch), -1)
    return prompt_embeds, pooled_prompt_embeds

def compute_time_ids(original_size, crops_coords_top_left, target_size, device, dtype):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids

# --- Main Training ---

def train_sdxl_lora():
    config = setup_config()
    lab.init()
    lab.set_config(config)
    
    lab.log("🚀 Starting SDXL LoRA Training (Fixed VAE Precision)")
    
    start_time = datetime.now()
    os.makedirs(config["output_dir"], exist_ok=True)

    mixed_precision = "fp16"
    if torch.backends.mps.is_available(): mixed_precision = "no"
    elif not torch.cuda.is_available(): mixed_precision = "no"

    accelerator_config = ProjectConfiguration(project_dir=config["output_dir"], logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=mixed_precision,
        project_config=accelerator_config,
        log_with="wandb" if os.getenv("WANDB_PROJECT") else None
    )
    set_seed(config["seed"])

    # Load Models
    lab.log("Loading SDXL components...")
    try:
        tokenizer_1 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer_2", use_fast=False)
        
        text_encoder_1 = CLIPTextModel.from_pretrained(config["model_name"], subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config["model_name"], subfolder="text_encoder_2")
        
        vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(config["model_name"], subfolder="scheduler")
        lab.log("✅ SDXL models loaded.")
    except Exception as e:
        lab.error(f"Failed to load models: {e}")
        return {"status": "error"}

    # Freeze
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_rank"],
        init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    # Dataset
    lab.log(f"Downloading dataset...")
    try:
        train_dataset = SimpsonsParquetDataset(config["dataset_url"], resolution=config["resolution"])
        train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)
    except Exception as e:
        lab.error(f"Dataset error: {e}")
        return {"status": "error"}

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        lab.log("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer_cls = torch.optim.AdamW
        lab.log("Using standard AdamW optimizer")

    optimizer = optimizer_cls(unet.parameters(), lr=config["learning_rate"], weight_decay=1e-2)

    # Prepare
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Handle Precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    # Move models to GPU
    # IMPORTANT: VAE must stay in float32 to avoid NaN
    vae.to(accelerator.device, dtype=torch.float32) 
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device) # UNet handles its own dtype via PEFT usually, but good to ensure

    # Training Loop
    steps_per_epoch = len(train_dataloader) // config["gradient_accumulation_steps"]
    num_epochs = int(np.ceil(config["max_train_steps"] / max(1, steps_per_epoch)))
    
    lab.log(f"Starting training for {config['max_train_steps']} steps...")
    global_step = 0
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    for epoch in range(num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Encode Images -> Latents (Force FP32 for VAE)
                # We cast pixels to float32 for the VAE
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Cast latents back to weight_dtype (fp16) for the UNet
                latents = latents.to(dtype=weight_dtype)

                # 2. Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Text Embeds
                prompt_embeds, pooled_prompt_embeds = compute_embeddings(
                    batch["text"], tokenizers, text_encoders, accelerator.device
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                # 4. Time IDs
                add_time_ids = torch.cat(
                    [compute_time_ids((config["resolution"], config["resolution"]), (0, 0), (config["resolution"], config["resolution"]), accelerator.device, weight_dtype) for _ in range(bsz)]
                )

                # 5. Predict
                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample

                if noise_scheduler.config.prediction_type == "epsilon": target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction": target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else: target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Check for NaN
                if torch.isnan(loss):
                    print("⚠️ Loss is NaN. Skipping step.")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                
                # Gradient Clipping (Crucial for stability)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress = int((global_step / config["max_train_steps"]) * 100)
                lab.update_progress(min(progress, 95))
                
                if global_step % 10 == 0:
                    lab.log(f"Step {global_step} - Loss: {loss.item():.4f}")

            if global_step >= config["max_train_steps"]:
                break
        
        lab.log(f"📊 Completed epoch {epoch + 1}")

    # Save
    lab.log("Saving final model...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(config["output_dir"])

        summary_file = os.path.join(config["output_dir"], "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "model": "SDXL Base 1.0",
                "steps": global_step,
                "final_loss": loss.item() if not torch.isnan(loss) else "NaN",
                "completed_at": datetime.now().isoformat()
            }, f, indent=2)
        lab.save_artifact(summary_file, "training_summary.json")

        # Save Final Model
        model_dir = os.path.join(config["output_dir"], "final_model")
        os.makedirs(model_dir, exist_ok=True)
        for file in os.listdir(config["output_dir"]):
            if file.endswith((".safetensors", ".json")) and not file.startswith("training_"):
                shutil.copy2(os.path.join(config["output_dir"], file), os.path.join(model_dir, file))
        
        saved_path = lab.save_model(model_dir, name="sdxl-lora-simpsons")
        lab.log(f"✅ Model registered: {saved_path}")
        
        try:
            import wandb
            if wandb.run is not None: wandb.finish()
        except: pass

    lab.finish("SDXL Training Complete")
    return {"status": "success"}

if __name__ == "__main__":
    train_sdxl_lora()