#!/usr/bin/env python3
"""
Training script for Stable Diffusion XL (SDXL) using LoRA.
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

from lab import lab

from huggingface_hub import login
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

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
        "learning_rate": 1e-5, 
        "max_train_steps": 300,
        "lora_rank": 8,
        "seed": 42,
        "validation_prompt": "A single-family home in a typical American suburb in the 1990s",
    }

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

def generate_and_save_sample(accelerator, config, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, vae, unet, scheduler, filename, label):
    """Generates an image and saves it as a lab artifact."""
    lab.log(f"Generating {label} image...")
    
    # Create pipeline from components
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
        force_zeros_for_empty_prompt=True,
    )
    pipeline.to(accelerator.device)
    
    # Use standard inference settings
    with torch.no_grad():
        image = pipeline(
            prompt=config["validation_prompt"],
            num_inference_steps=30,
            generator=torch.Generator(device=accelerator.device).manual_seed(config["seed"]),
        ).images[0]
    
    # Save locally and then as artifact
    path = os.path.join(config["output_dir"], filename)
    image.save(path)
    lab.save_artifact(path, filename)
    lab.log(f"✅ {label} image saved to {filename}")
    
    # Clean up to save VRAM (pipeline shares components, so just delete the wrapper)
    del pipeline
    torch.cuda.empty_cache()


def train_sdxl_lora():
    config = setup_config()
    lab.init()
    lab.set_config(config)
    
    lab.log("🚀 Starting SDXL LoRA Training")
    
    os.makedirs(config["output_dir"], exist_ok=True)

    mixed_precision = "fp16"
    if torch.backends.mps.is_available(): mixed_precision = "no"
    elif not torch.cuda.is_available(): mixed_precision = "no"

    accelerator_config = ProjectConfiguration(project_dir=config["output_dir"], logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=mixed_precision,
        project_config=accelerator_config,
    )
    set_seed(config["seed"])

    # Load Models
    lab.log("Loading SDXL components...")
    tokenizer_1 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer_2", use_fast=False)
    text_encoder_1 = CLIPTextModel.from_pretrained(config["model_name"], subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config["model_name"], subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config["model_name"], subfolder="scheduler")

    # Generate BEFORE image (using the base model)
    if accelerator.is_main_process:
        generate_and_save_sample(
            accelerator, config, tokenizer_1, tokenizer_2, 
            text_encoder_1, text_encoder_2, vae, unet, 
            noise_scheduler, "before_finetuning.png", "Before Fine-Tuning"
        )

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_rank"],
        init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    # Dataset
    train_dataset = SimpsonsParquetDataset(config["dataset_url"], resolution=config["resolution"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])

    # Prepare for accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Precision Handling
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32) # VAE always float32
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Training Loop
    global_step = 0
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    lab.log(f"Starting training...")
    while global_step < config["max_train_steps"]:
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Encode Latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                # Noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Embeddings
                prompt_embeds, pooled_prompt_embeds = compute_embeddings(batch["text"], tokenizers, text_encoders, accelerator.device)
                add_time_ids = torch.cat([compute_time_ids((config["resolution"], config["resolution"]), (0, 0), (config["resolution"], config["resolution"]), accelerator.device, weight_dtype)])

                # Predict
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                lab.update_progress(int((global_step / config["max_train_steps"]) * 100))
            
            if global_step >= config["max_train_steps"]: break

    # Generate AFTER image
    if accelerator.is_main_process:
        generate_and_save_sample(
            accelerator, config, tokenizer_1, tokenizer_2, 
            text_encoder_1, text_encoder_2, vae, unet, 
            noise_scheduler, "after_finetuning.png", "After Fine-Tuning"
        )

    # Save final model
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(config["output_dir"])
        lab.save_model(config["output_dir"], name="sdxl-lora-simpsons")

    lab.finish("SDXL Training Complete")

if __name__ == "__main__":
    train_sdxl_lora()
