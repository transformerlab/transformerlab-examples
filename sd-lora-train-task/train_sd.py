#!/usr/bin/env python3
"""
Training script for Stable Diffusion XL (SDXL) using LoRA and the Simpsons dataset.
Handles SDXL's dual-encoder architecture and resolution conditioning.
"""

import os
import json
import shutil
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

# Imports
from huggingface_hub import login
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection

# --- Configuration ---

def setup_config():
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    return {
        "model_name": os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0"),
        "dataset_name": os.getenv("DATASET_NAME", "Norod78/simpsons-blip-captions"),
        "output_dir": "./output",
        "resolution": 1024,  # SDXL optimal resolution
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,  # Lower LR for SDXL
        "max_train_steps": 500, # Increased steps for real data
        "lora_rank": 8,         # Slightly higher rank for SDXL
        "seed": 42,
    }

# --- SDXL Helper Functions ---

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        return ValueError(f"{model_class} is not supported.")

def compute_embeddings(prompt_batch, tokenizers, text_encoders, device):
    """
    Computes SDXL embeddings using both text encoders.
    """
    prompt_embeds_list = []
    
    # Iterate over both tokenizers/encoders (CLIP + OpenCLIP)
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        
        # We use hidden_states[-2] for the concatenated prompt embeds
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    # Concatenate the embeddings from both encoders
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    
    # The "pooled" embedding comes only from the second encoder (OpenCLIP)
    pooled_prompt_embeds = pooled_prompt_embeds.view(len(prompt_batch), -1)
    
    return prompt_embeds, pooled_prompt_embeds

def compute_time_ids(original_size, crops_coords_top_left, target_size, device, dtype):
    """
    Creates the time IDs required for SDXL resolution conditioning.
    """
    # Simple version: Assume batch size 1 or handled by broadcaster
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids

# --- Main Training Logic ---

def train_sdxl_lora():
    config = setup_config()
    lab.init()
    lab.set_config(config)
    
    lab.log("🚀 Starting SDXL LoRA Training (Simpsons Dataset)")
    lab.log(f"Model: {config['model_name']}")
    lab.log(f"Dataset: {config['dataset_name']}")
    
    start_time = datetime.now()
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Hardware Detection & Accelerator
    mixed_precision = "fp16"
    if torch.backends.mps.is_available():
        mixed_precision = "no"
        lab.log("🍎 Mac (MPS) detected: Disabling mixed precision.")
    elif not torch.cuda.is_available():
        mixed_precision = "no"
        lab.log("⚠️ No GPU detected: Disabling mixed precision.")

    accelerator_config = ProjectConfiguration(project_dir=config["output_dir"], logging_dir=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=mixed_precision,
        project_config=accelerator_config,
        log_with="wandb" if os.getenv("WANDB_PROJECT") else None
    )
    
    set_seed(config["seed"])

    # 2. Load Models (Complex SDXL Setup)
    lab.log("Loading SDXL components (This may take a moment)...")
    try:
        # Load Tokenizers
        tokenizer_1 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer_2", use_fast=False)
        
        # Load Text Encoders
        text_encoder_1 = CLIPTextModel.from_pretrained(config["model_name"], subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config["model_name"], subfolder="text_encoder_2")
        
        # Load VAE and UNet
        vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
        
        noise_scheduler = DDPMScheduler.from_pretrained(config["model_name"], subfolder="scheduler")
        
        lab.log("✅ SDXL models loaded.")
    except Exception as e:
        lab.error(f"Failed to load models: {e}")
        return {"status": "error", "error": str(e)}

    # 3. Freeze & Configure LoRA
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 4. Dataset Loading & Processing
    lab.log(f"Loading dataset: {config['dataset_name']}...")
    try:
        dataset = load_dataset(config["dataset_name"])
        
        # Image Transforms
        train_transforms = transforms.Compose([
            transforms.Resize(config["resolution"], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config["resolution"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def preprocess(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["text"] = examples["text"] # Pass text through for on-the-fly tokenization
            return examples

        train_dataset = dataset["train"].with_transform(preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)
        
        lab.log(f"✅ Dataset loaded with {len(train_dataset)} examples.")
    except Exception as e:
        lab.error(f"Dataset error: {e}")
        return {"status": "error", "error": str(e)}

    # 5. Optimizer
    # Use 8-bit optimizer if available (Linux/CUDA only usually)
    try:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
        lab.log("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer_cls = torch.optim.AdamW
        lab.log("Using standard AdamW optimizer")

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-2,
    )

    # 6. Prepare
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Move frozen models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype) # VAE often safer in fp32, but we'll try weight_dtype

    # 7. Training Loop
    num_epochs = int(np.ceil(config["max_train_steps"] / (len(train_dataloader) // config["gradient_accumulation_steps"])))
    lab.log(f"Starting training for {config['max_train_steps']} steps...")
    
    global_step = 0
    loss_val = 0.0
    
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    for epoch in range(num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # A. Encode Images -> Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # B. Sample Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # C. Encode Text (SDXL Dual Encoder)
                # We do this inside the loop to handle the batch text
                prompt_embeds, pooled_prompt_embeds = compute_embeddings(
                    batch["text"], tokenizers, text_encoders, accelerator.device
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                # D. Compute Time IDs (Resolution Conditioning)
                # For this simple script, we assume original size = target size = resolution
                # Format: [original_height, original_width, crop_top, crop_left, target_height, target_width]
                add_time_ids = torch.cat(
                    [compute_time_ids((config["resolution"], config["resolution"]), (0, 0), (config["resolution"], config["resolution"]), accelerator.device, weight_dtype) for _ in range(bsz)]
                )

                # E. Predict
                # SDXL UNet expects 'added_cond_kwargs' for the pooled embeds and time ids
                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions
                ).sample

                # F. Loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    # SDXL default is usually epsilon or v_prediction depending on scheduler
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress = int((global_step / config["max_train_steps"]) * 100)
                lab.update_progress(min(progress, 95))
                loss_val = loss.item()
                
                if global_step % 10 == 0:
                    lab.log(f"Step {global_step} - Loss: {loss_val:.4f}")

            if global_step >= config["max_train_steps"]:
                break
        
        lab.log(f"📊 Completed epoch {epoch + 1}")

    # 8. Save Artifacts (Robust)
    lab.log("Saving final model and artifacts...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Save LoRA weights
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(config["output_dir"])

        # Save Summary JSON
        summary_file = os.path.join(config["output_dir"], "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "model": "SDXL Base 1.0",
                "dataset": config["dataset_name"],
                "steps": global_step,
                "final_loss": loss_val,
                "resolution": config["resolution"],
                "completed_at": datetime.now().isoformat()
            }, f, indent=2)
        lab.save_artifact(summary_file, "training_summary.json")

        # Save Summary TXT
        duration = datetime.now() - start_time
        txt_file = os.path.join(config["output_dir"], "final_summary.txt")
        with open(txt_file, "w") as f:
            f.write(f"SDXL LoRA Training\nDataset: {config['dataset_name']}\nDuration: {duration}\nLoss: {loss_val:.4f}")
        lab.save_artifact(txt_file, "final_summary.txt")

        # Register Model in Lab
        model_dir = os.path.join(config["output_dir"], "final_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Move safetensors/json files to model dir
        for file in os.listdir(config["output_dir"]):
            if file.endswith((".safetensors", ".json")) and not file.startswith("training_"):
                shutil.copy2(os.path.join(config["output_dir"], file), os.path.join(model_dir, file))
        
        saved_path = lab.save_model(model_dir, name="sdxl-lora-simpsons")
        lab.log(f"✅ Model registered: {saved_path}")

        # WandB finish
        try:
            import wandb
            if wandb.run is not None: wandb.finish()
        except: pass

    lab.finish("SDXL Training Complete")
    return {"status": "success", "loss": loss_val}

if __name__ == "__main__":
    train_sdxl_lora()
