import math
import random
import json
import gc
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers

# Try to import xformers for memory optimization
try:
    import xformers  # noqa: F401
    xformers_available = True
except ImportError:
    xformers_available = False

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def cleanup_pipeline():
    """Clean up pipeline to free VRAM"""
    try:
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to cleanup pipeline: {str(e)}")


cleanup_pipeline()


def compute_loss_weighting(args, timesteps, noise_scheduler):
    """
    Compute loss weighting for improved training stability.
    Supports min-SNR weighting similar to Kohya's implementation.
    """
    if args.get("min_snr_gamma") is not None and args.get("min_snr_gamma") != "":
        snr = compute_snr(noise_scheduler, timesteps)
        min_snr_gamma = float(args.get("min_snr_gamma"))
        snr_weight = torch.stack([snr, min_snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        return snr_weight
    elif args.get("snr_gamma") is not None and args.get("snr_gamma") != "":
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, float(args["snr_gamma"]) * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        return mse_loss_weights
    return None


def compute_loss(model_pred, target, timesteps, noise_scheduler, args):
    """
    Compute loss with support for different loss types and weighting schemes.
    """
    loss_type = args.get("loss_type", "l2")
    
    if loss_type == "l2":
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(model_pred.float(), target.float(), reduction="none", beta=args.get("huber_c", 0.1))
    else:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    
    # Apply loss weighting if specified
    loss_weights = compute_loss_weighting(args, timesteps, noise_scheduler)
    
    if loss_weights is not None and not torch.all(loss_weights == 0):
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_weights
        return loss.mean()
    else:
        return loss.mean()


def compute_time_ids(original_size, crops_coords_top_left, target_size, dtype, device, weight_dtype=None):
    """
    Compute time IDs for SDXL conditioning.
    """
    if weight_dtype is None:
        weight_dtype = dtype
    
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype, device=device)
    return add_time_ids


def encode_prompt(
    pipe,
    text_encoders,
    tokenizers,
    prompt,
    device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    lora_scale=None,
    clip_skip=None,
):
    """
    Enhanced SDXL-compatible encode_prompt function that properly handles dual text encoders
    and pooled embeddings for SDXL models.
    """
    tokenizers = (
        tokenizers if tokenizers is not None
        else [pipe.tokenizer, pipe.tokenizer_2] if hasattr(pipe, "tokenizer_2") else [pipe.tokenizer]
    )
    text_encoders = (
        text_encoders if text_encoders is not None
        else [pipe.text_encoder, pipe.text_encoder_2] if hasattr(pipe, "text_encoder_2") else [pipe.text_encoder]
    )
    
    if prompt_embeds is None:
        prompt_2 = prompt if hasattr(pipe, "text_encoder_2") else None
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_sdxl(
            text_encoders,
            tokenizers,
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            clip_skip=clip_skip,
        )
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def encode_prompt_sdxl(
    text_encoders,
    tokenizers,
    prompt,
    prompt_2,
    device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt=None,
    negative_prompt_2=None,
    clip_skip=None,
):
    """
    Encodes the prompt into text encoder hidden states for SDXL.
    """
    prompt_embeds_list = []
    prompts = [prompt, prompt_2] if prompt_2 else [prompt]
    
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        if prompt is None:
            prompt = ""
        
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        
        max_length = tokenizer.model_max_length
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            print(f"The following part of your input was truncated: {removed_text}")
        
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        
        prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    
    # Get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None
    if do_classifier_free_guidance and negative_prompt_2 is None:
        negative_prompt_2 = negative_prompt
    
    if do_classifier_free_guidance and negative_prompt is None:
        negative_prompt = ""
        negative_prompt_2 = ""
    
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    
    if do_classifier_free_guidance:
        negative_prompt_embeds_list = []
        negative_prompts = [negative_prompt, negative_prompt_2] if negative_prompt_2 else [negative_prompt]
        
        for negative_prompt, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):
            if negative_prompt is None:
                negative_prompt = ""
            
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            
            if clip_skip is None:
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            else:
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-(clip_skip + 2)]
            
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        
        if zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(negative_prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoders[0].dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    else:
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
    
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)
    if negative_pooled_prompt_embeds is not None:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def generate_sample_images(pipeline, prompt, num_images, output_dir, prefix, args, device, weight_dtype):
    """
    Generate sample images and save them with the specified prefix.
    
    Args:
        pipeline: The diffusion pipeline to use for generation
        prompt: Text prompt for image generation
        num_images: Number of images to generate
        output_dir: Directory to save images
        prefix: Filename prefix (e.g., 'before_train' or 'after_train')
        args: Configuration arguments
        device: Device to run on
        weight_dtype: Weight data type
    """
    lab.log(f"Generating {num_images} {prefix} images...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        with torch.no_grad():
            image = pipeline(
                prompt,
                num_inference_steps=int(args.get("eval_num_inference_steps", 50)),
                guidance_scale=float(args.get("eval_guidance_scale", 7.5)),
                height=int(args.get("resolution", 512)),
                width=int(args.get("resolution", 512)),
            ).images[0]
        
        # Save image
        image_filename = f"{prefix}_{i+1:02d}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        
        # Save as artifact through lab
        lab.save_artifact(image_path, image_filename)
        lab.log(f"Saved {image_filename}")
    
    lab.log(f"✅ Generated {num_images} {prefix} images")


def train_diffusion_lora():
    """Main training function for Stable Diffusion LoRA"""
    
    # Training configuration with defaults
    training_config = {
        "experiment_name": "stable-diffusion-lora-training",
        "model_name": "CompVis/stable-diffusion-v1-4",
        "dataset": "nkasmanoff/nasa_earth_instagram",
        "output_dir": "./output",
        "log_to_wandb": False,
        "_config": {
            # Model settings
            "model_architecture": "StableDiffusionPipeline",
            "revision": None,
            "variant": None,
            
            # Dataset settings
            "caption_column": "text",
            "image_column": "image",
            "caption_dropout_rate": 0.0,
            "trigger_word": "",
            
            # Training settings - reduced for small dataset
            "num_train_epochs": 1,  # Few epochs for quick testing
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resolution": 256,  # Smaller resolution for faster training
            "dataloader_num_workers": 0,
            
            # Image augmentation
            "center_crop": False,
            "image_interpolation_mode": "lanczos",
            "random_flip": False,
            "color_jitter": False,
            "color_jitter_brightness": 0.1,
            "color_jitter_contrast": 0.1,
            "color_jitter_saturation": 0.1,
            "color_jitter_hue": 0.05,
            "random_rotation": False,
            "rotation_degrees": 5,
            "rotation_prob": 0.3,
            
            # LoRA settings - smaller for faster training
            "lora_r": 4,  # Smaller LoRA rank
            "lora_alpha": 8,
            
            # Optimizer settings
            "learning_rate": 0.0001,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,  # No warmup for small dataset
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            
            # Advanced settings
            "mixed_precision": "fp16",
            "gradient_checkpointing": False,
            "enable_xformers_memory_efficient_attention": False,
            "use_ema": False,
            "ema_decay": 0.9999,
            "noise_offset": 0,
            "prediction_type": None,
            "loss_type": "l2",
            "huber_c": 0.1,
            "min_snr_gamma": None,
            "snr_gamma": None,
            
            # Evaluation settings
            "eval_prompt": "a cute pokemon",
            "eval_steps": 1,
            "eval_num_inference_steps": 25,  # Faster inference
            "eval_guidance_scale": 7.5,
            
            # Output settings
            "adaptor_name": "adaptor",
            "adaptor_output_dir": "./output/lora",
        }
    }
    
    try:
        # Initialize lab
        lab.init()
        lab.set_config(training_config)
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"🚀 Training started at {start_time}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")
        
        args = training_config["_config"]
        
        # Setup output directories
        output_dir = training_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        eval_images_dir = os.path.join(output_dir, "eval_images")
        os.makedirs(eval_images_dir, exist_ok=True)
        
        lab.log("***** Running training *****")
        
        # Load dataset
        lab.log("Loading dataset...")
        from datasets import load_dataset
        
        dataset_name = training_config["dataset"]
        try:
            # Load the full dataset first, then slice it to avoid cache issues
            datasets_dict = load_dataset(dataset_name, split="train")
            # Take only the first 20 samples for testing
            dataset = datasets_dict.select(range(min(20, len(datasets_dict))))
            lab.log(f"✅ Loaded dataset: {dataset_name}")
        except Exception as e:
            lab.log(f"❌ Failed to load dataset {dataset_name}: {e}")
            raise RuntimeError(f"Could not load dataset {dataset_name}. Please check the dataset name and ensure it's available.")
        
        lab.log(f"Loaded dataset with {len(dataset)} examples (limited to 20 for testing)")
        lab.update_progress(10)
        
        # Model and tokenizer loading
        pretrained_model_name_or_path = training_config["model_name"]
        revision = args.get("revision", None)
        variant = args.get("variant", None)
        model_architecture = args.get("model_architecture", "StableDiffusionPipeline")
        
        # Load pipeline
        lab.log(f"Loading pipeline: {pretrained_model_name_or_path}")
        pipeline_kwargs = {
            "torch_dtype": torch.float16,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        
        temp_pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, **pipeline_kwargs)
        
        # Extract components
        noise_scheduler = temp_pipeline.scheduler
        tokenizer = temp_pipeline.tokenizer
        text_encoder = temp_pipeline.text_encoder
        vae = temp_pipeline.vae
        
        # Handle different architectures
        if hasattr(temp_pipeline, "transformer"):
            unet = temp_pipeline.transformer
            model_component_name = "transformer"
        else:
            unet = temp_pipeline.unet
            model_component_name = "unet"
        
        # Handle SDXL dual text encoders
        text_encoder_2 = getattr(temp_pipeline, "text_encoder_2", None)
        tokenizer_2 = getattr(temp_pipeline, "tokenizer_2", None)
        
        # Clean up temporary pipeline
        del temp_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        lab.log(f"✅ Model components loaded: {type(unet).__name__}")
        if text_encoder_2 is not None:
            lab.log("Dual text encoder setup detected (SDXL)")
        
        lab.update_progress(20)
        
        # Freeze parameters
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        if text_encoder_2 is not None:
            text_encoder_2.requires_grad_(False)
        
        # Enable optimizations
        if args.get("enable_xformers_memory_efficient_attention", False) and xformers_available:
            try:
                unet.enable_xformers_memory_efficient_attention()
                if hasattr(vae, "enable_xformers_memory_efficient_attention"):
                    vae.enable_xformers_memory_efficient_attention()
                lab.log("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                lab.log(f"⚠️  Failed to enable xFormers: {e}")
        
        if args.get("gradient_checkpointing", False):
            unet.enable_gradient_checkpointing()
            if hasattr(text_encoder, "gradient_checkpointing_enable"):
                text_encoder.gradient_checkpointing_enable()
            if text_encoder_2 is not None and hasattr(text_encoder_2, "gradient_checkpointing_enable"):
                text_encoder_2.gradient_checkpointing_enable()
            lab.log("✅ Gradient checkpointing enabled")
        
        # Mixed precision
        weight_dtype = torch.float32
        mixed_precision = args.get("mixed_precision", None)
        if mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        # Detect architecture
        is_sdxl = "StableDiffusionXLPipeline" in model_architecture
        is_sd3 = "StableDiffusion3Pipeline" in model_architecture
        is_flux = "FluxPipeline" in model_architecture
        
        lab.log(f"Architecture: SDXL={is_sdxl}, SD3={is_sd3}, Flux={is_flux}")
        
        # Define LoRA target modules
        if is_sdxl:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
            architecture_name = "SDXL"
        elif is_sd3:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            architecture_name = "SD3"
        elif is_flux:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            architecture_name = "Flux"
        else:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
            architecture_name = "SD 1.x"
        
        lab.log(f"Using LoRA target modules for {architecture_name}: {target_modules}")
        
        unet_lora_config = LoraConfig(
            r=int(args.get("lora_r", 8)),
            lora_alpha=int(args.get("lora_alpha", 16)),
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        if text_encoder_2 is not None:
            text_encoder_2.to(device, dtype=weight_dtype)
        
        unet.add_adapter(unet_lora_config)
        if mixed_precision == "fp16":
            cast_training_params(unet, dtype=torch.float32)
        
        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
        
        lab.log("✅ LoRA adapters added to model")
        lab.update_progress(30)
        
        # EMA (optional)
        ema_unet = None
        if args.get("use_ema", False):
            try:
                from diffusers.training_utils import EMAModel
                lora_parameters = [p for p in unet.parameters() if p.requires_grad]
                if lora_parameters:
                    ema_unet = EMAModel(lora_parameters, decay=args.get("ema_decay", 0.9999))
                    lab.log("✅ EMA enabled for LoRA parameters")
            except Exception as e:
                lab.log(f"⚠️  EMA initialization failed: {e}")
                ema_unet = None
        
        # Generate before_train images
        eval_prompt = args.get("eval_prompt", "").strip()
        if eval_prompt:
            # Create evaluation pipeline
            eval_pipeline = AutoPipelineForText2Image.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                variant=variant,
                torch_dtype=weight_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if model_component_name == "transformer":
                eval_pipeline.transformer = unet
            else:
                eval_pipeline.unet = unet
            eval_pipeline = eval_pipeline.to(device)
            
            # Generate 5 before_train images
            generate_sample_images(
                eval_pipeline, 
                eval_prompt, 
                5, 
                eval_images_dir, 
                "before_train",
                args,
                device,
                weight_dtype
            )
            
            # Clean up evaluation pipeline
            del eval_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        lab.update_progress(35)
        
        # Data transforms with augmentation
        interpolation = getattr(transforms.InterpolationMode, args.get("image_interpolation_mode", "lanczos").upper(), None)
        resolution = int(args.get("resolution", 512))
        
        transform_list = [
            transforms.Resize(resolution, interpolation=interpolation),
        ]
        
        if args.get("center_crop", False):
            transform_list.append(transforms.CenterCrop(resolution))
        else:
            transform_list.append(transforms.RandomCrop(resolution))
        
        if args.get("random_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if args.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=args.get("color_jitter_brightness", 0.1),
                    contrast=args.get("color_jitter_contrast", 0.1),
                    saturation=args.get("color_jitter_saturation", 0.1),
                    hue=args.get("color_jitter_hue", 0.05),
                )
            )
        
        if args.get("random_rotation", False):
            transform_list.append(
                transforms.RandomApply(
                    [transforms.RandomRotation(args.get("rotation_degrees", 5))],
                    p=args.get("rotation_prob", 0.3)
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        train_transforms = transforms.Compose(transform_list)
        
        def tokenize_captions(examples, is_train=True):
            captions = []
            caption_column = args.get("caption_column", "text")
            trigger_word = args.get("trigger_word", "").strip()
            caption_dropout_rate = float(args.get("caption_dropout_rate", 0.0))
            
            if caption_column not in examples:
                lab.log(f"⚠️  Caption column '{caption_column}' not found, using empty captions")
                num_examples = len(next(iter(examples.values())))
                captions = [""] * num_examples
                caption_dropout_rate = 1.0
            else:
                for caption in examples[caption_column]:
                    if isinstance(caption, str):
                        processed_caption = caption
                    elif isinstance(caption, (list, np.ndarray)):
                        processed_caption = random.choice(caption) if is_train else caption[0]
                    else:
                        raise ValueError(f"Caption column should contain strings or lists of strings")
                    
                    if is_train and caption_dropout_rate > 0 and random.random() < caption_dropout_rate:
                        processed_caption = ""
                    else:
                        if trigger_word:
                            processed_caption = f"{trigger_word}, {processed_caption}"
                    
                    captions.append(processed_caption)
            
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            result = {"input_ids": inputs.input_ids}
            
            if tokenizer_2 is not None:
                inputs_2 = tokenizer_2(
                    captions,
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                result["input_ids_2"] = inputs_2.input_ids
            
            return result
        
        image_column = args.get("image_column", "image")
        
        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            
            processed_images = []
            original_sizes = []
            crop_coords_top_left = []
            target_sizes = []
            
            for image in images:
                original_size = image.size
                original_sizes.append(original_size)
                
                transformed_image = train_transforms(image)
                processed_images.append(transformed_image)
                
                if args.get("center_crop", False):
                    crop_size = resolution
                    left = (original_size[0] - crop_size) // 2
                    top = (original_size[1] - crop_size) // 2
                    crop_coords_top_left.append((left, top))
                else:
                    crop_coords_top_left.append((0, 0))
                
                target_size = (resolution, resolution)
                target_sizes.append(target_size)
            
            examples["pixel_values"] = processed_images
            examples["original_sizes"] = original_sizes
            examples["crop_coords_top_left"] = crop_coords_top_left
            examples["target_sizes"] = target_sizes
            
            tokenization_results = tokenize_captions(examples)
            examples["input_ids"] = tokenization_results["input_ids"]
            
            if "input_ids_2" in tokenization_results:
                examples["input_ids_2"] = tokenization_results["input_ids_2"]
            
            return examples
        
        train_dataset = dataset.with_transform(preprocess_train)
        
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            
            batch = {"pixel_values": pixel_values, "input_ids": input_ids}
            
            if "input_ids_2" in examples[0]:
                input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
                batch["input_ids_2"] = input_ids_2
            
            if "original_sizes" in examples[0]:
                batch["original_sizes"] = [example["original_sizes"] for example in examples]
                batch["crop_coords_top_left"] = [example["crop_coords_top_left"] for example in examples]
                batch["target_sizes"] = [example["target_sizes"] for example in examples]
            
            return batch
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=int(args.get("train_batch_size", 1)),
            num_workers=int(args.get("dataloader_num_workers", 0)),
        )
        
        lab.log(f"✅ Dataset prepared with {len(train_dataset)} examples")
        lab.update_progress(40)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=float(args.get("learning_rate", 1e-4)),
            betas=(float(args.get("adam_beta1", 0.9)), float(args.get("adam_beta2", 0.999))),
            weight_decay=float(args.get("adam_weight_decay", 1e-2)),
            eps=float(args.get("adam_epsilon", 1e-8)),
        )
        
        # Scheduler
        num_train_epochs = int(args.get("num_train_epochs", 100))
        gradient_accumulation_steps = int(args.get("gradient_accumulation_steps", 1))
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        
        lr_scheduler = get_scheduler(
            args.get("lr_scheduler", "constant"),
            optimizer=optimizer,
            num_warmup_steps=int(args.get("lr_warmup_steps", 50)),
            num_training_steps=max_train_steps,
        )
        
        # Training loop
        lab.log("***** Starting Training *****")
        lab.log(f"  Num examples = {len(train_dataset)}")
        lab.log(f"  Num Epochs = {num_train_epochs}")
        lab.log(f"  Batch size = {args.get('train_batch_size', 1)}")
        lab.log(f"  Total optimization steps = {max_train_steps}")
        
        global_step = 0
        for epoch in range(num_train_epochs):
            unet.train()
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                if args.get("noise_offset", 0):
                    noise += args["noise_offset"] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text encoding
                if is_sdxl:
                    prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    text_encoders = [text_encoder, text_encoder_2] if text_encoder_2 is not None else [text_encoder]
                    tokenizers_list = [tokenizer, tokenizer_2] if tokenizer_2 is not None else [tokenizer]
                    
                    class TempPipeline:
                        def __init__(self, text_encoder, text_encoder_2, tokenizer, tokenizer_2):
                            self.text_encoder = text_encoder
                            self.text_encoder_2 = text_encoder_2
                            self.tokenizer = tokenizer
                            self.tokenizer_2 = tokenizer_2
                    
                    temp_pipe = TempPipeline(text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                    encoder_hidden_states, _, pooled_prompt_embeds, _ = encode_prompt(
                        temp_pipe,
                        text_encoders,
                        tokenizers_list,
                        prompts,
                        device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]
                    pooled_prompt_embeds = None
                
                # Loss target
                prediction_type = args.get("prediction_type", None)
                if prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=prediction_type)
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                unet_kwargs = {"timestep": timesteps, "encoder_hidden_states": encoder_hidden_states, "return_dict": False}
                
                # SDXL requires additional conditioning
                if is_sdxl:
                    batch_size = noisy_latents.shape[0]
                    
                    if pooled_prompt_embeds is not None:
                        text_embeds = (
                            pooled_prompt_embeds.repeat(batch_size, 1)
                            if pooled_prompt_embeds.shape[0] == 1
                            else pooled_prompt_embeds
                        )
                    else:
                        text_embeds = torch.zeros(batch_size, 1280, device=device, dtype=weight_dtype)
                    
                    if "original_sizes" in batch:
                        time_ids_list = []
                        for i in range(batch_size):
                            time_ids = compute_time_ids(
                                batch["original_sizes"][i],
                                batch["crop_coords_top_left"][i],
                                batch["target_sizes"][i],
                                dtype=weight_dtype,
                                device=device,
                            )
                            time_ids_list.append(time_ids)
                        time_ids = torch.cat(time_ids_list, dim=0)
                    else:
                        time_ids = torch.tensor(
                            [[resolution, resolution, 0, 0, resolution, resolution]] * batch_size,
                            device=device,
                            dtype=weight_dtype,
                        )
                    
                    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
                    unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
                
                model_pred = unet(noisy_latents, **unet_kwargs)[0]
                loss = compute_loss(model_pred, target, timesteps, noise_scheduler, args)
                
                loss.backward()
                epoch_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(list(lora_layers), float(args.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update EMA
                    if ema_unet is not None:
                        try:
                            lora_parameters_for_ema = [p for p in unet.parameters() if p.requires_grad]
                            if lora_parameters_for_ema:
                                ema_unet.step(lora_parameters_for_ema)
                        except Exception as e:
                            lab.log(f"⚠️  EMA step failed: {e}")
                            ema_unet = None
                    
                    # Memory cleanup
                    if torch.cuda.is_available() and global_step % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    global_step += 1
                    
                    # Progress reporting
                    percent_complete = 40 + (50.0 * global_step / max_train_steps)
                    lab.update_progress(int(percent_complete))
                    
                    if global_step % 10 == 0:
                        lab.log(f"Epoch {epoch+1}/{num_train_epochs}, Step {global_step}/{max_train_steps}, Loss: {loss.item():.4f}")
                    
                    if global_step >= max_train_steps:
                        break
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            lab.log(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
            
            if global_step >= max_train_steps:
                break
        
        lab.log("✅ Training completed")
        lab.update_progress(90)
        
        # Generate after_train images
        if eval_prompt:
            # Create evaluation pipeline
            eval_pipeline = AutoPipelineForText2Image.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                variant=variant,
                torch_dtype=weight_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            if model_component_name == "transformer":
                eval_pipeline.transformer = unet
            else:
                eval_pipeline.unet = unet
            eval_pipeline = eval_pipeline.to(device)
            
            # Generate 5 after_train images
            generate_sample_images(
                eval_pipeline,
                eval_prompt,
                5,
                eval_images_dir,
                "after_train",
                args,
                device,
                weight_dtype
            )
            
            # Clean up evaluation pipeline
            del eval_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save LoRA weights
        unet = unet.to(torch.float32)
        model_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        
        save_directory = args.get("adaptor_output_dir", os.path.join(output_dir, "lora"))
        os.makedirs(save_directory, exist_ok=True)
        
        lab.log(f"Saving LoRA weights to {save_directory}")
        
        # Save configuration info
        save_info = {
            "model_architecture": model_architecture,
            "lora_config": {
                "r": str(unet_lora_config.r),
                "lora_alpha": str(unet_lora_config.lora_alpha),
                "target_modules": str(unet_lora_config.target_modules),
            },
            "training_completed": True,
        }
        
        config_path = os.path.join(save_directory, "lora_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(save_info, f, indent=4)
        
        # Save LoRA weights based on architecture
        saved_successfully = False
        
        if not is_sdxl and not is_sd3 and not is_flux:
            try:
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_directory,
                    unet_lora_layers=model_lora_state_dict,
                    safe_serialization=True,
                )
                lab.log(f"✅ LoRA weights saved (SD 1.x)")
                saved_successfully = True
            except Exception as e:
                lab.log(f"⚠️  Error with SD 1.x save: {e}")
        
        if not saved_successfully and is_sdxl:
            try:
                from diffusers import StableDiffusionXLPipeline
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=save_directory,
                    unet_lora_layers=model_lora_state_dict,
                    text_encoder_lora_layers=None,
                    text_encoder_2_lora_layers=None,
                    safe_serialization=True,
                )
                lab.log(f"✅ LoRA weights saved (SDXL)")
                saved_successfully = True
            except Exception as e:
                lab.log(f"⚠️  Error with SDXL save: {e}")
        
        if not saved_successfully and is_sd3:
            try:
                from diffusers import StableDiffusion3Pipeline
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=save_directory,
                    unet_lora_layers=model_lora_state_dict,
                    safe_serialization=True,
                )
                lab.log(f"✅ LoRA weights saved (SD3)")
                saved_successfully = True
            except Exception as e:
                lab.log(f"⚠️  Error with SD3 save: {e}")
        
        if not saved_successfully and is_flux:
            try:
                from diffusers import FluxPipeline
                try:
                    FluxPipeline.save_lora_weights(
                        save_directory=save_directory,
                        transformer_lora_layers=model_lora_state_dict,
                        safe_serialization=True,
                    )
                except TypeError:
                    FluxPipeline.save_lora_weights(
                        save_directory=save_directory,
                        unet_lora_layers=model_lora_state_dict,
                        safe_serialization=True,
                    )
                lab.log(f"✅ LoRA weights saved (Flux)")
                saved_successfully = True
            except Exception as e:
                lab.log(f"⚠️  Error with Flux save: {e}")
        
        if not saved_successfully:
            try:
                from safetensors.torch import save_file
                save_file(model_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.safetensors"))
                lab.log(f"✅ LoRA weights saved (safetensors fallback)")
                saved_successfully = True
            except Exception as e:
                lab.log(f"⚠️  Error with safetensors: {e}")
                torch.save(model_lora_state_dict, os.path.join(save_directory, "pytorch_lora_weights.bin"))
                lab.log(f"✅ LoRA weights saved (PyTorch fallback)")
                saved_successfully = True
        
        # Save the model using lab
        saved_model_path = lab.save_model(save_directory, name="diffusion_lora_model")
        lab.log(f"✅ Model saved to: {saved_model_path}")
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")
        
        # Create training summary
        summary = {
            "training_type": "Stable Diffusion LoRA",
            "model_name": pretrained_model_name_or_path,
            "architecture": architecture_name,
            "dataset": dataset_name,
            "lora_r": args.get("lora_r"),
            "lora_alpha": args.get("lora_alpha"),
            "num_epochs": num_train_epochs,
            "total_steps": global_step,
            "learning_rate": args.get("learning_rate"),
            "batch_size": args.get("train_batch_size"),
            "resolution": args.get("resolution"),
            "training_duration": str(training_duration),
            "completed_at": end_time.isoformat(),
        }
        
        summary_file = os.path.join(output_dir, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        lab.save_artifact(summary_file, "training_summary.json")
        lab.log("✅ Training summary saved")
        
        lab.update_progress(100)
        lab.finish("Training completed successfully")
        
        return {
            "status": "success",
            "duration": str(training_duration),
            "output_dir": output_dir,
            "saved_model_path": saved_model_path,
            "architecture": architecture_name,
        }
    
    except KeyboardInterrupt:
        lab.error("Training stopped by user")
        return {"status": "stopped"}
    
    except Exception as e:
        error_msg = str(e)
        lab.log(f"❌ Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting Stable Diffusion LoRA training...")
    result = train_diffusion_lora()
    print("Training result:", result)
