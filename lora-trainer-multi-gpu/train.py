#!/usr/bin/env python3
"""
Multi-GPU LoRA Training script for LLMs using HuggingFace SFTTrainer with TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Multi-GPU training with accelerate
- Using lab.get_hf_callback() for automatic progress tracking and checkpoint saving
- LoRA fine-tuning with 4-bit quantization
- Automatic wandb integration when configured
"""

import os
import json
import subprocess
import re
from datetime import datetime

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def find_lora_target_modules(model, keyword="proj"):
    """
    Returns all submodule names (e.g., 'q_proj') suitable for LoRA injection.
    These can be passed directly to LoraConfig as `target_modules`.
    """
    import torch.nn as nn
    
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and keyword in name:
            # Keep full relative module name, excluding the root prefix (e.g., "model.")
            cleaned_name = ".".join(name.split(".")[1:]) if name.startswith("model.") else name
            module_names.add(cleaned_name.split(".")[-1])  # Use just the relative layer name
    return sorted(module_names)


def train_with_lora():
    """Training function using HuggingFace SFTTrainer with LoRA and multi-GPU support"""

    # Configure GPU usage - use all available GPUs
    # GPU selection is handled by accelerate launch
    gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()

        # Get parameters from task configuration (set via UI)
        # These parameters are accessible via lab.get_config() after lab.init()
        config = lab.get_config()

        # Extract parameters with defaults
        # All these can be set in the UI when creating/launching the task
        model_name = config.get("model_name", "meta-llama/Llama-2-7b-hf")
        dataset_name = config.get("dataset", "Trelis/touch-rugby-rules")
        output_dir = config.get("output_dir", "./output")
        log_to_wandb = config.get("log_to_wandb", True)
        fuse_model = config.get("fuse_model", False)

        # LoRA configuration
        lora_r = config.get("lora_r", 16)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)

        # Training hyperparameters
        # Convert string values to appropriate types (parameters from sweeps may come as strings)
        learning_rate_raw = config.get("learning_rate", 5e-5)
        learning_rate = (
            float(learning_rate_raw) if isinstance(learning_rate_raw, (str, int, float)) else learning_rate_raw
        )

        batch_size_raw = config.get("batch_size", 4)
        batch_size = int(batch_size_raw) if isinstance(batch_size_raw, (str, int, float)) else batch_size_raw

        num_train_epochs = config.get("num_train_epochs", 1)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        warmup_ratio = config.get("warmup_ratio", 0.03)
        weight_decay = config.get("weight_decay", 0.01)
        learning_rate_schedule = config.get("learning_rate_schedule", "constant")
        max_steps = config.get("max_steps", -1)

        # Device configuration
        train_device = config.get("train_device", "cuda")
        gpu_ids_config = config.get("gpu_ids", "auto")

        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")

        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"Learning rate: {learning_rate}")
        lab.log(f"Batch size: {batch_size}")
        lab.log(f"Number of epochs: {num_train_epochs}")
        lab.log(f"LoRA R: {lora_r}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
        lab.log(f"Training device: {train_device}")
        if train_device == "cuda" and gpu_ids_config != "auto":
            lab.log(f"GPU IDs: {gpu_ids_config}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_name)
            lab.log(f"Loaded dataset with {len(dataset['train'])} examples")

        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            # Create a small fake dataset for testing
            from datasets import Dataset

            dataset = {
                "train": Dataset.from_list(
                    [
                        {"text": "What are the rules of touch rugby?"},
                        {"text": "How many players are on a touch rugby team?"},
                        {"text": "What is the objective of touch rugby?"},
                    ]
                )
            }
            lab.log("Using fake dataset for testing")

        lab.update_progress(20)

        # Load model and tokenizer
        lab.log("Loading model and tokenizer...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            # 4-bit quantization configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # Determine device_map based on training device
            device_map = None if train_device == "cuda" else "auto"

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    use_cache=False,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except TypeError:
                # Some models don't support use_cache parameter
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True,
                )

            model.config.pretraining_tp = 1

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            lab.log(f"✅ Loaded model: {model_name}")

        except ImportError:
            lab.log("⚠️  Transformers not available, skipping training")
            lab.error("Training skipped - transformers not available")
            return {"status": "skipped", "reason": "transformers not available"}
        except Exception as e:
            lab.log(f"Error loading model: {e}")
            lab.error("Training failed - model loading error")
            raise e

        lab.update_progress(40)

        # Setup LoRA configuration
        lab.log("Setting up LoRA configuration...")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Find target modules automatically
            lora_target_modules = find_lora_target_modules(model)
            if not lora_target_modules:
                lab.log("⚠️  No target modules found automatically, using default modules")
                lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

            lab.log(f"LoRA target modules: {lora_target_modules}")

            # LoRA configuration
            peft_config = LoraConfig(
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                r=int(lora_r),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
            )

            # Prepare model for LoRA
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            lab.log(f"✅ LoRA configuration applied successfully")
            lab.log(f"Trainable parameters: {model.print_trainable_parameters()}")

        except Exception as e:
            lab.log(f"Error setting up LoRA: {e}")
            lab.error("Training failed - LoRA setup error")
            raise e

        lab.update_progress(55)

        # Set up SFTTrainer with multi-GPU support
        lab.log("Setting up SFTTrainer...")
        try:
            from trl import SFTTrainer, SFTConfig

            # Build training config
            training_args_dict = {
                "output_dir": output_dir,
                "num_train_epochs": int(num_train_epochs),
                "per_device_train_batch_size": int(batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "learning_rate": float(learning_rate),
                "warmup_ratio": float(warmup_ratio),
                "weight_decay": float(weight_decay),
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "report_to": ["wandb"] if log_to_wandb else [],
                "run_name": f"lora-train-{lab.job.id}",
                "logging_dir": f"{output_dir}/logs",
                "remove_unused_columns": False,
                "push_to_hub": False,
                "dataset_text_field": "text",
                "bf16": True,
                # Multi-GPU optimization
                "save_total_limit": 3,
                "save_strategy": "steps",
                "load_best_model_at_end": False,
                "dataloader_num_workers": 0,
                "lr_scheduler_type": learning_rate_schedule,
                "max_grad_norm": 0.3,
                "ddp_find_unused_parameters": False,
            }

            # Only add max_steps if it's a positive integer
            if max_steps is not None and max_steps > 0:
                training_args_dict["max_steps"] = int(max_steps)

            # Only add resume_from_checkpoint if it's provided
            if checkpoint:
                training_args_dict["resume_from_checkpoint"] = checkpoint

            training_args = SFTConfig(**training_args_dict)

            # Get TransformerLab callback for automatic progress tracking
            transformerlab_callback = lab.get_hf_callback()

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                processing_class=tokenizer,
                peft_config=peft_config,
                callbacks=[transformerlab_callback],
            )

            lab.log("✅ Trainer created")

        except Exception as e:
            lab.log(f"Error setting up trainer: {e}")
            lab.error("Training failed - trainer setup error")
            raise e

        lab.update_progress(65)

        # Start training
        lab.log("Starting training...")

        try:
            if "trainer" in locals():
                # Real training with SFTTrainer
                trainer.train(resume_from_checkpoint=checkpoint)
                lab.log("✅ Training completed with SFTTrainer")

                # Create training progress summary artifact
                progress_file = os.path.join(output_dir, "training_progress_summary.json")
                with open(progress_file, "w") as f:
                    json.dump(
                        {
                            "training_type": "SFTTrainer with LoRA",
                            "total_epochs": num_train_epochs,
                            "model_name": model_name,
                            "dataset": dataset_name,
                            "lora_r": lora_r,
                            "lora_alpha": lora_alpha,
                            "lora_dropout": lora_dropout,
                            "completed_at": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                    )

                progress_artifact_path = lab.save_artifact(progress_file, "training_progress_summary.json")
                lab.log(f"Saved training progress: {progress_artifact_path}")

                # Save the trained model (adapter)
                lab.log("Saving trained LoRA adapter...")
                trainer.save_model(output_dir)
                lab.log("✅ LoRA adapter saved")

                # Fuse model if requested
                if fuse_model:
                    lab.log("Fusing model with LoRA adapter...")
                    try:
                        from peft import PeftModel
                        from transformers import AutoConfig

                        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                        model_architecture = model_config.architectures[0]

                        # Load base model in full precision for merging
                        lab.log("Loading base model for fusion...")
                        device = "cuda:0" if torch.cuda.is_available() else "cpu"

                        base_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map=device,
                            trust_remote_code=True,
                        )

                        # Load and merge adapter
                        peft_model = PeftModel.from_pretrained(base_model, output_dir)
                        merged_model = peft_model.merge_and_unload()

                        # Save fused model
                        fused_model_dir = os.path.join(output_dir, "fused_model")
                        os.makedirs(fused_model_dir, exist_ok=True)
                        merged_model.save_pretrained(fused_model_dir)
                        tokenizer.save_pretrained(fused_model_dir)

                        lab.log(f"✅ Fused model saved to {fused_model_dir}")

                        # Save fused model using TransformerLab
                        saved_fused_path = lab.save_model(fused_model_dir, name="fused_lora_model")
                        lab.log(f"✅ Fused model saved to job models directory: {saved_fused_path}")

                    except Exception as e:
                        lab.log(f"⚠️  Error fusing model: {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            lab.log(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise

        lab.update_progress(85)

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

        # Save final artifacts
        final_model_file = os.path.join(output_dir, "final_model_summary.txt")
        with open(final_model_file, "w") as f:
            f.write("Final Model Summary\n")
            f.write("==================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"LoRA R: {lora_r}\n")
            f.write(f"LoRA Alpha: {lora_alpha}\n")
            f.write(f"LoRA Dropout: {lora_dropout}\n")
            f.write(f"Completed at: {end_time}\n")

        # Save final model summary as artifact
        final_model_path = lab.save_artifact(final_model_file, "final_model_summary.txt")
        lab.log(f"Saved final model summary: {final_model_path}")

        # Save training configuration as artifact
        config_file = os.path.join(output_dir, "training_config.json")
        training_config_dict = {
            "model_name": model_name,
            "dataset": dataset_name,
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_train_epochs": num_train_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "learning_rate_schedule": learning_rate_schedule,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "train_device": train_device,
            "gpu_ids": gpu_ids_config,
            "log_to_wandb": log_to_wandb,
            "fuse_model": fuse_model,
        }

        with open(config_file, "w") as f:
            json.dump(training_config_dict, f, indent=2)

        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")

        # Save the adapter as model
        saved_model_path = output_dir
        saved_path = lab.save_model(saved_model_path, name="lora_trained_adapter")
        lab.log(f"✅ LoRA adapter saved to job models directory: {saved_path}")

        # Finish wandb run if it was initialized
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
                lab.log("✅ Wandb run finished")
        except Exception:
            pass

        print("Complete")

        # Complete the job in TransformerLab via facade
        lab.finish("Training completed successfully!")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": output_dir,
            "saved_model_path": saved_path,
            "trainer_type": "SFTTrainer with LoRA",
            "gpu_used": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        }

    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")

        import traceback

        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    result = train_with_lora()
    print("Training result:", result)
