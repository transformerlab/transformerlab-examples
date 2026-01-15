#!/usr/bin/env python3
"""
Training script for GPT-OSS model using HuggingFace SFTTrainer with TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Using lab.get_hf_callback() for automatic progress tracking and checkpoint saving
- LoRA support for efficient fine-tuning of large models
"""

import os
import json
import subprocess
import re
from datetime import datetime

from lab import lab

# Login to huggingface
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))


def train_gpt_oss():
    """Training function using HuggingFace SFTTrainer with GPT-OSS model"""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()

        # Get parameters from task configuration (set via UI)
        # These parameters are accessible via lab.get_config() after lab.init()
        config = lab.get_config()

        # Extract parameters with defaults
        # All these can be set in the UI when creating/launching the task
        model_name = config.get("model_name", "openai/gpt-oss-20b")
        dataset_name = config.get("dataset", "HuggingFaceH4/Multilingual-Thinking")
        output_dir = config.get("output_dir", "./output")
        enable_lora = config.get("enable_lora", False)
        enable_profiling = config.get("enable_profiling", False)

        # Convert string values to appropriate types (parameters from sweeps may come as strings)
        learning_rate_raw = config.get("learning_rate", 2e-4)
        learning_rate = (
            float(learning_rate_raw) if isinstance(learning_rate_raw, (str, int, float)) else learning_rate_raw
        )

        batch_size_raw = config.get("per_device_train_batch_size", 1)
        batch_size = int(batch_size_raw) if isinstance(batch_size_raw, (str, int, float)) else batch_size_raw

        num_train_epochs = config.get("num_train_epochs", 1)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        warmup_ratio = config.get("warmup_ratio", 0.03)
        save_steps = config.get("save_steps", 500)
        max_steps = config.get("max_steps", -1)

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
        lab.log(f"LoRA enabled: {enable_lora}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)




        lab.update_progress(20)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            num_proc = int(os.cpu_count() / 2)
            dataset = load_dataset(dataset_name, split="train", num_proc=num_proc)
            lab.log(f"Loaded dataset with {len(dataset)} examples")

        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            lab.error("Dataset loading failed")
            raise e

        lab.update_progress(35)

        # Load model and tokenizer
        lab.log("Loading model and tokenizer...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import Mxfp4Config

            # Setup quantization for efficient memory usage
            quantization_config = Mxfp4Config(dequantize=True)

            device_map_args = {}
            if enable_lora:
                device_map_args = {'device_map': 'auto'}

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                torch_dtype="auto",
                use_cache=False,
                quantization_config=quantization_config,
                **device_map_args,
            )

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            lab.log(f"Loaded model: {model_name}")

        except ImportError:
            lab.log("⚠️  Transformers not available, skipping training")
            lab.error("Training skipped - transformers not available")
            return {"status": "skipped", "reason": "transformers not available"}
        except Exception as e:
            lab.log(f"Error loading model: {e}")
            lab.error("Training failed - model loading error")
            raise e

        lab.update_progress(50)

        # Setup LoRA if enabled
        if enable_lora:
            lab.log("Setting up LoRA...")
            try:
                from peft import get_peft_model, LoraConfig

                # Determine target parameters based on model
                num_layers = 0
                target_parameters = []
                if model_name == 'openai/gpt-oss-120b':
                    num_layers = 36
                elif model_name == 'openai/gpt-oss-20b':
                    num_layers = 24

                for i in range(num_layers):
                    target_parameters.append(f'{i}.mlp.experts.gate_up_proj')
                    target_parameters.append(f'{i}.mlp.experts.down_proj')

                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules="all-linear",
                    target_parameters=target_parameters,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                lab.log("✅ LoRA configured")

            except Exception as e:
                lab.log(f"Error setting up LoRA: {e}")
                lab.error("LoRA setup failed")
                raise e

        lab.update_progress(60)

        # Set up SFTTrainer with automatic TransformerLab integration
        lab.log("Setting up trainer...")
        try:
            from trl import SFTTrainer, SFTConfig

            # SFTConfig with automatic checkpoint saving
            training_args_dict = {
                "output_dir": output_dir,
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "logging_steps": 1,
                "save_steps": save_steps,
                "save_total_limit": 3,
                "save_strategy": "steps",
                "dataset_text_field": "text",
                "bf16": False,
                "remove_unused_columns": False,
                "push_to_hub": False,
                "logging_dir": f"{output_dir}/logs",
                "dataloader_num_workers": 0,
                "load_best_model_at_end": False,
                "lr_scheduler_type": "cosine_with_min_lr",
                "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
            }

            # Only add max_steps if it's a positive integer
            if max_steps is not None and max_steps > 0:
                training_args_dict["max_steps"] = int(max_steps)

            # Only add resume_from_checkpoint if it's provided
            if checkpoint:
                training_args_dict["resume_from_checkpoint"] = checkpoint

            training_args = SFTConfig(**training_args_dict)

            # Get TransformerLab callback for automatic progress tracking and checkpoint saving
            transformerlab_callback = lab.get_hf_callback()

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[transformerlab_callback],
            )

            lab.log("✅ Trainer created")

        except Exception as e:
            lab.log(f"Error setting up trainer: {e}")
            lab.error("Training failed - trainer setup error")
            raise e

        lab.update_progress(70)

        # Start training
        lab.log("Starting training...")
        try:
            trainer.train(resume_from_checkpoint=checkpoint)
            lab.log("✅ Training completed")

            # Save trained model
            lab.log("Saving trained model...")
            trainer.save_model()
            lab.log("✅ Model saved")

        except Exception as e:
            lab.log(f"Error during training: {e}")
            lab.error(f"Training failed: {str(e)}")
            raise

        lab.update_progress(85)

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

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
            "enable_lora": enable_lora,
        }

        with open(config_file, "w") as f:
            json.dump(training_config_dict, f, indent=2)

        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")

        # Save the model using TransformerLab's model directory
        saved_model_path = output_dir
        saved_path = lab.save_model(saved_model_path, name="trained_model")
        lab.log(f"✅ Model saved to job models directory: {saved_path}")

        lab.update_progress(95)

        # Complete the job in TransformerLab
        lab.finish("Training completed successfully!")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": output_dir,
            "saved_model_path": saved_path,
            "trainer_type": "SFTTrainer",
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
    result = train_gpt_oss()
    print("Training result:", result)