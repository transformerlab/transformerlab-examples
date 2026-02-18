#!/usr/bin/env python3
"""
MLX LoRA Training Script with TransformerLab Integration

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Using lab SDK for progress tracking, logging, and artifact saving
- Training with MLX LoRA on Apple Silicon
- Automatic checkpoint saving and model fusion
"""

import os
import re
import subprocess
import time
import json
import yaml
from datetime import datetime

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def prepare_dataset_files(data_directory, train_dataset, valid_dataset, formatting_template=None, chat_template=None, model_name=None, chat_column="messages"):
    """
    Prepare dataset files for MLX training.
    MLX expects train.jsonl and valid.jsonl files in a specific format.
    """
    import datasets
    from transformers import AutoTokenizer
    
    os.makedirs(data_directory, exist_ok=True)
    
    # Helper function to format examples
    def format_example(example):
        # If using chat template
        if chat_template and chat_column in example:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if hasattr(tokenizer, 'apply_chat_template'):
                    formatted_text = tokenizer.apply_chat_template(
                        example[chat_column],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    return {"text": formatted_text}
            except Exception as e:
                lab.log(f"Error applying chat template: {e}")
        
        # If using simple formatting template
        if formatting_template:
            try:
                formatted_text = formatting_template.format(**example)
                return {"text": formatted_text}
            except Exception as e:
                lab.log(f"Error applying formatting template: {e}")
        
        # Default: use 'text' field or convert to string
        if "text" in example:
            return {"text": example["text"]}
        else:
            return {"text": str(example)}
    
    # Process and save train dataset
    if train_dataset:
        lab.log(f"Preparing training dataset with {len(train_dataset)} examples...")
        train_formatted = []
        for example in train_dataset:
            formatted = format_example(example)
            if formatted and "text" in formatted:
                train_formatted.append(formatted)
        
        train_file = os.path.join(data_directory, "train.jsonl")
        with open(train_file, "w") as f:
            for item in train_formatted:
                f.write(json.dumps(item) + "\n")
        lab.log(f"Saved {len(train_formatted)} training examples to {train_file}")
    
    # Process and save validation dataset
    if valid_dataset:
        lab.log(f"Preparing validation dataset with {len(valid_dataset)} examples...")
        valid_formatted = []
        for example in valid_dataset:
            formatted = format_example(example)
            if formatted and "text" in formatted:
                valid_formatted.append(formatted)
        
        valid_file = os.path.join(data_directory, "valid.jsonl")
        with open(valid_file, "w") as f:
            for item in valid_formatted:
                f.write(json.dumps(item) + "\n")
        lab.log(f"Saved {len(valid_formatted)} validation examples to {valid_file}")


def train_mlx_lora():
    """Train a model using MLX LoRA."""
    
    try:
        # Initialize lab
        lab.init()
        
        # Get parameters from task configuration
        config = lab.get_config()
        
        # Extract parameters with defaults
        model_name = config.get("model_name", "mlx-community/Llama-3.2-1B-Instruct-4bit")
        dataset_name = config.get("dataset", "yahma/alpaca-cleaned")
        lora_layers = config.get("lora_layers", 16)
        learning_rate = config.get("learning_rate", 5e-5)
        batch_size = config.get("batch_size", 4)
        steps_per_eval = config.get("steps_per_eval", 200)
        iters = config.get("iters", 1000)
        adaptor_name = config.get("adaptor_name", "adaptor")
        fuse_model = config.get("fuse_model", True)
        num_train_epochs = config.get("num_train_epochs", -1)
        steps_per_report = config.get("steps_per_report", 100)
        save_every = config.get("save_every", 100)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        chat_template = config.get("formatting_chat_template", None)
        chat_column = config.get("chatml_formatted_column", "messages")
        formatting_template = config.get("formatting_template", None)
        log_to_wandb = config.get("log_to_wandb", True)
        
        # Convert to appropriate types
        lora_layers = int(lora_layers)
        learning_rate = float(learning_rate)
        batch_size = int(batch_size)
        steps_per_eval = int(steps_per_eval)
        iters = int(iters)
        steps_per_report = int(steps_per_report)
        save_every = int(save_every)
        lora_rank = int(lora_rank) if lora_rank else None
        lora_alpha = int(lora_alpha) if lora_alpha else None
        
        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"🚀 Training started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"LoRA Layers: {lora_layers}")
        lab.log(f"LoRA Rank: {lora_rank}")
        lab.log(f"LoRA Alpha: {lora_alpha}")
        lab.log(f"Learning Rate: {learning_rate}")
        lab.log(f"Batch Size: {batch_size}")
        lab.log(f"Iterations: {iters}")
        
        lab.update_progress(5)
        
        # Calculate iterations based on epochs if specified
        if num_train_epochs is not None and num_train_epochs >= 0:
            if num_train_epochs == 0:
                lab.log("Training is set to 0 epochs which is not allowed. Setting to 1 epoch.")
                num_train_epochs = 1
        
        # Create LoRA config file if rank/alpha are specified
        config_file = None
        if lora_rank and lora_alpha:
            config_file = os.path.join("./output", "lora_config.yaml")
            os.makedirs("./output", exist_ok=True)
            
            lora_scale = lora_alpha / lora_rank
            lora_config = {
                "lora_parameters": {
                    "alpha": lora_alpha,
                    "rank": lora_rank,
                    "scale": lora_scale,
                    "dropout": 0.0
                }
            }
            
            with open(config_file, "w") as f:
                yaml.dump(lora_config, f)
            
            lab.log(f"LoRA config created: {lora_config}")
        
        lab.update_progress(10)
        
        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset
            
            datasets = load_dataset(dataset_name)
            
            # Get train and validation splits
            train_dataset = None
            valid_dataset = None
            
            if "train" in datasets:
                train_dataset = datasets["train"]
                lab.log(f"Loaded training dataset with {len(train_dataset)} examples")
            
            # Try to get validation set
            if "validation" in datasets:
                valid_dataset = datasets["validation"]
            elif "test" in datasets:
                valid_dataset = datasets["test"]
            elif "valid" in datasets:
                valid_dataset = datasets["valid"]
            elif train_dataset:
                # Split train into train/valid if no validation set exists
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                valid_dataset = split_dataset["test"]
                lab.log(f"Split dataset into {len(train_dataset)} train and {len(valid_dataset)} validation examples")
            
            if valid_dataset:
                lab.log(f"Loaded validation dataset with {len(valid_dataset)} examples")
            
            # Calculate iterations based on epochs if needed
            if num_train_epochs is not None and num_train_epochs > 0 and train_dataset:
                steps_per_epoch = len(train_dataset) // batch_size
                if steps_per_epoch == 0:
                    steps_per_epoch = 1
                total_steps = steps_per_epoch * num_train_epochs
                iters = total_steps
                lab.log(f"Using epoch-based training: {num_train_epochs} epochs")
                lab.log(f"Steps per epoch: {steps_per_epoch}")
                lab.log(f"Total training iterations: {iters}")
        
        except Exception as e:
            lab.log(f"❌ Failed to load dataset: {e}")
            lab.finish("Training failed due to dataset loading error.")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(20)
        
        # Prepare dataset files for MLX
        lab.log("Preparing dataset files for MLX...")
        data_directory = os.path.join("./output", "data")
        try:
            prepare_dataset_files(
                data_directory=data_directory,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                formatting_template=formatting_template,
                chat_template=chat_template,
                model_name=model_name,
                chat_column=chat_column
            )
        except Exception as e:
            lab.log(f"❌ Failed to prepare dataset: {e}")
            lab.finish("Training failed due to dataset preparation error.")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(30)
        
        # Set output directory for the adaptor
        adaptor_output_dir = os.path.join("./output", "adaptors", adaptor_name)
        os.makedirs(adaptor_output_dir, exist_ok=True)
        lab.log(f"Adaptor will be saved to: {adaptor_output_dir}")
        
        # Prepare the MLX LoRA training command
        popen_command = [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            model_name,
            "--iters",
            str(iters),
            "--train",
            "--adapter-path",
            adaptor_output_dir,
            "--num-layers",
            str(lora_layers),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--data",
            data_directory,
            "--steps-per-report",
            str(steps_per_report),
            "--steps-per-eval",
            str(steps_per_eval),
            "--save-every",
            str(save_every),
        ]
        
        # Add config file if it exists
        if config_file:
            popen_command.extend(["--config", config_file])
        
        lab.log("Running MLX LoRA training command:")
        lab.log(" ".join(popen_command))
        
        lab.update_progress(35)
        
        # Track start time for ETA calculation
        training_start_time = time.time()
        
        # Run the MLX LoRA training process
        lab.log("🚀 Training beginning...")
        try:
            with subprocess.Popen(
                popen_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            ) as process:
                for line in process.stdout:
                    # Parse progress from output
                    pattern = r"Iter (\d+):"
                    match = re.search(pattern, line)
                    if match:
                        iteration = int(match.group(1))
                        percent_complete = (iteration / iters) * 100
                        # Scale progress from 35% to 85%
                        scaled_progress = 35 + (percent_complete * 0.5)
                        lab.update_progress(int(scaled_progress))
                        
                        # Calculate ETA
                        if iteration > 0:
                            elapsed_time = time.time() - training_start_time
                            iterations_remaining = iters - iteration
                            if iterations_remaining > 0:
                                avg_time_per_iter = elapsed_time / iteration
                                estimated_time_remaining = avg_time_per_iter * iterations_remaining
                                lab.log(f"Progress: {percent_complete:.2f}% (Iter {iteration}/{iters}) - ETA: {int(estimated_time_remaining)}s")
                        
                        # Parse training metrics
                        train_pattern = r"Train loss (\d+\.\d+), Learning Rate (\d+\.[e\-\d]+), It/sec (\d+\.\d+), Tokens/sec (\d+\.\d+)"
                        train_match = re.search(train_pattern, line)
                        if train_match:
                            loss = float(train_match.group(1))
                            it_per_sec = float(train_match.group(3))
                            tokens_per_sec = float(train_match.group(4))
                            lab.log(f"  Loss: {loss:.4f}, It/sec: {it_per_sec:.2f}, Tokens/sec: {tokens_per_sec:.2f}")
                        
                        # Parse validation metrics
                        val_pattern = r"Val loss (\d+\.\d+), Val took (\d+\.\d+)s"
                        val_match = re.search(val_pattern, line)
                        if val_match:
                            val_loss = float(val_match.group(1))
                            val_time = float(val_match.group(2))
                            lab.log(f"  Validation Loss: {val_loss:.4f} (took {val_time:.2f}s)")
                    
                    # Print all output
                    print(line, end="", flush=True)
                
                # Check return code
                return_code = process.wait()
                if return_code != 0:
                    lab.log(f"❌ Training failed with return code: {return_code}")
                    lab.finish("Training failed")
                    return {"status": "error", "error": f"Training process failed with code {return_code}"}
        
        except Exception as e:
            lab.log(f"❌ Training process error: {e}")
            lab.finish("Training failed")
            return {"status": "error", "error": str(e)}
        
        lab.log("✅ Training completed successfully")
        lab.update_progress(85)
        
        # Save adaptor as artifact
        try:
            adaptor_artifact_path = lab.save_artifact(adaptor_output_dir, f"adaptor_{adaptor_name}")
            lab.log(f"✅ Saved adaptor to artifacts: {adaptor_artifact_path}")
        except Exception as e:
            lab.log(f"⚠️  Could not save adaptor to artifacts: {e}")
        
        # Fuse the model if requested
        final_model_path = None
        if fuse_model:
            lab.log("🔧 Fusing adaptor with base model...")
            
            # Extract model name for fused model
            base_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
            fused_model_name = f"{base_model_name}_{adaptor_name}_fused"
            fused_model_location = os.path.join("./output", "models", fused_model_name)
            os.makedirs(fused_model_location, exist_ok=True)
            
            fuse_command = [
                "python",
                "-m",
                "mlx_lm.fuse",
                "--model",
                model_name,
                "--adapter-path",
                adaptor_output_dir,
                "--save-path",
                fused_model_location,
            ]
            
            lab.log("Running fusion command:")
            lab.log(" ".join(fuse_command))
            
            try:
                with subprocess.Popen(
                    fuse_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    
                    return_code = process.wait()
                    if return_code == 0:
                        lab.log("✅ Model fusion completed successfully")
                        final_model_path = fused_model_location
                        
                        # Save fused model as artifact
                        try:
                            model_artifact_path = lab.save_model(fused_model_location, name=fused_model_name)
                            lab.log(f"✅ Saved fused model to artifacts: {model_artifact_path}")
                        except Exception as e:
                            lab.log(f"⚠️  Could not save fused model to artifacts: {e}")
                    else:
                        lab.log(f"⚠️  Model fusion failed with return code: {return_code}")
            
            except Exception as e:
                lab.log(f"⚠️  Error during model fusion: {e}")
        else:
            lab.log("Skipping model fusion (fuse_model=False)")
            final_model_path = adaptor_output_dir
        
        lab.update_progress(95)
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")
        
        # Save training summary
        summary_data = {
            "training_type": "MLX LoRA",
            "model_name": model_name,
            "dataset": dataset_name,
            "lora_layers": lora_layers,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "iterations": iters,
            "adaptor_name": adaptor_name,
            "fused": fuse_model,
            "training_duration": str(training_duration),
            "completed_at": end_time.isoformat()
        }
        
        summary_file = os.path.join("./output", "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        summary_artifact_path = lab.save_artifact(summary_file, "training_summary.json")
        lab.log(f"✅ Saved training summary: {summary_artifact_path}")
        
        # Finish wandb run if it was initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                lab.log("✅ Wandb run finished")
        except Exception:
            pass
        
        lab.update_progress(100)
        
        # Complete the job
        lab.finish("Training completed successfully with MLX LoRA")
        
        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "adaptor_path": adaptor_output_dir,
            "model_path": final_model_path,
            "trainer_type": "MLX LoRA",
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
    print("🚀 Starting MLX LoRA training...")
    result = train_mlx_lora()
    print("Training result:", result)
