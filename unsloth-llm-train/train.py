#!/usr/bin/env python3
"""
Training script using Unsloth FastLanguageModel with LoRA fine-tuning.
Demonstrates efficient LLM training with 4-bit quantization and LoRA adapters.
"""

from unsloth import FastLanguageModel
import os
from datetime import datetime

from lab import lab
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


class LabCallback(TrainerCallback):
    """Custom callback to update TransformerLab progress and save checkpoints"""

    def __init__(self):
        self.training_started = False
        self.total_steps = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training begins"""
        lab.log("🚀 Training started with Unsloth FastLanguageModel")
        self.training_started = True
        if state.max_steps and state.max_steps > 0:
            self.total_steps = state.max_steps
        else:
            # Estimate steps if not provided
            self.total_steps = 1000

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after each training step"""
        if self.total_steps:
            progress = int((state.global_step / self.total_steps) * 100)
            progress = min(progress, 95)  # Keep some buffer for final operations
            lab.update_progress(progress)

        # Log training metrics if available
        if state.log_history:
            latest_log = state.log_history[-1]
            if "loss" in latest_log:
                lab.log(f"Step {state.global_step}: loss={latest_log['loss']:.4f}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when a checkpoint is saved"""
        lab.log(f"💾 Checkpoint saved at step {state.global_step}")

        # Attempt to save the checkpoint using lab's checkpoint mechanism
        if hasattr(args, "output_dir"):
            checkpoint_dir = None
            # Find the most recent checkpoint
            if os.path.exists(args.output_dir):
                checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    # Sort by checkpoint number
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    latest_checkpoint = checkpoints[-1]
                    checkpoint_dir = os.path.join(args.output_dir, latest_checkpoint)

                    # Save checkpoint to TransformerLab
                    try:
                        saved_path = lab.save_checkpoint(checkpoint_dir, f"checkpoint-{state.global_step}")
                        lab.log(f"✅ Saved checkpoint to TransformerLab: {saved_path}")
                    except Exception as e:
                        lab.log(f"⚠️  Could not save checkpoint to TransformerLab: {e}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each epoch"""
        if state.epoch:
            lab.log(f"📊 Completed epoch {int(state.epoch)} / {args.num_train_epochs}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training ends"""
        lab.log("✅ Training completed successfully")
        lab.update_progress(95)


def train_with_unsloth():
    """Training function using Unsloth FastLanguageModel with LoRA fine-tuning"""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Training configuration
    training_config = {
        "experiment_name": "unsloth-lora-training",
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct",  # Small model for testing
        "dataset": "Trelis/touch-rugby-rules",  # Example dataset
        "template_name": "unsloth-demo",
        "output_dir": "./output",
        "log_to_wandb": True,
        "_config": {
            "dataset_name": "Trelis/touch-rugby-rules",
            "lr": 2e-4,
            "num_train_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 100,
            "max_seq_length": 2048,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "logging_steps": 1,
            "save_steps": 50,
            "weight_decay": 0.01,
            "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        },
    }

    try:
        # Initialize lab with default/simple API
        lab.init()
        lab.set_config(training_config)

        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")

        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Create output directory if it doesn't exist
        os.makedirs(training_config["output_dir"], exist_ok=True)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            dataset = load_dataset(training_config["dataset"])
            lab.log(f"Loaded dataset with {len(dataset['train'])} examples")

        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            # Create a small fake dataset for testing
            from datasets import Dataset

            dataset = {
                "train": Dataset.from_list(
                    [
                        {
                            "instruction": "What are the rules of touch rugby?",
                            "output": "Touch rugby is a non-contact sport...",
                        },
                        {
                            "instruction": "How many players are on a touch rugby team?",
                            "output": "A touch rugby team has 6 players on the field...",
                        },
                        {
                            "instruction": "What is the objective of touch rugby?",
                            "output": "The objective is to score tries by touching the ball down...",
                        },
                    ]
                )
            }
            lab.log("Using fake dataset for testing")

        lab.update_progress(20)

        # Load model and tokenizer using Unsloth
        lab.log("Loading model and tokenizer with Unsloth...")
        try:
            import torch

            model_name = training_config["model_name"]
            max_seq_length = training_config["_config"]["max_seq_length"]

            # Load model with 4-bit quantization for memory efficiency
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=True,  # Enable 4-bit quantization
                use_gradient_checkpointing="unsloth",  # Efficient backpropagation
            )

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            lab.log(f"✅ Loaded model: {model_name}")

            # Add LoRA adapters for efficient fine-tuning
            lab.log("Adding LoRA adapters...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=training_config["_config"]["lora_r"],  # Rank of LoRA matrices
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=training_config["_config"]["lora_alpha"],  # Scaling factor
                lora_dropout=training_config["_config"]["lora_dropout"],  # Dropout rate
                bias="none",  # Don't train bias
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

            lab.log("✅ LoRA adapters added successfully")

        except ImportError as e:
            lab.log(f"⚠️  Unsloth not available: {e}")
            lab.log("Install with: pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo")
            lab.finish("Training skipped - unsloth not available")
            return {"status": "skipped", "reason": "unsloth not available"}
        except Exception as e:
            lab.log(f"Error loading model: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed - model loading error")
            return {"status": "error", "error": str(e)}

        lab.update_progress(40)

        # Prepare dataset with chat template
        lab.log("Preparing dataset with chat template...")
        try:
            # Apply chat template to format the dataset
            def format_dataset(example):
                # Handle different dataset formats - process one example at a time
                if "instruction" in example and "output" in example:
                    # Format: instruction-output pairs
                    instruction = example["instruction"]
                    output = example["output"]
                    # Use the model's chat template if available
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        messages = [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": output},
                        ]
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    else:
                        # Fallback format
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    return {"text": text}
                elif "text" in example:
                    # Already formatted - ensure it's a string
                    text = example["text"]
                    if isinstance(text, list):
                        text = "\n".join(str(t) for t in text)
                    return {"text": str(text)}
                else:
                    # Try to use first text-like field
                    for key in example.keys():
                        if "text" in key.lower() or "content" in key.lower():
                            text = example[key]
                            if isinstance(text, list):
                                text = "\n".join(str(t) for t in text)
                            return {"text": str(text)}
                    # Last resort: combine all fields
                    text_parts = [f"{k}: {v}" for k, v in example.items()]
                    return {"text": "\n".join(text_parts)}

            # Apply formatting - process one example at a time to avoid batching issues
            dataset["train"] = dataset["train"].map(
                format_dataset,
                batched=False,  # Process one at a time to avoid nested structures
                remove_columns=[col for col in dataset["train"].column_names if col != "text"],
            )

            lab.log("✅ Dataset formatted successfully")
            lab.log(f"Sample text length: {len(dataset['train'][0]['text']) if len(dataset['train']) > 0 else 0}")

        except Exception as e:
            lab.log(f"⚠️  Error formatting dataset: {e}")
            import traceback

            traceback.print_exc()

        lab.update_progress(50)

        # Set up SFTTrainer
        lab.log("Setting up SFTTrainer...")
        try:
            from trl import SFTTrainer, SFTConfig

            # Training arguments optimized for Unsloth
            training_args = SFTConfig(
                output_dir=training_config["output_dir"],
                num_train_epochs=training_config["_config"]["num_train_epochs"],
                per_device_train_batch_size=training_config["_config"]["batch_size"],
                gradient_accumulation_steps=training_config["_config"]["gradient_accumulation_steps"],
                learning_rate=training_config["_config"]["lr"],
                warmup_steps=training_config["_config"]["warmup_steps"],
                max_steps=training_config["_config"]["max_steps"],
                weight_decay=training_config["_config"]["weight_decay"],
                logging_steps=training_config["_config"]["logging_steps"],
                save_steps=training_config["_config"]["save_steps"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
                lr_scheduler_type="linear",
                seed=3407,
                logging_dir=f"{training_config['output_dir']}/logs",
                remove_unused_columns=False,
                push_to_hub=False,
                dataset_text_field="text",  # SFTTrainer will automatically tokenize this field
                max_seq_length=training_config["_config"]["max_seq_length"],
                packing=False,  # Don't pack sequences
                resume_from_checkpoint=checkpoint if checkpoint else None,
                save_total_limit=3,  # Keep only the last 3 checkpoints
                save_strategy="steps",
                load_best_model_at_end=False,
                dataloader_num_workers=training_config["_config"]["dataloader_num_workers"],
            )

            # Create custom callback for TransformerLab integration
            transformerlab_callback = LabCallback()

            # SFTTrainer handles tokenization automatically when using dataset_text_field
            # No need for a custom data collator
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                tokenizer=tokenizer,
                callbacks=[transformerlab_callback],
            )

            lab.log("✅ SFTTrainer created successfully")

        except Exception as e:
            lab.log(f"Error setting up SFTTrainer: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed - trainer setup error")
            return {"status": "error", "error": str(e)}

        lab.update_progress(60)

        # Start training
        lab.log("🚀 Starting training...")

        try:
            # Train the model
            trainer.train()
            lab.log("✅ Training completed successfully")

            # Save the fine-tuned model
            lab.log("Saving fine-tuned model...")
            model.save_pretrained(training_config["output_dir"])
            tokenizer.save_pretrained(training_config["output_dir"])
            lab.log("✅ Model and tokenizer saved")

            # Create training summary artifact
            progress_file = os.path.join(training_config["output_dir"], "training_summary.json")
            import json

            with open(progress_file, "w") as f:
                json.dump(
                    {
                        "training_type": "Unsloth FastLanguageModel with LoRA",
                        "model_name": training_config["model_name"],
                        "dataset": training_config["dataset"],
                        "lora_r": training_config["_config"]["lora_r"],
                        "lora_alpha": training_config["_config"]["lora_alpha"],
                        "lora_dropout": training_config["_config"]["lora_dropout"],
                        "max_seq_length": training_config["_config"]["max_seq_length"],
                        "learning_rate": training_config["_config"]["lr"],
                        "batch_size": training_config["_config"]["batch_size"],
                        "gradient_accumulation_steps": training_config["_config"]["gradient_accumulation_steps"],
                        "completed_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            progress_artifact_path = lab.save_artifact(progress_file, "training_summary.json")
            lab.log(f"Saved training summary: {progress_artifact_path}")

        except Exception as e:
            lab.log(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed")
            return {"status": "error", "error": str(e)}

        lab.update_progress(90)

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

        # Save final artifacts
        final_model_file = os.path.join(training_config["output_dir"], "final_model_summary.txt")
        with open(final_model_file, "w") as f:
            f.write("Final Model Summary\n")
            f.write("==================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Model: {training_config['model_name']}\n")
            f.write(f"Dataset: {training_config['dataset']}\n")
            f.write(f"LoRA Rank: {training_config['_config']['lora_r']}\n")
            f.write(f"LoRA Alpha: {training_config['_config']['lora_alpha']}\n")
            f.write(f"Completed at: {end_time}\n")

        final_model_path = lab.save_artifact(final_model_file, "final_model_summary.txt")
        lab.log(f"Saved final model summary: {final_model_path}")

        # Save training configuration as artifact
        config_file = os.path.join(training_config["output_dir"], "training_config.json")
        with open(config_file, "w") as f:
            json.dump(training_config, f, indent=2)

        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")

        # Save the trained model
        model_dir = os.path.join(training_config["output_dir"], "final_model")
        os.makedirs(model_dir, exist_ok=True)

        # Copy model files to final_model directory
        import shutil

        for file in os.listdir(training_config["output_dir"]):
            if file.endswith((".bin", ".safetensors", ".json", ".txt")) and not file.startswith("checkpoint"):
                src = os.path.join(training_config["output_dir"], file)
                dst = os.path.join(model_dir, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        saved_path = lab.save_model(model_dir, name="unsloth_trained_model")
        lab.log(f"✅ Model saved to job models directory: {saved_path}")

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
        lab.finish("Training completed successfully with Unsloth")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": training_config["output_dir"],
            "saved_model_path": saved_path,
            "trainer_type": "Unsloth FastLanguageModel",
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
    print("🚀 Starting Unsloth training...")
    result = train_with_unsloth()
    print("Training result:", result)
