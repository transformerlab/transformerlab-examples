from unsloth import is_bfloat16_supported
import os
import torch
from datetime import datetime
import json
import io

from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from datasets import Audio

from trainer import CsmAudioTrainer, OrpheusAudioTrainer

from lab import lab

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
                checkpoints = [
                    d
                    for d in os.listdir(args.output_dir)
                    if d.startswith("checkpoint-")
                ]
                if checkpoints:
                    # Sort by checkpoint number
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    latest_checkpoint = checkpoints[-1]
                    checkpoint_dir = os.path.join(args.output_dir, latest_checkpoint)

                    # Save checkpoint to TransformerLab
                    try:
                        saved_path = lab.save_checkpoint(
                            checkpoint_dir, f"checkpoint-{state.global_step}"
                        )
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


def generate_audio_sample(model, tokenizer, text, sampling_rate, output_path, device="cuda"):
    """
    Generate a synthetic audio sample from text using the model.
    
    Args:
        model: The TTS model
        tokenizer/processor: The model's processor
        text: Text to synthesize
        sampling_rate: Audio sampling rate
        output_path: Path to save the audio file
        device: Device to use (cuda or cpu)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        
        model.eval()
        with torch.no_grad():
            # Prepare inputs
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            
            # Generate audio
            outputs = model.generate(**inputs)
            
            # Get audio array
            if hasattr(outputs, 'waveform'):
                audio_array = outputs.waveform.cpu().numpy()
            else:
                audio_array = outputs.cpu().numpy()
            
            # Ensure 1D array
            if len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()
            
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            # Convert to int16 for WAV file
            audio_int16 = np.int16(audio_array * 32767)
            
            # Save as WAV file
            wavfile.write(output_path, sampling_rate, audio_int16)
            return True
    except Exception as e:
        lab.log(f"⚠️  Error generating audio sample: {e}")
        return False


def save_audio_samples(model_before, tokenizer_before, model_after, tokenizer_after, 
                       text_sample, sampling_rate, output_dir, device="cuda"):
    """
    Generate and save before/after training audio samples.
    
    Args:
        model_before: Pre-trained model (before training)
        tokenizer_before: Pre-trained model's processor
        model_after: Fine-tuned model (after training)
        tokenizer_after: Fine-tuned model's processor
        text_sample: Sample text to synthesize
        sampling_rate: Audio sampling rate
        output_dir: Directory to save samples
        device: Device to use
    
    Returns:
        Tuple of (before_audio_path, after_audio_path) or (None, None) if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    before_audio_path = os.path.join(output_dir, "sample_before_training.wav")
    after_audio_path = os.path.join(output_dir, "sample_after_training.wav")
    
    lab.log(f"🎵 Generating audio samples with text: '{text_sample}'...")
    
    # Generate before training sample
    lab.log("Generating pre-trained model sample...")
    before_success = generate_audio_sample(
        model_before, tokenizer_before, text_sample, 
        sampling_rate, before_audio_path, device
    )
    
    if before_success:
        lab.log(f"✅ Generated before-training sample: {before_audio_path}")
    else:
        before_audio_path = None
    
    # Generate after training sample
    lab.log("Generating fine-tuned model sample...")
    after_success = generate_audio_sample(
        model_after, tokenizer_after, text_sample, 
        sampling_rate, after_audio_path, device
    )
    
    if after_success:
        lab.log(f"✅ Generated after-training sample: {after_audio_path}")
    else:
        after_audio_path = None
    
    return before_audio_path, after_audio_path



    """Train an audio model using unsloth."""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Initialize lab
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "unsloth/orpheus-3b-0.1-ft")
        dataset = config.get("dataset", "bosonai/EmergentTTS-Eval")
        audio_column_name = config.get("audio_column_name", "audio")
        text_column_name = config.get("text_column_name", "text_to_synthesize")
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.0)
        lora_r = config.get("lora_r", 16)
        maximum_sequence_length = config.get("maximum_sequence_length", 1024)
        max_grad_norm = config.get("max_grad_norm", 0.3)
        learning_rate = config.get("learning_rate", 5e-05)
        learning_rate_schedule = config.get("learning_rate_schedule", "linear")
        batch_size = config.get("batch_size", 1)
        num_train_epochs = config.get("num_train_epochs", 1)
        weight_decay = config.get("weight_decay", 0.0)
        adam_beta1 = config.get("adam_beta1", 0.9)
        adam_beta2 = config.get("adam_beta2", 0.999)
        adam_epsilon = config.get("adam_epsilon", 1e-08)
        sampling_rate = config.get("sampling_rate", 24000)
        max_steps = config.get("max_steps", -1)
        model_architecture = config.get("model_architecture", "OrpheusForConditionalGeneration")

        # Training configuration
        training_config = {
            "experiment_name": "unsloth-tts-training",
            "model_name": model_name,
            "dataset": dataset,
            "template_name": "unsloth-tts-demo",
            "output_dir": "./output",
            "log_to_wandb": False,
            "_config": {
                "dataset_name": dataset,
                "audio_column_name": audio_column_name,
                "text_column_name": text_column_name,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_r": lora_r,
                "maximum_sequence_length": maximum_sequence_length,
                "max_grad_norm": max_grad_norm,
                "learning_rate": learning_rate,
                "learning_rate_schedule": learning_rate_schedule,
                "batch_size": batch_size,
                "num_train_epochs": num_train_epochs,
                "weight_decay": weight_decay,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_epsilon": adam_epsilon,
                # "report_to": "wandb",
                "sampling_rate": sampling_rate,
                "max_steps": max_steps,
                "model_architecture": model_architecture,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
        }

        lab.set_config(training_config)

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

            datasets = load_dataset(training_config["dataset"])
            dataset = datasets["train"]
            lab.log(f"Loaded dataset with {len(datasets['train'])} training examples.")

            if (
                training_config["_config"]["audio_column_name"]
                not in dataset.column_names
                or training_config["_config"]["text_column_name"]
                not in dataset.column_names
            ):
                lab.log(
                    f"Missing required columns: '{training_config['_config']['audio_column_name']}' and '{training_config['_config']['text_column_name']}'."
                )
                lab.finish("Training failed due to missing dataset columns.")
                return {"status": "error", "error": "Missing required dataset columns."}

        except Exception as e:
            lab.log(f"❌ Failed to load dataset: {e}")
            lab.finish("Training failed due to dataset loading error.")
            return {"status": "error", "error": str(e)}

        lab.log("Preparing dataset...")
        try:
            # Getting the speaker id is important for multi-speaker models and speaker consistency
            speaker_key = "source"
            if (
                "source" not in dataset.column_names
                and "speaker_id" not in dataset.column_names
            ):
                print('No speaker found, adding default "source" of 0 for all examples')
                new_column = ["0"] * len(dataset)
                dataset = dataset.add_column("source", new_column)
            elif (
                "source" not in dataset.column_names
                and "speaker_id" in dataset.column_names
            ):
                speaker_key = "speaker_id"

            dataset = dataset.cast_column(
                training_config["_config"]["audio_column_name"],
                Audio(sampling_rate=training_config["_config"]["sampling_rate"]),
            )
            max_audio_length = max(
                len(example[training_config["_config"]["audio_column_name"]]["array"])
                for example in dataset
            )
        except Exception as e:
            lab.log(f"❌ Failed to prepare dataset: {e}")
            lab.finish("Training failed due to dataset preparation error.")
            return {"status": "error", "error": str(e)}

        # Load model and tokenizer using
        lab.log("Loading model and tokenizer and trainer...")
        try:
            model_name = training_config["model_name"]
            context_length = training_config["_config"]["maximum_sequence_length"]
            device = training_config["_config"]["device"]
            lora_r = training_config["_config"]["lora_r"]
            lora_alpha = training_config["_config"]["lora_alpha"]
            lora_dropout = training_config["_config"]["lora_dropout"]
            sampling_rate = training_config["_config"]["sampling_rate"]
            max_seq_length = training_config["_config"]["maximum_sequence_length"]
            audio_column_name = training_config["_config"]["audio_column_name"]
            text_column_name = training_config["_config"]["text_column_name"]
            batch_size = training_config["_config"]["batch_size"]

            if (
                training_config["_config"]["model_architecture"]
                == "CsmForConditionalGeneration"
            ):
                model_trainer = CsmAudioTrainer(
                    model_name=model_name,
                    speaker_key=speaker_key,
                    context_length=context_length,
                    device=device,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    sampling_rate=sampling_rate,
                    max_audio_length=max_audio_length,
                    audio_column_name=audio_column_name,
                    text_column_name=text_column_name,
                )

            elif (
                training_config["_config"]["model_architecture"]
                == "OrpheusForConditionalGeneration"
            ):
                model_trainer = OrpheusAudioTrainer(
                    model_name=model_name,
                    speaker_key=speaker_key,
                    context_length=max_seq_length,
                    device=device,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    sampling_rate=sampling_rate,
                    max_audio_length=max_audio_length,
                    batch_size=batch_size,
                    audio_column_name=audio_column_name,
                    text_column_name=text_column_name,
                )
            else:
                lab.log(
                    f"❌ Model architecture {training_config['_config']['model_architecture']} is not supported for audio training. Please use 'CsmForConditionalGeneration' or 'OrpheusForConditionalGeneration'."
                )
                lab.finish("Training failed due to unsupported model architecture.")
                return {"status": "error", "error": "Unsupported model architecture."}

            model = model_trainer.model
            tokenizer = model_trainer.processor
        except Exception as e:
            lab.log(f"❌ Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed due to model loading error.")
            return {"status": "error", "error": str(e)}

        lab.log("Preprocessing dataset...")

        try:
            processed_ds = dataset.map(
                model_trainer.preprocess_dataset,
                remove_columns=dataset.column_names,
                desc="Preprocessing dataset",
            )

            processed_ds = processed_ds.filter(lambda x: x is not None)

            lab.log(f"Processed dataset length: {len(processed_ds)}")

        except Exception as e:
            lab.log(f"❌ Failed to preprocess dataset: {e}")
            lab.finish("Training failed due to dataset preprocessing error.")
            return {"status": "error", "error": str(e)}

        lab.log("Setting up trainer...")
        try:
            progress_callback = LabCallback()

            trainer = Trainer(
                model=model,
                train_dataset=processed_ds,
                callbacks=[progress_callback],
                args=TrainingArguments(
                    logging_dir=f"{training_config['output_dir']}/logs",
                    num_train_epochs=training_config["_config"]["num_train_epochs"],
                    per_device_train_batch_size=training_config["_config"][
                        "batch_size"
                    ],
                    gradient_accumulation_steps=2,
                    warmup_ratio=0.03,
                    max_steps=training_config["_config"]["max_steps"],
                    learning_rate=training_config["_config"]["learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=10,
                    optim="adamw_8bit",
                    save_strategy="epoch",
                    weight_decay=training_config["_config"]["weight_decay"],
                    lr_scheduler_type=training_config["_config"][
                        "learning_rate_schedule"
                    ],
                    max_grad_norm=training_config["_config"]["max_grad_norm"],
                    adam_beta1=training_config["_config"]["adam_beta1"],
                    adam_beta2=training_config["_config"]["adam_beta2"],
                    adam_epsilon=training_config["_config"]["adam_epsilon"],
                    disable_tqdm=False,
                    seed=3407,
                    report_to="wandb" if training_config.get("log_to_wandb", True) else "none",
                    output_dir=training_config["output_dir"],
                    resume_from_checkpoint=checkpoint if checkpoint else None,
                ),
            )
            lab.log("Trainer setup complete.")

        except Exception as e:
            lab.log(f"❌ Failed to set up trainer: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed due to trainer setup error.")
            return {"status": "error", "error": str(e)}

        # Train the model
        lab.log("Starting training...")
        try:
            # Save pre-trained model reference for audio sample comparison
            pretrained_model = model
            pretrained_tokenizer = tokenizer
            
            trainer.train()
            lab.log("✅ Training completed successfully")

            # Save the fine-tuned model
            lab.log("Saving fine-tuned model...")
            model.save_pretrained(training_config["output_dir"])
            tokenizer.save_pretrained(training_config["output_dir"])
            lab.log("✅ Model and tokenizer saved")
            
            # Generate and save before/after training audio samples
            lab.log("📊 Generating before/after training audio samples...")
            try:
                # Use a sample text from the dataset or a default one
                sample_texts = [
                    "Hello, this is a test of the text-to-speech system.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Welcome to the audio synthesis demonstration.",
                ]
                
                # Get a sample text from the dataset if possible
                sample_text = sample_texts[0]
                if len(dataset) > 0:
                    try:
                        sample_data = dataset[0]
                        if training_config["_config"]["text_column_name"] in sample_data:
                            sample_text = sample_data[training_config["_config"]["text_column_name"]]
                            lab.log(f"Using dataset sample text: '{sample_text}'")
                    except Exception:
                        lab.log("Using default sample text for audio generation")
                
                before_audio, after_audio = save_audio_samples(
                    model_before=pretrained_model,
                    tokenizer_before=pretrained_tokenizer,
                    model_after=model,
                    tokenizer_after=tokenizer,
                    text_sample=sample_text,
                    sampling_rate=training_config["_config"]["sampling_rate"],
                    output_dir=training_config["output_dir"],
                    device=training_config["_config"]["device"]
                )
                
                # Save audio samples as artifacts
                if before_audio and os.path.exists(before_audio):
                    before_artifact_path = lab.save_artifact(
                        before_audio, 
                        "sample_before_training.wav"
                    )
                    lab.log(f"✅ Saved before-training audio artifact: {before_artifact_path}")
                
                if after_audio and os.path.exists(after_audio):
                    after_artifact_path = lab.save_artifact(
                        after_audio, 
                        "sample_after_training.wav"
                    )
                    lab.log(f"✅ Saved after-training audio artifact: {after_artifact_path}")
                    
            except Exception as e:
                lab.log(f"⚠️  Could not generate audio samples: {e}")
                import traceback
                traceback.print_exc()

            # Create training summary artifact
            progress_file = os.path.join(
                training_config["output_dir"], "training_summary.json"
            )

            with open(progress_file, "w") as f:
                json.dump(
                    {
                        "training_type": "Unsloth FastLanguageModel with LoRA",
                        "model_name": training_config["model_name"],
                        "dataset": training_config["dataset"],
                        "lora_r": training_config["_config"]["lora_r"],
                        "lora_alpha": training_config["_config"]["lora_alpha"],
                        "lora_dropout": training_config["_config"]["lora_dropout"],
                        "max_seq_length": training_config["_config"][
                            "maximum_sequence_length"
                        ],
                        "learning_rate": training_config["_config"]["learning_rate"],
                        "batch_size": training_config["_config"]["batch_size"],
                        "gradient_accumulation_steps": 2,
                        "completed_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            progress_artifact_path = lab.save_artifact(
                progress_file, "training_summary.json"
            )
            lab.log(f"Saved training summary: {progress_artifact_path}")

        except Exception as e:
            lab.log(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
            lab.finish("Training failed")
            return {"status": "error", "error": str(e)}

        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

        # Save final artifacts
        final_model_file = os.path.join(
            training_config["output_dir"], "final_model_summary.txt"
        )
        with open(final_model_file, "w") as f:
            f.write("Final Model Summary\n")
            f.write("==================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Model: {training_config['model_name']}\n")
            f.write(f"Dataset: {training_config['dataset']}\n")
            f.write(f"LoRA Rank: {training_config['_config']['lora_r']}\n")
            f.write(f"LoRA Alpha: {training_config['_config']['lora_alpha']}\n")
            f.write(f"Completed at: {end_time}\n")

        final_model_path = lab.save_artifact(
            final_model_file, "final_model_summary.txt"
        )
        lab.log(f"Saved final model summary: {final_model_path}")

        # Save training configuration as artifact
        config_file = os.path.join(
            training_config["output_dir"], "training_config.json"
        )
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
            if file.endswith(
                (".bin", ".safetensors", ".json", ".txt")
            ) and not file.startswith("checkpoint"):
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
            "trainer_type": "Unsloth TTS Trainer",
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
    result = train_model()
    print("Training result:", result)
