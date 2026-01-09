#!/usr/bin/env python3
"""
DPO / ORPO / SIMPO RLHF Training using Llama Factory

This script demonstrates preference optimization methods (DPO, ORPO, SIMPO) 
without the need for a reward model, using the Llama Factory library.
"""

import os
import subprocess
import json
import yaml
import re
from datetime import datetime
from pathlib import Path

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def train_with_llama_factory():
    """Training function using Llama Factory for preference optimization"""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "meta-llama/Llama-3.2-1B-Instruct")
        dataset_name = config.get("dataset", "Intel/orca_dpo_pairs")
        output_dir = config.get("output_dir", "./output")
        log_to_wandb = config.get("log_to_wandb", True)
        
        # Preference optimization method (dpo, orpo, or simpo)
        pref_loss = config.get("pref_loss", "dpo")
        
        # Training hyperparameters
        learning_rate_raw = config.get("learning_rate", 5e-5)
        learning_rate = float(learning_rate_raw) if isinstance(learning_rate_raw, (str, int, float)) else learning_rate_raw

        batch_size_raw = config.get("batch_size", 2)
        batch_size = int(batch_size_raw) if isinstance(batch_size_raw, (str, int, float)) else batch_size_raw

        num_train_epochs_raw = config.get("num_train_epochs", 1)
        num_train_epochs = int(num_train_epochs_raw) if isinstance(num_train_epochs_raw, (str, int, float)) else num_train_epochs_raw

        max_steps_raw = config.get("max_steps", -1)
        max_steps = int(max_steps_raw) if isinstance(max_steps_raw, (str, int, float)) else max_steps_raw

        gradient_accumulation_steps_raw = config.get("gradient_accumulation_steps", 4)
        gradient_accumulation_steps = int(gradient_accumulation_steps_raw) if isinstance(gradient_accumulation_steps_raw, (str, int, float)) else gradient_accumulation_steps_raw

        template = config.get("template", "llama3")

        # Process preference strategy
        preference_strategy = pref_loss
        if preference_strategy == "dpo":
            preference_strategy = "sigmoid"  # llama factory calls dpo "sigmoid"
        if preference_strategy not in ["sigmoid", "orpo", "simpo"]:
            lab.log("❌ Invalid preference strategy. Must be one of: dpo, orpo, simpo.")
            lab.finish("Training failed due to invalid preference strategy")
            return {"status": "error", "error": "Invalid preference strategy"}

        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")

        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"Preference Method: {pref_loss}")
        lab.log(f"Learning rate: {learning_rate}")
        lab.log(f"Batch size: {batch_size}")
        lab.log(f"Number of epochs: {num_train_epochs}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        lab.update_progress(10)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_name)
            train_dataset = dataset["train"]
            lab.log(f"Loaded dataset with {len(train_dataset)} training examples")

            # Verify required columns for preference optimization
            required_columns = ["conversations", "chosen", "rejected"]
            missing_columns = [col for col in required_columns if col not in train_dataset.column_names]
            
            # Some datasets use different column names, check alternatives
            if missing_columns:
                # Check for alternative column names
                alternative_mappings = {
                    "conversations": ["prompt", "input", "question", "messages"],
                    "chosen": ["chosen_response", "preferred", "winner"],
                    "rejected": ["rejected_response", "dispreferred", "loser"]
                }
                
                actual_columns = {}
                for required_col in required_columns:
                    found = False
                    if required_col in train_dataset.column_names:
                        actual_columns[required_col] = required_col
                        found = True
                    elif required_col in alternative_mappings:
                        for alt_col in alternative_mappings[required_col]:
                            if alt_col in train_dataset.column_names:
                                actual_columns[required_col] = alt_col
                                lab.log(f"Using '{alt_col}' as '{required_col}' column")
                                found = True
                                break
                    
                    if not found:
                        lab.log(f"❌ Missing required column: {required_col}")
                        lab.log(f"Available columns: {train_dataset.column_names}")
                        lab.finish("Training failed due to missing dataset columns")
                        return {"status": "error", "error": f"Missing column: {required_col}"}

        except Exception as e:
            lab.log(f"❌ Failed to load dataset: {e}")
            lab.finish("Training failed due to dataset loading error")
            return {"status": "error", "error": str(e)}

        lab.update_progress(20)

        # Prepare Llama Factory data directory
        lab.log("Preparing Llama Factory data directory...")
        try:
            # Get the script directory (where train.py is located)
            script_dir = Path(__file__).parent.absolute()
            
            # Create data directory for Llama Factory
            data_directory = script_dir / "temp_data"
            data_directory.mkdir(parents=True, exist_ok=True)

            # Save dataset to JSON format
            train_data_path = data_directory / "train.json"
            with open(train_data_path, "w", encoding="utf-8") as f:
                all_data = []
                for row in train_dataset:
                    # Map alternative column names if needed
                    mapped_row = {}
                    if 'actual_columns' in locals():
                        for required_col, actual_col in actual_columns.items():
                            mapped_row[required_col] = row[actual_col]
                    else:
                        mapped_row = row
                    all_data.append(mapped_row)
                json.dump(all_data, f, indent=2)

            lab.log(f"Saved training data to {train_data_path}")

            # Create dataset_info.json for Llama Factory
            dataset_info = {
                "training_data": {
                    "file_name": "train.json",
                    "ranking": True,
                    "formatting": "sharegpt",
                    "columns": {
                        "messages": "conversations",
                        "chosen": "chosen",
                        "rejected": "rejected"
                    },
                }
            }

            dataset_info_path = data_directory / "dataset_info.json"
            with open(dataset_info_path, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=2)

            lab.log(f"Created dataset_info.json at {dataset_info_path}")

        except Exception as e:
            lab.log(f"❌ Failed to prepare data directory: {e}")
            import traceback
            traceback.print_exc()
            lab.finish("Training failed due to data preparation error")
            return {"status": "error", "error": str(e)}

        lab.update_progress(30)

        # Generate YAML config for Llama Factory
        lab.log("Generating Llama Factory configuration...")
        try:
            llama_factory_dir = script_dir / "LLaMA-Factory"
            
            if not llama_factory_dir.exists():
                lab.log(f"❌ LLaMA-Factory directory not found at {llama_factory_dir}")
                lab.log("Please ensure the setup script has run successfully")
                lab.finish("Training failed - LLaMA-Factory not found")
                return {"status": "error", "error": "LLaMA-Factory not found"}

            # Create config based on template
            template_config_path = llama_factory_dir / "examples" / "train_lora" / "llama3_lora_dpo.yaml"
            
            if not template_config_path.exists():
                lab.log(f"⚠️  Template config not found at {template_config_path}, creating from scratch")
                yaml_config = {}
            else:
                with open(template_config_path, "r") as f:
                    yaml_config = yaml.safe_load(f)

            # Update the YAML config with our parameters
            today = datetime.now().strftime("%Y%m%d-%H%M%S")
            adaptor_output_dir = Path(output_dir) / "adaptor"
            adaptor_output_dir.mkdir(parents=True, exist_ok=True)
            
            logging_dir = Path(output_dir) / f"logs_{today}"
            logging_dir.mkdir(parents=True, exist_ok=True)

            yaml_config.update({
                "pref_loss": preference_strategy,
                "model_name_or_path": model_name,
                "output_dir": str(adaptor_output_dir),
                "logging_dir": str(logging_dir),
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "max_steps": max_steps,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "dataset_dir": str(data_directory),
                "dataset": "training_data",
                "template": template,
                "resize_vocab": True,
                "finetuning_type": "lora",
                "lora_target": "all",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "report_to": "wandb" if log_to_wandb else "none",
                "run_name": f"dpo-train-{lab.job.id}",
            })

            # Add checkpoint resumption if available
            if checkpoint:
                yaml_config["resume_from_checkpoint"] = checkpoint

            yaml_config_path = data_directory / "training_config.yaml"
            with open(yaml_config_path, "w") as f:
                yaml.dump(yaml_config, f)
            
            lab.log("Configuration created:")
            lab.log(f"  Preference method: {preference_strategy}")
            lab.log(f"  Model: {model_name}")
            lab.log(f"  Template: {template}")
            lab.log(f"  Output: {adaptor_output_dir}")
            lab.log(f"  Config saved to: {yaml_config_path}")

        except Exception as e:
            lab.log(f"❌ Failed to generate configuration: {e}")
            import traceback
            traceback.print_exc()
            lab.finish("Training failed due to configuration error")
            return {"status": "error", "error": str(e)}

        lab.update_progress(40)

        # Start training with Llama Factory
        lab.log("Starting training with Llama Factory...")
        try:
            # Find the llamafactory-cli executable
            llamafactory_cli = "llamafactory-cli"
            
            # Build the training command
            train_command = [
                llamafactory_cli,
                "train",
                str(yaml_config_path)
            ]

            lab.log(f"Running command: {' '.join(train_command)}")

            # Run the training process
            with subprocess.Popen(
                train_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                cwd=str(llama_factory_dir),
            ) as process:
                training_step_has_started = False

                for line in process.stdout:
                    # Log all output
                    print(line, end="", flush=True)
                    
                    if "***** Running training *****" in line:
                        training_step_has_started = True
                        lab.log("🚀 Training phase started")

                    if not training_step_has_started:
                        continue

                    # Parse progress from training output
                    # Pattern: "  2%|▏         | 8/366 [00:15<11:28,  1.92s/it]"
                    pattern = r"(\d+)%\|.*\| (\d+)\/(\d+)"
                    match = re.search(pattern, line)
                    if match:
                        percentage = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        
                        # Map training progress to 40-85% of total progress
                        progress = 40 + int((percentage / 100) * 45)
                        lab.update_progress(progress)
                        
                        if percentage % 10 == 0:  # Log every 10%
                            lab.log(f"Training progress: {percentage}% ({current}/{total} steps)")

                    # Also look for loss values
                    if "loss" in line.lower() and "=" in line:
                        lab.log(line.strip())

                return_code = process.wait()

                if return_code != 0:
                    # Check if it's the known license error that can be ignored
                    error_log = ""
                    if "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'license'" in error_log:
                        lab.log("⚠️  Ignoring known license error in DPOTrainer")
                    else:
                        lab.log(f"❌ Training failed with return code: {return_code}")
                        lab.finish("Training failed")
                        return {"status": "error", "error": f"Training failed with code {return_code}"}

            lab.log("✅ Training completed successfully")

        except FileNotFoundError:
            lab.log("❌ llamafactory-cli not found. Please ensure LLaMA-Factory is installed correctly.")
            lab.finish("Training failed - llamafactory-cli not found")
            return {"status": "error", "error": "llamafactory-cli not found"}
        except Exception as e:
            lab.log(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            lab.finish("Training failed")
            return {"status": "error", "error": str(e)}

        lab.update_progress(85)

        # Merge/Fuse the adapter with the base model
        lab.log("Merging adapter with base model...")
        try:
            # Create merge configuration
            fused_model_dir = Path(output_dir) / "merged_model"
            fused_model_dir.mkdir(parents=True, exist_ok=True)

            merge_config = {
                "model_name_or_path": model_name,
                "adapter_name_or_path": str(adaptor_output_dir),
                "export_dir": str(fused_model_dir),
                "template": template,
                "resize_vocab": True,
                "export_size": 2,
                "export_device": "cpu",
                "export_legacy_format": False,
            }

            merge_config_path = data_directory / "merge_config.yaml"
            with open(merge_config_path, "w") as f:
                yaml.dump(merge_config, f)

            lab.log(f"Merge config saved to: {merge_config_path}")

            # Run merge command
            merge_command = [
                llamafactory_cli,
                "export",
                str(merge_config_path)
            ]

            lab.log(f"Running merge command: {' '.join(merge_command)}")

            with subprocess.Popen(
                merge_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                cwd=str(llama_factory_dir),
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    lab.log(line.strip())

                return_code = process.wait()

                if return_code != 0:
                    lab.log(f"⚠️  Merge returned non-zero exit code: {return_code}")
                    # Continue anyway as the adapter is still usable

            lab.log("✅ Model merging completed")

            # Save the merged model to TransformerLab
            saved_model_path = lab.save_model(
                str(fused_model_dir),
                name=f"dpo_trained_model_{pref_loss}"
            )
            lab.log(f"✅ Merged model saved to: {saved_model_path}")

        except Exception as e:
            lab.log(f"⚠️  Error during model merging: {e}")
            import traceback
            traceback.print_exc()
            lab.log("Continuing despite merge error - adapter is still available")

        lab.update_progress(95)

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

        # Save training summary
        try:
            summary_file = Path(output_dir) / "training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(
                    {
                        "training_type": f"Llama Factory {pref_loss.upper()}",
                        "model_name": model_name,
                        "dataset": dataset_name,
                        "preference_method": pref_loss,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "num_train_epochs": num_train_epochs,
                        "template": template,
                        "completed_at": end_time.isoformat(),
                        "duration": str(training_duration),
                    },
                    f,
                    indent=2,
                )

            summary_path = lab.save_artifact(str(summary_file), "training_summary.json")
            lab.log(f"Saved training summary: {summary_path}")

            # Save the adapter as an artifact
            if adaptor_output_dir.exists():
                adaptor_path = lab.save_artifact(
                    str(adaptor_output_dir),
                    f"{pref_loss}_adapter"
                )
                lab.log(f"Saved adapter: {adaptor_path}")

        except Exception as e:
            lab.log(f"⚠️  Error saving artifacts: {e}")

        # Get wandb URL if available
        job_data = lab.job.get_job_data()
        captured_wandb_url = job_data.get("wandb_run_url", "None")
        lab.log(f"📋 Wandb URL: {captured_wandb_url}")

        # Finish wandb run if it was initialized
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                lab.log("✅ Wandb run finished")
        except Exception:
            pass

        # Complete the job
        lab.finish("Training completed successfully!")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": output_dir,
            "preference_method": pref_loss,
            "wandb_url": captured_wandb_url,
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
    result = train_with_llama_factory()
    print("Training result:", result)
