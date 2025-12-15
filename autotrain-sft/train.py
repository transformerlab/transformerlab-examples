#!/usr/bin/env python3
"""
Example: SFT training with Hugging Face AutoTrain (llm-sft) integrated with the lab SDK.

This script demonstrates how you could launch an AutoTrain SFT run for
`meta-llama/Llama-3.2-1B-Instruct` on the `Trelis/touch-rugby-rules` dataset
while reporting progress, logs, and artifacts via the `lab` facade.

Notes:
- This example assumes the `autotrain` CLI from `autotrain-advanced` is installed
  and on PATH (e.g. `pip install autotrain-advanced`).
- It also assumes you have a valid Hugging Face token in `HF_TOKEN` and that
  AutoTrain can use it (typically via `huggingface-cli login` or env vars).
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from lab import lab


def _write_autotrain_config(config_path: str, training_config: dict, quick_test: bool) -> None:
    """Write an AutoTrain llm-sft YAML config file based on `training_config`."""
    base_model = training_config["model_name"]
    dataset_name = training_config["dataset"]
    project_name = training_config.get("project_name", "llama32-touch-rugby-sft")

    # Keep the config fairly small for quick_test, larger for full training
    epochs = 1 if quick_test else 3
    batch_size = 2
    lr = 3e-5
    # Checkpointing: save every 100 steps for quick test, 500 for full training
    save_steps = 100 if quick_test else 500
    # Keep last 3 checkpoints
    save_total_limit = 3

    yaml_content = dedent(
        f"""
        task: llm-sft
        base_model: {base_model}
        project_name: {project_name}
        log: tensorboard
        backend: local

        data:
          path: {dataset_name}
          train_split: train
          valid_split: null
          chat_template: tokenizer
          column_mapping:
            text_column: messages

        params:
          block_size: 1024
          model_max_length: 2048
          max_prompt_length: 512
          epochs: {epochs}
          batch_size: {batch_size}
          lr: {lr}
          padding: right
          optimizer: adamw_torch
          scheduler: linear
          gradient_accumulation: 4
          mixed_precision: fp16
          save_steps: {save_steps}
          save_total_limit: {save_total_limit}
          logging_steps: 10

        hub:
          username: your_hf_username
          token: $HF_TOKEN
          push_to_hub: false
        """
    ).lstrip()

    with open(config_path, "w") as f:
        f.write(yaml_content)


def _monitor_checkpoints(output_dir: str, project_name: str, stop_event: threading.Event) -> None:
    """
    Monitor the AutoTrain output directory for checkpoint creation and save them via lab facade.

    AutoTrain typically saves checkpoints to: <output_dir>/<project_name>/checkpoints/checkpoint-<step>
    This function runs in a background thread and watches for new checkpoint directories.
    """
    project_path = Path(output_dir) / project_name
    checkpoints_dir = project_path / "checkpoints"
    seen_checkpoints = set()

    while not stop_event.is_set():
        try:
            if checkpoints_dir.exists():
                # Look for checkpoint-* directories
                checkpoint_dirs = [
                    d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
                ]

                for checkpoint_dir in checkpoint_dirs:
                    checkpoint_name = checkpoint_dir.name
                    if checkpoint_name not in seen_checkpoints:
                        seen_checkpoints.add(checkpoint_name)
                        try:
                            # Save checkpoint via lab facade
                            saved_path = lab.save_checkpoint(str(checkpoint_dir), checkpoint_name)
                            lab.log(f"💾 Saved checkpoint: {checkpoint_name} -> {saved_path}")
                        except Exception as e:  # noqa: BLE001
                            lab.log(f"⚠️ Could not save checkpoint {checkpoint_name}: {e}")
        except Exception:  # noqa: BLE001
            # Silently continue monitoring even if there's an error
            pass

        time.sleep(5)  # Check every 5 seconds


def _parse_progress_from_log(line: str) -> dict:
    """
    Parse AutoTrain stdout for progress indicators (epochs, steps, loss, etc.).
    Returns a dict with parsed info, or None if nothing useful found.

    Handles tqdm-style progress bars like: "1%|          | 2/363 [00:04<12:25,  2.07s/it]"
    """
    progress_info = {}

    # First, try to parse tqdm-style progress bar format:
    # Format: "X%|          | current/total [elapsed<remaining, speed]"
    # Example: "1%|          | 2/363 [00:04<12:25,  2.07s/it]"
    # More flexible pattern that handles variations in spacing and bar characters
    tqdm_match = re.search(r"(\d+)%.*?(\d+)/(\d+)\s+\[", line)
    if tqdm_match:
        percentage = int(tqdm_match.group(1))
        current = int(tqdm_match.group(2))
        total = int(tqdm_match.group(3))

        progress_info["percentage"] = percentage
        progress_info["step"] = current
        progress_info["total_steps"] = total

        # Also try to parse speed from the time info: "2.07s/it"
        speed_match = re.search(r"([\d.]+)s/it", line)
        if speed_match:
            try:
                progress_info["speed"] = float(speed_match.group(1))
            except ValueError:
                pass

        return progress_info

    # Fallback: Look for epoch indicators: "Epoch X/Y" or "epoch X"
    epoch_match = re.search(r"epoch[:\s]+(\d+)(?:/(\d+))?", line, re.IGNORECASE)
    if epoch_match:
        progress_info["epoch"] = int(epoch_match.group(1))
        if epoch_match.group(2):
            progress_info["total_epochs"] = int(epoch_match.group(2))

    # Fallback: Look for step indicators: "Step X" or "step X/Y"
    step_match = re.search(r"step[:\s]+(\d+)(?:/(\d+))?", line, re.IGNORECASE)
    if step_match:
        progress_info["step"] = int(step_match.group(1))
        if step_match.group(2):
            progress_info["total_steps"] = int(step_match.group(2))

    # Look for loss values: "loss: 0.123" or "loss = 0.123"
    loss_match = re.search(r"loss[:\s=]+([\d.]+)", line, re.IGNORECASE)
    if loss_match:
        try:
            progress_info["loss"] = float(loss_match.group(1))
        except ValueError:
            pass

    return progress_info if progress_info else None


def train_with_autotrain(quick_test: bool = True):
    """
    Launch an AutoTrain llm-sft job and integrate it with the lab facade.

    Args:
        quick_test: If True, use a very small training setup (1 epoch).
    """

    # Use a single GPU by default if available
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    training_config = {
        "experiment_name": "autotrain-dialogpt-sft",
        "model_name": "microsoft/DialoGPT-small",
        "dataset": "HuggingFaceH4/no_robots",
        "template_name": "autotrain-llm-sft-demo",
        "project_name": "dialogpt-no-robots-sft",
        "output_dir": "./autotrain_output",
        "quick_test": quick_test,
        "_config": {
            "quick_test": quick_test,
        },
    }

    try:
        # Initialise lab and attach configuration
        lab.init()
        lab.set_config(training_config)

        # Basic logging
        start_time = datetime.now()
        mode = "Quick test" if quick_test else "Full training"
        lab.log(f"🚀 {mode} AutoTrain SFT started at {start_time}")
        lab.log(f"Base model: {training_config['model_name']}")
        lab.log(f"Dataset: {training_config['dataset']}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        os.makedirs(training_config["output_dir"], exist_ok=True)
        lab.update_progress(5)

        # Optionally hint about HF auth
        if not os.getenv("HF_TOKEN"):
            lab.log("⚠️ HF_TOKEN not set; ensure AutoTrain is authenticated with Hugging Face.")
        else:
            # Authenticate with Hugging Face
            lab.log("🔐 Authenticating with Hugging Face...")
            auth_cmd = ["huggingface-cli", "login", "--token", os.getenv("HF_TOKEN")]
            try:
                subprocess.run(auth_cmd, check=True, capture_output=True, text=True)
                lab.log("✅ Hugging Face authentication successful.")
            except subprocess.CalledProcessError as e:
                lab.log(f"❌ Hugging Face authentication failed: {e}")
                lab.finish("Authentication failed")
                return {"status": "error", "error": "HF auth failed"}

        # Write AutoTrain config YAML into the output directory
        autotrain_config_path = os.path.join(training_config["output_dir"], "autotrain_llm_sft.yaml")
        _write_autotrain_config(autotrain_config_path, training_config, quick_test)
        training_config["_config"]["autotrain_config_path"] = autotrain_config_path

        lab.log(f"📝 Wrote AutoTrain config to: {autotrain_config_path}")
        lab.update_progress(15)

        # Save the config as an artifact in TransformerLab
        config_artifact_path = lab.save_artifact(autotrain_config_path, "autotrain_llm_sft.yaml")
        lab.log(f"Saved AutoTrain config artifact: {config_artifact_path}")

        # Run the AutoTrain CLI. For autotrain-advanced the pattern is:
        #   autotrain --config <config_path>
        cmd = ["autotrain", "--config", autotrain_config_path]
        lab.log(f"🚀 Launching AutoTrain via CLI: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            lab.log("❌ `autotrain` CLI not found. Install with `pip install autotrain-advanced`.")
            lab.finish("Training skipped - autotrain CLI not available")
            return {"status": "skipped", "reason": "autotrain CLI not found"}

        # Start checkpoint monitoring in background thread
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_checkpoints,
            args=(training_config["output_dir"], training_config["project_name"], stop_monitoring),
            daemon=True,
        )
        monitor_thread.start()
        lab.log("🔍 Started checkpoint monitoring thread")

        # Stream logs from AutoTrain into lab.log and parse for progress
        current_progress = 20
        lab.update_progress(current_progress)

        # Track parsed progress info
        last_epoch = None
        last_step = None
        total_epochs = None
        total_steps = None
        last_logged_percentage = -1  # Track last logged percentage to avoid spam

        if process.stdout is not None:
            for idx, line in enumerate(process.stdout):
                line = line.rstrip()
                if line:
                    lab.log(f"[autotrain] {line}")

                    # Parse progress from log line
                    progress_info = _parse_progress_from_log(line)
                    if progress_info:
                        # Prioritize percentage from tqdm progress bar (most accurate)
                        if "percentage" in progress_info:
                            # Map tqdm percentage (0-100%) to our progress range (20% to 85%)
                            tqdm_pct = progress_info["percentage"]
                            mapped_progress = 20 + int((tqdm_pct / 100) * 65)
                            current_progress = max(current_progress, min(mapped_progress, 85))
                            lab.update_progress(current_progress)

                            # Log progress updates every 5% to avoid spam
                            if tqdm_pct >= last_logged_percentage + 5 or tqdm_pct == 100:
                                if "step" in progress_info and "total_steps" in progress_info:
                                    speed_info = (
                                        f" ({progress_info['speed']:.2f}s/it)" if "speed" in progress_info else ""
                                    )
                                    lab.log(
                                        f"📊 Progress: {progress_info['step']}/{progress_info['total_steps']} ({tqdm_pct}%){speed_info}"
                                    )
                                last_logged_percentage = tqdm_pct

                        if "epoch" in progress_info:
                            last_epoch = progress_info["epoch"]
                            if "total_epochs" in progress_info:
                                total_epochs = progress_info["total_epochs"]

                        if "step" in progress_info:
                            last_step = progress_info["step"]
                            if "total_steps" in progress_info:
                                total_steps = progress_info["total_steps"]

                        if "loss" in progress_info:
                            lab.log(f"📊 Loss: {progress_info['loss']:.4f}")

                        # Fallback: Update progress based on parsed info (if no percentage)
                        if "percentage" not in progress_info:
                            if total_epochs and last_epoch is not None:
                                # Progress based on epochs (20% to 85%)
                                epoch_progress = 20 + int((last_epoch / total_epochs) * 65)
                                current_progress = max(current_progress, min(epoch_progress, 85))
                                lab.update_progress(current_progress)
                            elif total_steps and last_step is not None:
                                # Progress based on steps (20% to 85%)
                                step_progress = 20 + int((last_step / total_steps) * 65)
                                current_progress = max(current_progress, min(step_progress, 85))
                                lab.update_progress(current_progress)

                # Fallback: bump progress slowly if no structured progress found
                if idx % 100 == 0 and current_progress < 85:
                    current_progress = min(current_progress + 2, 85)
                    lab.update_progress(current_progress)

        # Stop checkpoint monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=2)

        return_code = process.wait()
        if return_code != 0:
            msg = f"AutoTrain process exited with code {return_code}"
            lab.log(f"❌ {msg}")
            lab.finish("Training failed - AutoTrain error")
            return {"status": "error", "error": msg}

        lab.update_progress(95)

        # Final checkpoint collection: check for any checkpoints that might have been created
        # after monitoring stopped
        project_path = Path(training_config["output_dir"]) / training_config["project_name"]
        checkpoints_dir = project_path / "checkpoints"
        final_checkpoints = []

        if checkpoints_dir.exists():
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            if checkpoint_dirs:
                # Sort by checkpoint number (extract from checkpoint-<num>)
                checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)
                final_checkpoints = [d.name for d in checkpoint_dirs]
                lab.log(f"📦 Found {len(final_checkpoints)} checkpoint(s): {', '.join(final_checkpoints)}")

                # Save the latest checkpoint if not already saved
                if checkpoint_dirs:
                    latest_checkpoint = checkpoint_dirs[-1]
                    try:
                        saved_path = lab.save_checkpoint(str(latest_checkpoint), latest_checkpoint.name)
                        lab.log(f"💾 Saved final checkpoint: {latest_checkpoint.name} -> {saved_path}")
                    except Exception as e:  # noqa: BLE001
                        lab.log(f"⚠️ Could not save final checkpoint: {e}")

        # Post-training bookkeeping
        end_time = datetime.now()
        duration = end_time - start_time
        lab.log(f"✅ AutoTrain run completed in {duration}")

        # Write a small summary file and save as artifact
        summary_path = os.path.join(training_config["output_dir"], "autotrain_summary.json")
        summary = {
            "job_id": lab.job.id,
            "status": "success",
            "mode": "quick_test" if quick_test else "full_training",
            "model_name": training_config["model_name"],
            "dataset": training_config["dataset"],
            "duration": str(duration),
            "autotrain_config": os.path.basename(autotrain_config_path),
            "checkpoints_found": final_checkpoints,
            "checkpoint_count": len(final_checkpoints),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        summary_artifact = lab.save_artifact(summary_path, "autotrain_summary.json")
        lab.log(f"Saved AutoTrain summary artifact: {summary_artifact}")

        # Best-effort: try to save the output directory as a model
        try:
            saved_model_path = lab.save_model(
                training_config["output_dir"],
                name="autotrain_llama32_sft_model",
                architecture="llm",
                pipeline_tag="text-generation",
                parent_model=training_config["model_name"],
            )
            lab.log(f"✅ Saved model directory via lab.save_model: {saved_model_path}")
        except Exception as e:  # noqa: BLE001
            lab.log(f"⚠️ Could not save model directory: {e}")
            saved_model_path = None

        lab.update_progress(100)
        lab.finish("AutoTrain SFT run completed successfully")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(duration),
            "output_dir": training_config["output_dir"],
            "saved_model_path": saved_model_path,
            "mode": "quick_test" if quick_test else "full_training",
        }

    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:  # noqa: BLE001
        error_msg = str(e)
        print(f"AutoTrain SFT failed: {error_msg}")

        import traceback

        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    # Default to a quick-test style run (1 epoch, smaller config) to keep
    # example runs lightweight. To run a longer training, change quick_test=False
    # in the call below.
    print("🚀 Running AutoTrain quick test mode (no CLI flags)...")
    result = train_with_autotrain(quick_test=True)
    print("AutoTrain result:", result)
