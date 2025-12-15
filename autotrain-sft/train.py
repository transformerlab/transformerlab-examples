#!/usr/bin/env python3
"""
Example: SFT training with Hugging Face AutoTrain (llm-sft) integrated with the lab SDK.

This script demonstrates how you could launch an AutoTrain SFT run for
`meta-llama/Llama-3.2-1B-Instruct` on the `Trelis/touch-rugby-rules` dataset
while reporting progress, logs, and artifacts via the `lab` facade.

Notes:
- This example assumes the `autotrain` CLI from `autotrain-advanced` is installed
  and on PATH (e.g. `pip install \"autotrain-advanced[llm]\"`).
- It also assumes you have a valid Hugging Face token in `HF_TOKEN` and that
  AutoTrain can use it (typically via `huggingface-cli login` or env vars).
"""

import json
import os
import subprocess
from datetime import datetime
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
          valid_split: validation
          column_mapping:
            text_column: text

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

        hub:
          username: your_hf_username
          token: $HF_TOKEN
          push_to_hub: false
        """
    ).lstrip()

    with open(config_path, "w") as f:
        f.write(yaml_content)


def train_with_autotrain(quick_test: bool = True):
    """
    Launch an AutoTrain llm-sft job and integrate it with the lab facade.

    Args:
        quick_test: If True, use a very small training setup (1 epoch).
    """

    # Use a single GPU by default if available
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    training_config = {
        "experiment_name": "autotrain-llama32-sft",
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "dataset": "Trelis/touch-rugby-rules",
        "template_name": "autotrain-llm-sft-demo",
        "project_name": "llama32-touch-rugby-sft",
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

        # Write AutoTrain config YAML into the output directory
        autotrain_config_path = os.path.join(training_config["output_dir"], "autotrain_llm_sft.yaml")
        _write_autotrain_config(autotrain_config_path, training_config, quick_test)
        training_config["_config"]["autotrain_config_path"] = autotrain_config_path

        lab.log(f"📝 Wrote AutoTrain config to: {autotrain_config_path}")
        lab.update_progress(15)

        # Save the config as an artifact in TransformerLab
        config_artifact_path = lab.save_artifact(autotrain_config_path, "autotrain_llm_sft.yaml")
        lab.log(f"Saved AutoTrain config artifact: {config_artifact_path}")

        # Run the AutoTrain CLI
        cmd = ["autotrain", autotrain_config_path]
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

        # Stream logs from AutoTrain into lab.log
        current_progress = 20
        lab.update_progress(current_progress)

        if process.stdout is not None:
            for idx, line in enumerate(process.stdout):
                line = line.rstrip()
                if line:
                    lab.log(f"[autotrain] {line}")

                # Bump progress slowly up to 90% as logs stream by
                if idx % 50 == 0 and current_progress < 90:
                    current_progress += 5
                    lab.update_progress(current_progress)

        return_code = process.wait()
        if return_code != 0:
            msg = f"AutoTrain process exited with code {return_code}"
            lab.log(f"❌ {msg}")
            lab.finish("Training failed - AutoTrain error")
            return {"status": "error", "error": msg}

        lab.update_progress(95)

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
