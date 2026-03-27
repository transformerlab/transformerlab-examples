import os
import yaml
import subprocess
from datetime import datetime

from lab import lab


def train():
    """Distillation training using DistillKit"""

    try:
        # Initialize lab
        lab.init()
        config = lab.get_config()

        # Training configuration
        training_config = {
            "project_name": config.get("project_name", "distilkit-distillation"),
            "model": config.get("model", "Qwen/Qwen3-8B"),
            "output_path": config.get("output_path", "./output"),
            "sequence_length": int(config.get("sequence_length", 8192)),
            "dataset": {
                "train_dataset": {
                    "repo_id": config.get(
                        "train_dataset_repo", "arcee-ai/Qwen3-235B-Logits-Packed-8192"
                    ),
                    "split": config.get("train_dataset_split", "train"),
                },
                "prepacked": True,
            },
            "teacher": {
                "kind": config.get("teacher_kind", "dataset"),
                "logprob_compressor": {
                    "d": int(config.get("vocab_size", 151936)),
                    "delta_encoding": False,
                    "error_diffusion": False,
                    "exact_dtype": "float32",
                    "exact_k": int(config.get("exact_k", 32)),
                    "k": int(config.get("k", 128)),
                    "polynomial_terms": [0, 1, 2],
                    "residual_bins": [],
                    "term_dtype": "float32",
                },
            },
            "loss_functions": [
                {"function": "cross_entropy", "weight": 0.5},
                {
                    "function": "kl",
                    "weight": 0.5,
                    "temperature": 1.0,
                    "missing_probability_handling": "zero",
                    "sparse_chunk_length": 1024,
                },
            ],
            "training_args": {
                "num_train_epochs": int(config.get("num_train_epochs", 1)),
                "per_device_train_batch_size": int(
                    config.get("per_device_train_batch_size", 1)
                ),
                "gradient_accumulation_steps": int(
                    config.get("gradient_accumulation_steps", 8)
                ),
                "learning_rate": float(config.get("learning_rate", 2e-6)),
                "bf16": bool(config.get("bf16", True)),
                "optim": config.get("optim", "adamw_torch"),
                "gradient_checkpointing": bool(
                    config.get("gradient_checkpointing", True)
                ),
            },
        }

        lab.set_config(training_config)

        # Log start
        start_time = datetime.now()
        lab.log(f"Distillation started at {start_time}")

        # Create output directory
        os.makedirs(training_config["output_path"], exist_ok=True)

        # Create config.yaml for distillkit
        config_file = os.path.join(
            training_config["output_path"], "distill_config.yaml"
        )
        with open(config_file, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False)

        lab.log(f"Created distillkit config at {config_file}")

        # Run distillkit
        lab.log("Starting distillation with distillkit...")
        lab.update_progress(10)

        # Run the command
        cmd = ["distillkit", config_file]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Monitor progress (simple way, read output and update progress)
        progress = 10
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                lab.log(output.strip())
                # Simple progress update, assume linear or based on epochs
                if "epoch" in output.lower():
                    progress += 10
                    if progress <= 90:
                        lab.update_progress(progress)

        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

        lab.update_progress(100)

        # Save final model or artifacts
        # Assuming distillkit saves to output_path
        saved_path = lab.save_model(
            training_config["output_path"], name="distilled_model"
        )
        lab.log(f"Model saved to {saved_path}")

        end_time = datetime.now()
        duration = end_time - start_time
        lab.log(f"Distillation completed in {duration}")

        lab.finish("Distillation completed successfully")

        return {
            "status": "success",
            "output_dir": training_config["output_path"],
            "saved_model_path": saved_path,
            "duration": str(duration),
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"Distillkit failed: {e}"
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}

    except Exception as e:
        error_msg = str(e)
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    result = train()
    print(result)
