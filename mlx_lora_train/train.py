#!/usr/bin/env python3
"""
Fine-Tuning with LoRA or QLoRA using Apple MLX.

This script trains a language model using MLX LoRA fine-tuning,
following the new task format with the lab SDK.

https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md
"""

import json
import os
import re
import subprocess
import sys
import time
import yaml

from lab import lab

# Login to huggingface if token is available
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def prepare_dataset_files(
    data_directory,
    datasets,
    formatting_template=None,
    chat_template=None,
    model_name=None,
    chat_column="messages",
):
    """
    Prepare train.jsonl / valid.jsonl for MLX from HuggingFace dataset splits.

    Applies either a Jinja formatting_template or a chat_template (using the
    model's tokenizer) to each example, then writes one JSON-Lines file per split.
    """
    from jinja2 import Environment
    from transformers import AutoTokenizer

    tokenizer = None
    if chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _format(example):
        if chat_template and tokenizer and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                example[chat_column],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
            )
        if formatting_template:
            jinja_env = Environment()
            tmpl = jinja_env.from_string(formatting_template)
            return tmpl.render(example)
        raise ValueError("Either formatting_template or chat_template must be provided.")

    os.makedirs(data_directory, exist_ok=True)

    for split_name, split_data in datasets.items():
        output_file = os.path.join(data_directory, f"{split_name}.jsonl")
        with open(output_file, "w") as f:
            for i in range(len(split_data)):
                example = split_data[i]
                try:
                    rendered = _format(example)
                    rendered = rendered.replace("\n", "\\n").replace("\r", "\\r")
                    f.write(json.dumps({"text": rendered}) + "\n")
                except Exception:
                    print(f"Warning: Failed to process example {i} in '{split_name}'. Skipping.")
                    continue

        # Print one example for verification
        try:
            with open(output_file, "r") as f:
                first_line = f.readline()
                if first_line:
                    parsed = json.loads(first_line)
                    print(f"Example from {split_name} split:")
                    print(parsed.get("text", first_line))
        except Exception as e:
            print(f"Error reading example from {output_file}: {e}")


def load_datasets(dataset_name, splits=None, config_name=None):
    """
    Load datasets from HuggingFace or local path, handling split negotiation.

    Returns a dict mapping split name → HuggingFace Dataset object.
    If no validation split exists, automatically creates an 80/20 split.
    """
    from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names

    if splits is None:
        splits = ["train", "valid"]

    available_splits = get_dataset_split_names(dataset_name)
    available_configs = get_dataset_config_names(dataset_name)

    if available_configs and available_configs[0] == "default":
        available_configs.pop(0)
        config_name = None

    if not config_name and len(available_configs) > 0:
        config_name = available_configs[0]
        print(f"Using default config name: {config_name}")

    # Build a mapping of desired → actual split name
    dataset_splits = {}
    for s in splits:
        dataset_splits[s] = s

    if "train" in splits and "train" not in available_splits:
        dataset_splits["train"] = available_splits[0]
        print(f"Using `{dataset_splits['train']}` for the training split.")

    if "validation" in available_splits and "valid" in dataset_splits:
        dataset_splits["valid"] = "validation"
    elif "valid" in splits and "valid" not in available_splits:
        print("No validation split found, splitting train 80/20.")
        dataset_splits["valid"] = dataset_splits["train"] + "[-20%:]"
        dataset_splits["train"] = dataset_splits["train"] + "[:80%]"

    # Avoid identical train/valid
    for expected, actual in list(dataset_splits.items()):
        if expected != "train" and actual == dataset_splits["train"]:
            dataset_splits[expected] = dataset_splits["train"] + "[-20%:]"
            dataset_splits["train"] = dataset_splits["train"] + "[:80%]"

    result = {}
    for desired, actual in dataset_splits.items():
        result[desired] = load_dataset(dataset_name, config_name, split=actual)
        print(f"Loaded {desired} split ({actual}): {len(result[desired])} examples")

    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_mlx_lora():
    """Train a model using MLX LoRA fine-tuning."""
    print("MLX LoRA Trainer starting…")

    try:
        # Initialize lab — picks up _TFL_JOB_ID, _TFL_EXPERIMENT_ID from env
        lab.init()

        config = lab.get_config()

        # ----- Extract configuration parameters -----
        model_name = config.get("model_name", "")
        dataset_name = config.get("dataset_name", config.get("dataset", ""))
        lora_layers = config.get("lora_layers", 16)
        learning_rate = config.get("learning_rate", 5e-5)
        batch_size = int(config.get("batch_size", 4))
        steps_per_eval = int(config.get("steps_per_eval", 200))
        iters = int(config.get("iters", 1000))
        adaptor_name = config.get("adaptor_name", "adaptor")
        fuse_model = config.get("fuse_model", True)
        num_train_epochs = config.get("num_train_epochs", None)
        steps_per_report = int(config.get("steps_per_report", 10))
        save_every = int(config.get("save_every", 1000))
        lora_rank = config.get("lora_rank", None)
        lora_alpha = config.get("lora_alpha", None)
        chat_template = config.get("formatting_chat_template", None)
        chat_column = config.get("chatml_formatted_column", "messages")
        formatting_template = config.get("formatting_template", None)

        # ----- Load dataset -----
        lab.log("Loading dataset…")
        datasets = load_datasets(dataset_name, ["train", "valid"])
        lab.log(f"Dataset loaded: {len(datasets.get('train', []))} train, {len(datasets.get('valid', []))} valid")

        # ----- Epoch-based training -----
        if num_train_epochs is not None and str(num_train_epochs) != "" and int(num_train_epochs) >= 0:
            num_train_epochs = int(num_train_epochs)
            if num_train_epochs == 0:
                lab.log("Training set to 0 epochs – overriding to 1. Set -1 to disable epoch-based training.")
                num_train_epochs = 1
            num_examples = len(datasets["train"])
            steps_per_epoch = max(num_examples // batch_size, 1)
            iters = steps_per_epoch * num_train_epochs
            lab.log(
                f"Epoch-based training: {num_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch, {iters} total iterations"
            )

        # ----- Working directories -----
        output_dir = os.path.abspath(config.get("output_dir", "./output"))
        os.makedirs(output_dir, exist_ok=True)

        data_directory = os.path.join(output_dir, "data")
        adaptor_output_dir = os.path.join(output_dir, "adaptors", adaptor_name)
        os.makedirs(adaptor_output_dir, exist_ok=True)

        # ----- LoRA config YAML (rank / alpha) -----
        config_file = None
        if lora_rank or lora_alpha:
            lora_scale = int(lora_alpha) / int(lora_rank) if lora_alpha and lora_rank else 1
            lora_config = {
                "lora_parameters": {
                    "alpha": lora_alpha,
                    "rank": lora_rank,
                    "scale": lora_scale,
                    "dropout": 0,
                }
            }
            config_file = os.path.join(output_dir, "lora_config.yaml")
            with open(config_file, "w") as f:
                yaml.dump(lora_config, f)
            lab.log(f"LoRA config: {lora_config}")

        # ----- Prepare formatted dataset files -----
        prepare_dataset_files(
            data_directory=data_directory,
            datasets=datasets,
            formatting_template=formatting_template,
            chat_template=chat_template,
            model_name=model_name,
            chat_column=chat_column,
        )

        # ----- Check for checkpoint resume -----
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"Resuming training from checkpoint: {checkpoint}")
            adaptor_output_dir = checkpoint

        # ----- Build the MLX LoRA training command -----
        python_executable = sys.executable

        popen_command = [
            python_executable,
            "-um",
            "mlx_lm",
            "lora",
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
        if config_file:
            popen_command.extend(["--config", config_file])

        lab.log(f"Running command: {' '.join(popen_command)}")
        lab.log(f"Adaptor will be saved in: {adaptor_output_dir}")
        lab.update_progress(10)
        lab.log("Starting training…")

        # ----- Progress mapping: training runs from 10% → 90% -----
        TRAIN_PROGRESS_START = 10
        TRAIN_PROGRESS_END = 90

        start_time = time.time()
        last_progress_log_iter = 0

        # ----- Run the MLX LoRA training process -----
        with subprocess.Popen(
            popen_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy(),
        ) as process:
            for line in process.stdout:
                # Parse progress — MLX outputs "Iter N:" for each report
                iter_match = re.search(r"Iter (\d+):", line)
                if iter_match:
                    iteration = int(iter_match.group(1))
                    train_fraction = float(iteration) / float(iters) if iters > 0 else 1.0
                    overall_progress = TRAIN_PROGRESS_START + int(
                        train_fraction * (TRAIN_PROGRESS_END - TRAIN_PROGRESS_START)
                    )
                    lab.update_progress(min(overall_progress, TRAIN_PROGRESS_END))

                    # Log progress with ETA at regular intervals
                    log_interval = max(iters // 10, steps_per_report)
                    if iteration - last_progress_log_iter >= log_interval or iteration == iters:
                        last_progress_log_iter = iteration
                        elapsed = time.time() - start_time
                        if iteration > 0:
                            remaining_iters = int(iters) - iteration
                            eta_seconds = (
                                int((elapsed / iteration) * remaining_iters) if remaining_iters > 0 else 0
                            )
                            eta_str = (
                                time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 0 else "done"
                            )
                            lab.log(f"Iter {iteration}/{iters} ({train_fraction * 100:.1f}%) – ETA {eta_str}")

                    # Parse training metrics
                    train_match = re.search(
                        r"Train loss (\d+\.\d+),\s*Learning Rate (\S+),\s*"
                        r"It/sec (\d+\.\d+),\s*Tokens/sec (\d+\.\d+)",
                        line,
                    )
                    if train_match:
                        loss = float(train_match.group(1))
                        it_per_sec = float(train_match.group(3))
                        tokens_per_sec = float(train_match.group(4))
                        lab.log(f"  train/loss={loss:.4f}  it/sec={it_per_sec:.2f}  tok/sec={tokens_per_sec:.2f}")
                    else:
                        # Parse validation metrics
                        val_match = re.search(r"Val loss (\d+\.\d+),\s*Val took (\d+\.\d+)s", line)
                        if val_match:
                            validation_loss = float(val_match.group(1))
                            lab.log(f"  eval/loss={validation_loss:.4f}")

                print(line, end="", flush=True)

        # Check return code
        if process.returncode and process.returncode != 0:
            raise RuntimeError("Training failed — mlx_lm lora returned non-zero exit code.")

        lab.update_progress(TRAIN_PROGRESS_END)
        lab.log("Training completed.")

        # ----- Save checkpoint of the trained adaptor -----
        try:
            lab.save_checkpoint(adaptor_output_dir, f"adaptor_{adaptor_name}")
            lab.log(f"✅ Saved adaptor checkpoint: adaptor_{adaptor_name}")
        except Exception as e:
            lab.log(f"⚠️  Could not save adaptor checkpoint: {e}")

        # ----- Fuse or save adaptor -----
        if not fuse_model:
            lab.update_progress(92)
            lab.log(f"Adaptor training complete – saved at {adaptor_output_dir}")
            lab.update_progress(95)
        else:
            lab.update_progress(91)
            lab.log("Fusing adaptor with base model…")

            short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            fused_model_name = f"{short_name}_{adaptor_name}"
            fused_model_location = os.path.join(output_dir, "fused_model", fused_model_name)
            os.makedirs(fused_model_location, exist_ok=True)

            fuse_command = [
                python_executable,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                model_name,
                "--adapter-path",
                adaptor_output_dir,
                "--save-path",
                fused_model_location,
            ]

            lab.update_progress(92)
            with subprocess.Popen(
                fuse_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy(),
            ) as fuse_proc:
                for line in fuse_proc.stdout:
                    print(line, end="", flush=True)
                return_code = fuse_proc.wait()

            if return_code == 0:
                lab.update_progress(95)
                lab.log("Saving fused model…")
                lab.save_model(
                    fused_model_location,
                    name=fused_model_name,
                    architecture="MLX",
                    parent_model=model_name,
                )
                lab.log("✅ Model fusion complete.")
                lab.update_progress(98)
            else:
                raise RuntimeError(f"Model fusion failed with return code {return_code}")

        # ----- Record training duration -----
        elapsed_total = time.time() - start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
        lab.log(f"Total training duration: {duration_str}")

        lab.update_progress(100)
        lab.finish("Training completed successfully with MLX LoRA")

    except KeyboardInterrupt:
        lab.error("Stopped by user")
    except Exception as e:
        import traceback

        traceback.print_exc()
        lab.error(str(e))
        raise


if __name__ == "__main__":
    train_mlx_lora()
