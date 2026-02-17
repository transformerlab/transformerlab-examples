#!/usr/bin/env python3
"""
Nanochat Training Script with TransformerLab SDK Integration
Runs the nanochat speedrun training pipeline with proper SDK integration
Based on: https://github.com/karpathy/nanochat/blob/master/speedrun.sh
"""

import os
import subprocess
import sys
from datetime import datetime

from lab import lab


def run_command(command, description, stream_output=True, cwd=None, env=None):
    """Execute a command and log the output."""
    lab.log(f"🔧 {description}")
    try:
        # Merge environment variables
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)

        if stream_output:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=cmd_env,
            )
            for line in iter(process.stdout.readline, ""):
                if line:
                    print(line.rstrip())
                    lab.log(line.rstrip())
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        else:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=cmd_env,
            )
            if result.stdout:
                lab.log(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        lab.log(f"❌ Error running command: {command}")
        if hasattr(e, "stderr") and e.stderr:
            lab.log(f"Error output: {e.stderr}")
        raise


def get_available_gpus():
    """Detect the number of available GPUs using nvidia-smi or torch."""
    try:
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except (ImportError, Exception):
        pass
    return 0


def _bool(val):
    """Normalise a config value that may be a string to a Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def _int(val, default):
    """Safely cast a config value to int, falling back to *default*."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _float(val, default):
    """Safely cast a config value to float, falling back to *default*."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def setup_environment(base_dir, nanochat_dir, nproc):
    """Setup environment variables and install dependencies."""
    lab.log("🔧 Phase: Environment Setup")

    os.environ["NANOCHAT_BASE_DIR"] = base_dir
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NPROC_PER_NODE"] = str(nproc)

    lab.log(f"NANOCHAT_BASE_DIR: {base_dir}")
    lab.log(f"Using {nproc} GPU(s)")

    # Install uv if not available
    if subprocess.run("command -v uv", shell=True, capture_output=True).returncode != 0:
        lab.log("📦 Installing uv package manager...")
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Installing uv", stream_output=False)

    # Create venv and install dependencies
    lab.log("🔧 Creating virtual environment...")
    if not os.path.exists(os.path.join(nanochat_dir, ".venv")):
        run_command("uv venv", "Creating venv", cwd=nanochat_dir)

    lab.log("📦 Installing dependencies...")
    run_command("uv sync --extra gpu", "Installing dependencies", cwd=nanochat_dir)

    # Install transformerlab-sdk in the venv
    lab.log("📦 Installing transformerlab-sdk...")
    run_command("uv pip install transformerlab", "Installing SDK", cwd=nanochat_dir)


def train_tokenizer(nanochat_dir, initial_shards, total_shards, max_chars, vocab_size):
    """Train the BPE tokenizer and kick off background dataset download."""
    lab.log("🔧 Phase: Tokenizer Training")

    # Download initial dataset shards
    lab.log(f"📥 Downloading initial dataset shards ({initial_shards})...")
    run_command(
        f"uv run python -m nanochat.dataset -n {initial_shards}",
        "Downloading dataset shards",
        cwd=nanochat_dir,
    )

    # Start downloading remaining shards in background
    lab.log(f"📥 Starting background download of additional shards ({total_shards} total)...")
    dataset_process = subprocess.Popen(
        f"uv run python -m nanochat.dataset -n {total_shards}",
        shell=True,
        cwd=nanochat_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Train tokenizer
    lab.log(f"🎯 Training tokenizer (max_chars={max_chars:,}, vocab_size={vocab_size})...")
    tok_cmd = f"uv run python -m scripts.tok_train --max-chars={max_chars} --vocab-size={vocab_size}"
    run_command(tok_cmd, "Training tokenizer", cwd=nanochat_dir)

    # Evaluate tokenizer
    lab.log("📊 Evaluating tokenizer...")
    run_command("uv run python -m scripts.tok_eval", "Evaluating tokenizer", cwd=nanochat_dir)

    # Save tokenizer checkpoint
    base_dir = os.environ["NANOCHAT_BASE_DIR"]
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    if os.path.exists(tokenizer_dir):
        lab.save_checkpoint(tokenizer_dir, "tokenizer")
        lab.log("✅ Tokenizer checkpoint saved")

    return dataset_process


def train_base_model(nanochat_dir, dataset_process, nproc, cfg):
    """Pretrain the base model."""
    lab.log("🔧 Phase: Base Model Pretraining")

    # Wait for dataset download
    lab.log("⏳ Waiting for dataset download to complete...")
    dataset_process.wait()
    lab.log("✅ Dataset download complete")

    depth = cfg["depth"]
    wandb_run = cfg["wandb_run"]

    # Build base_train command with all configurable flags
    cmd_parts = [
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.base_train --",
        f"--depth={depth}",
        f"--device-batch-size={cfg['device_batch_size']}",
        f"--run={wandb_run}",
    ]

    # Optional flags
    if cfg["target_param_data_ratio"] > 0:
        cmd_parts.append(f"--target-param-data-ratio={cfg['target_param_data_ratio']}")
    if cfg["total_batch_size"] > 0:
        cmd_parts.append(f"--total-batch-size={cfg['total_batch_size']}")
    if cfg["num_iterations"] > 0:
        cmd_parts.append(f"--num-iterations={cfg['num_iterations']}")
    if cfg["max_seq_len"] != 2048:
        cmd_parts.append(f"--max-seq-len={cfg['max_seq_len']}")
    if cfg["window_pattern"] != "SSSL":
        cmd_parts.append(f"--window-pattern={cfg['window_pattern']}")
    if cfg["aspect_ratio"] != 64:
        cmd_parts.append(f"--aspect-ratio={cfg['aspect_ratio']}")
    if cfg["warmdown_ratio"] != 0.5:
        cmd_parts.append(f"--warmdown-ratio={cfg['warmdown_ratio']}")
    if cfg["weight_decay"] != 0.2:
        cmd_parts.append(f"--weight-decay={cfg['weight_decay']}")
    if cfg["eval_every"] != 250:
        cmd_parts.append(f"--eval-every={cfg['eval_every']}")
    if cfg["core_metric_every"] != 2000:
        cmd_parts.append(f"--core-metric-every={cfg['core_metric_every']}")
    if cfg["save_every"] != -1:
        cmd_parts.append(f"--save-every={cfg['save_every']}")
    if cfg["sample_every"] != 2000:
        cmd_parts.append(f"--sample-every={cfg['sample_every']}")
    if cfg["model_tag"]:
        cmd_parts.append(f"--model-tag={cfg['model_tag']}")
    if cfg["enable_fp8"]:
        cmd_parts.append("--fp8")

    base_train_cmd = " ".join(cmd_parts)
    lab.log(f"🎯 Training base d{depth} model...")
    lab.log(f"Command: {base_train_cmd}")
    run_command(base_train_cmd, "Training base model", cwd=nanochat_dir)

    # Evaluate base model: CORE metric, BPB on train/val, samples
    lab.log("📊 Evaluating base model (CORE, BPB, samples)...")
    eval_cmd = f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.base_eval -- --device-batch-size={cfg['device_batch_size']}"
    run_command(eval_cmd, "Evaluating base model", cwd=nanochat_dir)


def train_sft(nanochat_dir, nproc, cfg):
    """Run supervised finetuning (SFT)."""
    lab.log("🔧 Phase: Supervised Finetuning (SFT)")

    wandb_run = cfg["wandb_run"]

    # Download identity conversations for personality
    base_dir = os.environ["NANOCHAT_BASE_DIR"]
    identity_file = os.path.join(base_dir, "identity_conversations.jsonl")
    lab.log("📥 Downloading identity conversations...")
    run_command(
        f"curl -L -o {identity_file} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
        "Downloading identity conversations",
        stream_output=False,
    )

    # Build SFT command
    cmd_parts = [
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_sft --",
        f"--device-batch-size={cfg['sft_device_batch_size']}",
        f"--run={wandb_run}",
    ]
    if cfg["sft_num_iterations"] > 0:
        cmd_parts.append(f"--num-iterations={cfg['sft_num_iterations']}")
    if cfg["sft_mmlu_epochs"] != 3:
        cmd_parts.append(f"--mmlu-epochs={cfg['sft_mmlu_epochs']}")
    if cfg["sft_gsm8k_epochs"] != 4:
        cmd_parts.append(f"--gsm8k-epochs={cfg['sft_gsm8k_epochs']}")

    sft_cmd = " ".join(cmd_parts)
    lab.log(f"🎯 Running SFT...")
    lab.log(f"Command: {sft_cmd}")
    run_command(sft_cmd, "Supervised finetuning", cwd=nanochat_dir)

    # Evaluate SFT model
    lab.log("📊 Evaluating SFT model...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_eval -- -i sft",
        "Evaluating SFT model",
        cwd=nanochat_dir,
    )

    # Save SFT model to Model Zoo
    depth = cfg["depth"]
    model_tag = cfg["model_tag"] if cfg["model_tag"] else f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    if os.path.exists(checkpoint_dir):
        lab.log("💾 Saving SFT model to Model Zoo...")
        lab.save_model(
            source_path=checkpoint_dir,
            name=f"nanochat_{model_tag}_sft",
            architecture="gpt2",
            pipeline_tag="text-generation",
        )
        lab.log("✅ SFT model saved to Model Zoo")


def train_rl(nanochat_dir, nproc, cfg):
    """Run reinforcement learning on GSM8K (optional)."""
    if not cfg["enable_rl"]:
        lab.log("⏭️  Skipping RL training (disabled)")
        return

    lab.log("🔧 Phase: Reinforcement Learning (GSM8K)")
    wandb_run = cfg["wandb_run"]

    # Build RL command
    cmd_parts = [
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_rl --",
        f"--device-batch-size={cfg['rl_device_batch_size']}",
        f"--examples-per-step={cfg['rl_examples_per_step']}",
        f"--num-samples={cfg['rl_num_samples']}",
        f"--max-new-tokens={cfg['rl_max_new_tokens']}",
        f"--num-epochs={cfg['rl_num_epochs']}",
        f"--run={wandb_run}",
    ]

    rl_cmd = " ".join(cmd_parts)
    lab.log(f"🎯 Running RL on GSM8K...")
    lab.log(f"Command: {rl_cmd}")
    run_command(rl_cmd, "Reinforcement learning", cwd=nanochat_dir)

    # Evaluate RL model on GSM8K
    lab.log("📊 Evaluating RL model on GSM8K...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_eval -- -i rl -a GSM8K",
        "Evaluating RL model",
        cwd=nanochat_dir,
    )

    # Save RL model to Model Zoo
    base_dir = os.environ["NANOCHAT_BASE_DIR"]
    depth = cfg["depth"]
    model_tag = cfg["model_tag"] if cfg["model_tag"] else f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
    if os.path.exists(checkpoint_dir):
        lab.log("💾 Saving RL model to Model Zoo...")
        lab.save_model(
            source_path=checkpoint_dir,
            name=f"nanochat_{model_tag}_rl",
            architecture="gpt2",
            pipeline_tag="text-generation",
        )
        lab.log("✅ RL model saved to Model Zoo")


def generate_report(nanochat_dir):
    """Generate final training report."""
    lab.log("📝 Generating final report...")
    run_command("uv run python -m nanochat.report generate", "Generating report", cwd=nanochat_dir)

    report_path = os.path.join(nanochat_dir, "report.md")
    if os.path.exists(report_path):
        lab.save_artifact(report_path, "nanochat_training_report.md")
        lab.log("✅ Training report saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main training function — runs the nanochat speedrun with SDK integration."""

    start_time = datetime.now()

    # ------------------------------------------------------------------
    # 1. Initialise TransformerLab SDK and read configuration
    # ------------------------------------------------------------------
    try:
        lab.init()
        lab.log("🚀 Starting Nanochat Training with TransformerLab SDK")

        config = lab.get_config()
    except Exception as e:
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        import traceback
        traceback.print_exc()
        config = {}

    # Parse all parameters from config with sensible defaults matching latest speedrun.sh
    cfg = {
        # General
        "nproc_per_node":         _int(config.get("nproc_per_node"), 8),
        "enable_fp8":             _bool(config.get("enable_fp8", True)),
        "wandb_run_name":         str(config.get("wandb_run_name", "nanochat-speedrun")),

        # Dataset & Tokenizer
        "initial_dataset_shards": _int(config.get("initial_dataset_shards"), 8),
        "total_dataset_shards":   _int(config.get("total_dataset_shards"), 370),
        "tokenizer_max_chars":    _int(config.get("tokenizer_max_chars"), 2_000_000_000),
        "tokenizer_vocab_size":   _int(config.get("tokenizer_vocab_size"), 32768),

        # Base model pretraining
        "depth":                  _int(config.get("depth"), 26),
        "target_param_data_ratio": _float(config.get("target_param_data_ratio"), 8.25),
        "device_batch_size":      _int(config.get("device_batch_size"), 16),
        "total_batch_size":       _int(config.get("total_batch_size"), -1),
        "num_iterations":         _int(config.get("num_iterations"), -1),
        "max_seq_len":            _int(config.get("max_seq_len"), 2048),
        "window_pattern":         str(config.get("window_pattern", "SSSL")),
        "aspect_ratio":           _int(config.get("aspect_ratio"), 64),
        "warmdown_ratio":         _float(config.get("warmdown_ratio"), 0.5),
        "weight_decay":           _float(config.get("weight_decay"), 0.2),
        "eval_every":             _int(config.get("eval_every"), 250),
        "core_metric_every":      _int(config.get("core_metric_every"), 2000),
        "save_every":             _int(config.get("save_every"), -1),
        "sample_every":           _int(config.get("sample_every"), 2000),
        "model_tag":              str(config.get("model_tag", "")),

        # SFT
        "sft_device_batch_size":  _int(config.get("sft_device_batch_size"), 16),
        "sft_num_iterations":     _int(config.get("sft_num_iterations"), -1),
        "sft_mmlu_epochs":        _int(config.get("sft_mmlu_epochs"), 3),
        "sft_gsm8k_epochs":       _int(config.get("sft_gsm8k_epochs"), 4),

        # RL
        "enable_rl":              _bool(config.get("enable_rl", False)),
        "rl_num_epochs":          _int(config.get("rl_num_epochs"), 1),
        "rl_device_batch_size":   _int(config.get("rl_device_batch_size"), 8),
        "rl_examples_per_step":   _int(config.get("rl_examples_per_step"), 16),
        "rl_num_samples":         _int(config.get("rl_num_samples"), 16),
        "rl_max_new_tokens":      _int(config.get("rl_max_new_tokens"), 256),
    }

    lab.log(f"Training started at {start_time}")
    lab.log(f"Configuration: {cfg}")

    # ------------------------------------------------------------------
    # 2. Determine WANDB run name
    # ------------------------------------------------------------------
    if os.environ.get("WANDB_API_KEY"):
        cfg["wandb_run"] = cfg["wandb_run_name"]
        lab.log("🔑 WANDB_API_KEY found — wandb will log")
    else:
        cfg["wandb_run"] = "dummy"
        lab.log("ℹ️  No WANDB_API_KEY found, training will run without wandb logging")

    # ------------------------------------------------------------------
    # 3. Detect GPUs and resolve nproc
    # ------------------------------------------------------------------
    detected_gpus = get_available_gpus()
    nproc_requested = cfg["nproc_per_node"]
    if detected_gpus > 0:
        if nproc_requested > detected_gpus:
            lab.log(f"⚠️  Requested {nproc_requested} GPUs, but only {detected_gpus} detected. Capping to {detected_gpus}.")
            nproc = detected_gpus
        else:
            nproc = nproc_requested
    elif nproc_requested > 0:
        lab.log(f"⚠️  No GPUs detected via nvidia-smi or torch. Using {nproc_requested} as configured.")
        nproc = nproc_requested
    else:
        nproc = 1

    try:
        # ------------------------------------------------------------------
        # 4. Setup directories & clone repo
        # ------------------------------------------------------------------
        base_dir = os.path.expanduser("~/.cache/nanochat")
        os.makedirs(base_dir, exist_ok=True)
        lab.log(f"Training data directory: {base_dir}")

        nanochat_dir = os.path.abspath(os.path.expanduser("~/nanochat-repo"))

        if not os.path.exists(nanochat_dir) or not os.path.exists(os.path.join(nanochat_dir, "pyproject.toml")):
            if os.path.exists(nanochat_dir):
                lab.log(f"⚠️  {nanochat_dir} exists but is missing pyproject.toml. Re-cloning...")
                import shutil
                try:
                    shutil.rmtree(nanochat_dir)
                except Exception:
                    pass
            lab.log(f"📥 Cloning nanochat repository to {nanochat_dir}...")
            run_command(
                f"git clone https://github.com/karpathy/nanochat.git {nanochat_dir}",
                "Cloning nanochat repository",
            )
        else:
            lab.log(f"✅ Nanochat repository found: {nanochat_dir}")

        # Deactivate conda if active (to avoid conflicts with uv venv)
        if os.environ.get("CONDA_PREFIX"):
            lab.log("⚠️  Conda environment detected, clearing conda variables...")
            for var in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
                        "CONDA_SHLVL", "CONDA_PYTHON_EXE", "CONDA_EXE"]:
                os.environ.pop(var, None)
            lab.log("✅ Conda environment variables cleared")

        # ------------------------------------------------------------------
        # 5. Log training plan
        # ------------------------------------------------------------------
        lab.log("🚀 Starting Nanochat Speedrun Training Pipeline")
        lab.log(f"Pipeline: Tokenizer → Base d{cfg['depth']} Pretraining → SFT" + (" → RL" if cfg["enable_rl"] else ""))
        lab.log(f"Estimated time: ~3 hours on 8xH100 (depth={cfg['depth']}, fp8={cfg['enable_fp8']})")
        if cfg["enable_rl"]:
            lab.log("✅ RL training is ENABLED (will train on GSM8K)")
        else:
            lab.log("⏭️  RL training is DISABLED (set enable_rl=true to enable)")

        # ------------------------------------------------------------------
        # 6. Run all training phases
        # ------------------------------------------------------------------
        setup_environment(base_dir, nanochat_dir, nproc)

        # Initialize report (after environment setup)
        run_command("uv run python -m nanochat.report reset", "Initializing report", cwd=nanochat_dir)

        # Activate the venv so that `python` resolves to the project's venv
        venv_activate = os.path.join(nanochat_dir, ".venv", "bin", "activate")
        if os.path.exists(venv_activate):
            venv_bin = os.path.join(nanochat_dir, ".venv", "bin")
            os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")
            os.environ["VIRTUAL_ENV"] = os.path.join(nanochat_dir, ".venv")

        # Phase 1: Tokenizer
        dataset_process = train_tokenizer(
            nanochat_dir,
            initial_shards=cfg["initial_dataset_shards"],
            total_shards=cfg["total_dataset_shards"],
            max_chars=cfg["tokenizer_max_chars"],
            vocab_size=cfg["tokenizer_vocab_size"],
        )

        # Phase 2: Base model pretraining
        train_base_model(nanochat_dir, dataset_process, nproc, cfg)

        # Phase 3: SFT
        train_sft(nanochat_dir, nproc, cfg)

        # Phase 4: RL (optional)
        train_rl(nanochat_dir, nproc, cfg)

        # Phase 5: Report
        generate_report(nanochat_dir)

        # ------------------------------------------------------------------
        # 7. Wrap up
        # ------------------------------------------------------------------
        lab.update_progress(100)
        lab.log("✅ All training phases completed!")

        # List found checkpoints
        models_found = []
        depth = cfg["depth"]
        model_tag = cfg["model_tag"] if cfg["model_tag"] else f"d{depth}"
        for ckpt_dir_name in ["base_checkpoints", "chatsft_checkpoints", "chatrl_checkpoints"]:
            ckpt_path = os.path.join(base_dir, ckpt_dir_name, model_tag)
            if os.path.exists(ckpt_path):
                models_found.append(ckpt_dir_name)
        lab.log(f"✅ Found checkpoint dirs: {', '.join(models_found) if models_found else 'none'}")

        # Check for report
        report_path = os.path.join(nanochat_dir, "report.md")
        if os.path.exists(report_path):
            lab.log("✅ Training report generated successfully")
        else:
            lab.log("⚠️  Training report not found")

        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"🎉 Training completed in {training_duration}")

        lab.finish()

        return {
            "status": "success",
            "duration": str(training_duration),
            "model": f"nanochat-{model_tag}",
            "checkpoints": models_found,
            "base_dir": base_dir,
        }

    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        print("Training interrupted by user")
        return {"status": "stopped"}

    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n✅ Nanochat Training Result: {result}")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
