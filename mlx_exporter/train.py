#!/usr/bin/env python3
"""
MLX Exporter Script - Converts models to MLX format for Apple Silicon.
This script exports a model to MLX format so it can be run and trained on Mac with Apple Silicon.
"""

import os
import subprocess
import sys
from datetime import datetime

from lab import lab

# Login to huggingface if token is available
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def export_to_mlx():
    """Export a model to MLX format using mlx-lm conversion tools"""

    try:
        # Initialize lab
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "mlx-community/Llama-3.2-1B-Instruct-4bit")
        output_dir = config.get("output_dir", "./output")
        q_bits = config.get("q_bits", "4")
        
        # Convert q_bits to string if needed
        q_bits = str(q_bits)
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"MLX Export started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Quantization bits: {q_bits}")
        lab.log(f"Output directory: {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        lab.update_progress(5)

        # Prepare the mlx_lm convert command
        command = [
            sys.executable,
            "-u",
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            model_name,
            "--mlx-path",
            output_dir,
            "-q",
            "--q-bits",
            q_bits,
        ]

        lab.log("Starting MLX conversion...")
        lab.log(f"Running command: {' '.join(command)}")
        lab.update_progress(10)

        try:
            # Run the conversion command
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            all_output_lines = []
            progress_value = 10

            # Monitor the conversion process
            for line in process.stdout:
                line = line.strip()
                if line:
                    all_output_lines.append(line)
                    lab.log(line)

                    # Update progress based on output keywords
                    if "Loading" in line or "loading" in line.lower():
                        progress_value = 20
                        lab.log("📥 Loading model...")
                    elif "Fetching" in line or "fetching" in line.lower():
                        progress_value = 35
                        lab.log("📦 Fetching model files...")
                    elif "Using dtype" in line or "dtype" in line.lower():
                        progress_value = 50
                        lab.log("🔧 Preparing quantization...")
                    elif "Quantizing" in line or "quantizing" in line.lower():
                        progress_value = 65
                        lab.log("⚙️  Quantizing model...")
                    elif "Quantized model" in line or "quantized" in line.lower():
                        progress_value = 80
                        lab.log("✨ Finalizing model...")
                    elif "Saving" in line or "saving" in line.lower():
                        progress_value = 90
                        lab.log("💾 Saving model...")

                    lab.update_progress(progress_value)

            return_code = process.wait()

            if return_code != 0:
                error_msg = f"MLX conversion failed with return code {return_code}"
                lab.log(f"❌ {error_msg}")
                lab.error(error_msg)
                return {"status": "error", "error": error_msg, "output": "\n".join(all_output_lines)}

        except Exception as e:
            error_msg = f"MLX conversion failed with exception: {str(e)}"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": str(e)}

        # Success
        lab.log("✅ MLX conversion completed successfully!")
        lab.update_progress(100)

        # Log completion time
        end_time = datetime.now()
        duration = end_time - start_time
        lab.log(f"Export completed at {end_time}")
        lab.log(f"Total duration: {duration}")
        lab.log(f"Output saved to: {output_dir}")

        return {
            "status": "success",
            "model_name": model_name,
            "output_dir": output_dir,
            "q_bits": q_bits,
            "duration": str(duration),
        }

    except Exception as e:
        lab.log(f"❌ Unexpected error: {str(e)}")
        lab.error(f"Export failed: {str(e)}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    result = export_to_mlx()
    if result.get("status") == "error":
        sys.exit(1)
    sys.exit(0)
