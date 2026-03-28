#!/usr/bin/env python3
"""
GGUF Exporter Script - Converts models to GGUF format for CPU inference.
This script exports a model to GGUF format so you can interact with it on systems without GPUs.
"""

import os
import subprocess
import sys
from datetime import datetime
from huggingface_hub import snapshot_download
import contextlib
import io

from lab import lab

# Login to huggingface if token is available
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def export_to_gguf():
    """Export a model to GGUF format using llama.cpp conversion tools"""

    try:
        # Initialize lab
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "unsloth/Qwen2.5-0.5B-Instruct")
        output_dir = config.get("output_dir", "./output")
        outtype = config.get("outtype", "q8_0")
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"GGUF Export started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Output type: {outtype}")
        lab.log(f"Output directory: {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        lab.update_progress(5)

        # Download or locate the model
        lab.log("Locating model files...")
        model_path = model_name
        
        if not os.path.exists(model_path):
            lab.log("Downloading model from Hugging Face...")
            lab.update_progress(10)
            
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model_path = snapshot_download(
                        repo_id=model_name,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.py",
                            "tokenizer.model",
                            "*.tiktoken",
                        ],
                    )
                lab.log(f"✅ Model downloaded to: {model_path}")
            except Exception as e:
                lab.log(f"❌ Failed to download model: {e}")
                lab.error("Export failed due to model download error")
                return {"status": "error", "error": str(e)}
        else:
            lab.log(f"✅ Using local model at: {model_path}")

        lab.update_progress(20)

        # Prepare the output filename
        output_filename = os.path.join(output_dir, f"{model_name.replace('/', '_')}.gguf")
        
        # Clone llama.cpp if not already present
        llama_cpp_dir = os.path.join(os.path.dirname(__file__), "llama.cpp")
        if not os.path.exists(llama_cpp_dir):
            lab.log("Cloning llama.cpp repository...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
                lab.log("✅ llama.cpp cloned successfully")
            except subprocess.CalledProcessError as e:
                lab.log(f"❌ Failed to clone llama.cpp: {e}")
                lab.error("Export failed due to llama.cpp setup error")
                return {"status": "error", "error": str(e)}
        
        lab.update_progress(30)

        # Install llama.cpp requirements
        lab.log("Installing llama.cpp requirements...")
        requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file],
                    check=True,
                    capture_output=True,
                    text=True
                )
                lab.log("✅ Requirements installed")
            except subprocess.CalledProcessError as e:
                lab.log(f"⚠️  Warning: Some requirements may not have installed: {e}")

        lab.update_progress(40)

        # Run the conversion script
        conversion_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
        
        if not os.path.exists(conversion_script):
            # Try alternative script name
            conversion_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        
        if not os.path.exists(conversion_script):
            lab.log(f"❌ Conversion script not found in llama.cpp directory")
            lab.error("Export failed due to missing conversion script")
            return {"status": "error", "error": "Conversion script not found"}

        lab.log(f"Converting model to GGUF format ({outtype})...")
        lab.update_progress(50)
        
        command = [
            sys.executable,
            conversion_script,
            "--outfile",
            output_filename,
            "--outtype",
            outtype,
            model_path,
        ]

        lab.log(f"Running command: {' '.join(command)}")

        try:
            with subprocess.Popen(
                command,
                cwd=llama_cpp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            ) as process:
                output_lines = []
                progress_value = 50

                for line in process.stdout:
                    line = line.strip()
                    output_lines.append(line)
                    lab.log(line)

                    # Detect writing progress
                    if "Writing:" in line or "writing" in line.lower():
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            writing_percent = int(match.group(1))
                            # Map 0-100 writing progress to 50-95 overall progress
                            progress_value = 50 + int(writing_percent * 0.45)
                            lab.update_progress(progress_value)

                return_code = process.wait()

                if return_code != 0:
                    error_msg = f"GGUF conversion failed with return code {return_code}"
                    lab.log(f"❌ {error_msg}")
                    lab.log(f"Output:\n{''.join(output_lines)}")
                    lab.error(error_msg)
                    return {"status": "error", "error": error_msg}

        except Exception as e:
            error_msg = f"GGUF conversion failed with exception: {str(e)}"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": str(e)}

        lab.update_progress(95)

        # Verify output file exists
        if not os.path.exists(output_filename):
            error_msg = "GGUF file was not created"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": error_msg}

        # Get file size
        file_size = os.path.getsize(output_filename)
        file_size_mb = file_size / (1024 * 1024)
        lab.log(f"✅ GGUF file created: {output_filename} ({file_size_mb:.2f} MB)")

        # Save export summary
        export_summary_file = os.path.join(output_dir, "export_summary.json")
        import json
        
        with open(export_summary_file, "w") as f:
            json.dump(
                {
                    "export_type": "GGUF",
                    "model_name": model_name,
                    "output_file": output_filename,
                    "output_type": outtype,
                    "file_size_mb": file_size_mb,
                    "completed_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        export_artifact_path = lab.save_artifact(export_summary_file, "export_summary.json")
        lab.log(f"Saved export summary: {export_artifact_path}")

        # Save the GGUF file as a model artifact
        try:
            saved_model_path = lab.save_model(output_filename, name=f"{model_name.replace('/', '_')}_gguf")
            lab.log(f"✅ GGUF model saved to job models directory: {saved_model_path}")
        except Exception as e:
            lab.log(f"⚠️  Could not save GGUF model as artifact: {e}")

        end_time = datetime.now()
        export_duration = end_time - start_time
        lab.log(f"Export completed in {export_duration}")

        lab.update_progress(100)
        lab.finish("GGUF export completed successfully")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(export_duration),
            "output_file": output_filename,
            "output_type": outtype,
            "file_size_mb": file_size_mb,
        }

    except KeyboardInterrupt:
        lab.error("Export stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        lab.log(f"❌ Export failed: {error_msg}")
        
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting GGUF export...")
    result = export_to_gguf()
    print("Export result:", result)
