#!/usr/bin/env python3
"""
Llamafile Exporter Script - Exports GGUF models to self-executing llamafiles.
This script converts a GGUF model into a fully contained self-executing llamafile
that can run on multiple platforms without dependencies.
"""

import os
import subprocess
import shutil
from datetime import datetime

from lab import lab

# Login to huggingface if token is available
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


def export_to_llamafile():
    """Export a GGUF model to Llamafile format"""

    try:
        # Initialize lab
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "gpt-j-6b")
        model_path = config.get("model_path", "./model.gguf")
        output_dir = config.get("output_dir", "./output")

        # Derive model name without author
        input_model_id_without_author = model_name.split("/")[-1]
        outfile_name = f"{input_model_id_without_author}.llamafile"

        # Log start time
        start_time = datetime.now()
        lab.log(f"Llamafile Export started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Model path: {model_path}")
        lab.log(f"Output directory: {output_dir}")
        lab.log(f"Output filename: {outfile_name}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        lab.update_progress(5)

        # Verify input model exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": error_msg}

        lab.log(f"✅ Input model found: {model_path}")
        lab.update_progress(10)

        # Download llamafile and zipalign tools
        LATEST_VERSION = "0.9.0"
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        llamafile_path = os.path.join(plugin_dir, "llamafile")
        zipalign_path = os.path.join(plugin_dir, "zipalign")

        if not os.path.exists(llamafile_path):
            lab.log(f"Downloading llamafile {LATEST_VERSION}...")
            try:
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        f"https://github.com/Mozilla-Ocho/llamafile/releases/download/{LATEST_VERSION}/llamafile-{LATEST_VERSION}",
                        "-o",
                        llamafile_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.chmod(llamafile_path, 0o755)
                lab.log("✅ llamafile downloaded")
            except subprocess.CalledProcessError as e:
                lab.log(f"❌ Failed to download llamafile: {e}")
                lab.error("Export failed due to llamafile download error")
                return {"status": "error", "error": str(e)}
        else:
            lab.log("✅ Using existing llamafile")

        lab.update_progress(20)

        if not os.path.exists(zipalign_path):
            lab.log(f"Downloading zipalign {LATEST_VERSION}...")
            try:
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        f"https://github.com/Mozilla-Ocho/llamafile/releases/download/{LATEST_VERSION}/zipalign-{LATEST_VERSION}",
                        "-o",
                        zipalign_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.chmod(zipalign_path, 0o755)
                lab.log("✅ zipalign downloaded")
            except subprocess.CalledProcessError as e:
                lab.log(f"❌ Failed to download zipalign: {e}")
                lab.error("Export failed due to zipalign download error")
                return {"status": "error", "error": str(e)}
        else:
            lab.log("✅ Using existing zipalign")

        lab.update_progress(30)

        # Create .args file
        lab.log("Creating .args file...")
        argsfile = os.path.join(plugin_dir, ".args")
        argsoutput = f"""-m
{input_model_id_without_author}
--host
0.0.0.0
-ngl
9999
"""

        with open(argsfile, "w") as f:
            f.write(argsoutput)

        lab.log("✅ .args file created")
        lab.update_progress(40)

        # Copy base llamafile to create output file
        lab.log("Creating base llamafile...")
        temp_llamafile = os.path.join(plugin_dir, outfile_name)
        shutil.copy(llamafile_path, temp_llamafile)
        lab.log("✅ Base llamafile copied")
        lab.update_progress(50)

        # Merge files together in single executable using zipalign
        lab.log("Merging model with llamafile using zipalign...")
        subprocess_cmd = [
            "sh",
            zipalign_path,
            "-j0",
            temp_llamafile,
            model_path,
            argsfile,
        ]

        lab.log(f"Running command: {' '.join(subprocess_cmd)}")

        try:
            export_process = subprocess.run(
                subprocess_cmd,
                cwd=plugin_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            stdout = export_process.stdout
            for line in stdout.strip().splitlines():
                lab.log(line)

            if export_process.returncode != 0:
                error_msg = f"zipalign failed with return code {export_process.returncode}"
                lab.log(f"❌ {error_msg}")
                lab.log(f"Output:\n{stdout}")
                lab.error(error_msg)
                return {"status": "error", "error": error_msg}

        except Exception as e:
            error_msg = f"zipalign execution failed: {str(e)}"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": str(e)}

        lab.log("✅ Model merged successfully")
        lab.update_progress(80)

        # Move the final llamafile to output directory
        lab.log(f"Moving {outfile_name} to output directory...")
        final_llamafile_path = os.path.join(output_dir, outfile_name)

        try:
            shutil.move(temp_llamafile, final_llamafile_path)
            # Make sure the final file is executable
            os.chmod(final_llamafile_path, 0o755)
            lab.log(f"✅ Llamafile moved to: {final_llamafile_path}")
        except Exception as e:
            error_msg = f"Failed to move llamafile: {str(e)}"
            lab.log(f"❌ {error_msg}")
            lab.error(error_msg)
            return {"status": "error", "error": str(e)}

        lab.update_progress(90)

        # Get file size
        file_size = os.path.getsize(final_llamafile_path)
        file_size_mb = file_size / (1024 * 1024)
        lab.log(f"✅ Llamafile size: {file_size_mb:.2f} MB")

        # Create export summary
        export_summary_file = os.path.join(output_dir, "export_summary.json")
        import json

        with open(export_summary_file, "w") as f:
            json.dump(
                {
                    "export_type": "Llamafile",
                    "model_name": model_name,
                    "output_file": final_llamafile_path,
                    "file_size_mb": file_size_mb,
                    "completed_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        export_artifact_path = lab.save_artifact(
            export_summary_file, "export_summary.json"
        )
        lab.log(f"Saved export summary: {export_artifact_path}")

        # Save the llamafile as a model artifact
        try:
            saved_model_path = lab.save_model(
                final_llamafile_path, name=f"{input_model_id_without_author}_llamafile"
            )
            lab.log(f"✅ Llamafile saved to job models directory: {saved_model_path}")
        except Exception as e:
            lab.log(f"⚠️  Could not save llamafile as artifact: {e}")

        # Clean up temporary files
        try:
            if os.path.exists(argsfile):
                os.remove(argsfile)
                lab.log("🧹 Cleaned up .args file")
        except Exception as e:
            lab.log(f"⚠️  Could not clean up temporary files: {e}")

        end_time = datetime.now()
        export_duration = end_time - start_time
        lab.log(f"Export completed in {export_duration}")

        lab.update_progress(100)
        lab.finish("Llamafile export completed successfully")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(export_duration),
            "output_file": final_llamafile_path,
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
    print("🚀 Starting Llamafile export...")
    result = export_to_llamafile()
    print("Export result:", result)
