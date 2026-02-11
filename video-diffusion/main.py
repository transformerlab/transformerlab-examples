#!/usr/bin/env python3
import os
import torch
from datetime import datetime
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
import ftfy

from lab import lab

# Login to HuggingFace if token is provided
from huggingface_hub import login, snapshot_download
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

def main():
    try:
        # Initialize lab
        lab.init()
        config = lab.get_config()

        # Extract parameters with defaults
        model_id = config.get("model_id", "samuelchristlie/Wan2.1-T2V-1.3B-GGUF")
        prompt = config.get("prompt", "A cat walks on the grass, realistic")
        negative_prompt = config.get("negative_prompt", "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
        height = int(config.get("height", 480))
        width = int(config.get("width", 832))
        num_frames = int(config.get("num_frames", 81))
        guidance_scale = float(config.get("guidance_scale", 5.0))
        output_dir = config.get("output_dir", "./output")
        output_filename = config.get("output_filename", "generated_video.mp4")
        fps = int(config.get("fps", 15))

        # Log start
        start_time = datetime.now()
        lab.log(f"🚀 Video generation started at {start_time}")
        lab.log(f"Model: {model_id}")
        lab.log(f"Prompt: {prompt}")
        lab.log(f"Using GPU: {torch.cuda.is_available()}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load VAE and pipeline
        lab.update_progress(10)
        lab.log("Loading VAE and pipeline...")
        try:
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            err = str(e)
            if "config.json" in err or "does not appear to have a file named config.json" in err:
                lab.log("⚠️  config.json missing — downloading model repo to local cache...")
                local_repo = snapshot_download(repo_id=model_id, repo_type="model")
                lab.log(f"✅ Model repo downloaded to: {local_repo}")

                # Try loading from the downloaded local repo (allow remote code if present)
                try:
                    vae = AutoencoderKLWan.from_pretrained(local_repo, subfolder="vae", torch_dtype=torch.float32, trust_remote_code=True)
                    pipe = WanPipeline.from_pretrained(local_repo, vae=vae, torch_dtype=torch.bfloat16, trust_remote_code=True)
                    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                except Exception as e2:
                    lab.log(f"⚠️  Failed to instantiate pipeline from local repo: {e2}")
                    # Zip the downloaded repo and save as a model artifact so the files are available
                    import shutil
                    safe_name = model_id.replace("/", "_")
                    archive_base = os.path.join(output_dir, safe_name)
                    archive_path = shutil.make_archive(archive_base, "zip", local_repo)
                    artifact_path = lab.save_artifact(archive_path, os.path.basename(archive_path), type="model")
                    lab.log(f"✅ Downloaded model repo saved as artifact: {artifact_path}")
                    return {"status": "success", "downloaded_repo": local_repo, "artifact_path": artifact_path}
            else:
                raise

        # Generate video
        lab.update_progress(50)
        lab.log("Generating video...")
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale
        ).frames[0]

        # Export to video
        lab.update_progress(80)
        output_path = os.path.join(output_dir, output_filename)
        export_to_video(output, output_path, fps=fps)
        lab.log(f"✅ Video saved to {output_path}")

        # Save as artifact
        artifact_path = lab.save_artifact(output_path, output_filename, type="video")
        lab.log(f"Saved video artifact: {artifact_path}")

        # Complete
        end_time = datetime.now()
        duration = end_time - start_time
        lab.update_progress(100)
        lab.finish(f"Video generation completed in {duration}")

        return {
            "status": "success",
            "output_path": output_path,
            "artifact_path": artifact_path,
            "duration": str(duration),
            "model": model_id,
            "prompt": prompt
        }

    except Exception as e:
        error_msg = str(e)
        lab.error(f"Video generation failed: {error_msg}")
        return {"status": "error", "error": error_msg}

if __name__ == "__main__":
    result = main()
    print("Video generation result:", result)
