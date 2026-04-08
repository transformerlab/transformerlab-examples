#!/usr/bin/env python3
import os
import torch
from datetime import datetime
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from huggingface_hub import login, snapshot_download
from lab import lab

def safe_hf_login():
    token = os.getenv("HF_TOKEN")

    if not token:
        print("ℹ️ No HF_TOKEN found, proceeding without login (public models only).")
        return

    if not token.startswith("hf_"):
        print("⚠️ Invalid HF_TOKEN format, skipping login.")
        return

    try:
        login(token=token)
        print("✅ Hugging Face login successful")
    except Exception as e:
        print(f"⚠️ HF login failed, continuing without auth: {e}")


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"

def main():
    try:
        lab.init()
        config = lab.get_config()

        # Safe login (non-blocking)
        safe_hf_login()

        # Config
        model_id = config.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        prompt = config.get("prompt", "A cat walks on the grass, realistic")
        negative_prompt = config.get("negative_prompt", "")
        height = int(config.get("height", 480))
        width = int(config.get("width", 832))
        num_frames = int(config.get("num_frames", 81))
        guidance_scale = float(config.get("guidance_scale", 5.0))
        fps = int(config.get("fps", 15))

        output_dir = "./output"
        output_filename = "generated_video.mp4"
        os.makedirs(output_dir, exist_ok=True)

        start_time = datetime.now()
        lab.log(f"🚀 Start: {start_time}")
        lab.log(f"Model: {model_id}")

        # Device setup
        device = get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        lab.log(f"Using device: {device}")

        lab.update_progress(10)
        lab.log("Loading model...")

        try:
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=dtype,
            )

        except Exception as e:
            lab.log(f"⚠️ Direct load failed: {e}")
            lab.log("⬇️ Downloading model manually...")

            local_repo = snapshot_download(repo_id=model_id)

            vae = AutoencoderKLWan.from_pretrained(
                local_repo,
                subfolder="vae",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

            pipe = WanPipeline.from_pretrained(
                local_repo,
                vae=vae,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        pipe.to(device)

        lab.update_progress(50)
        lab.log("🎬 Generating video...")

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
            )

        frames = result.frames[0]

        lab.update_progress(80)
        output_path = os.path.join(output_dir, output_filename)

        export_to_video(frames, output_path, fps=fps)

        lab.log(f"✅ Saved: {output_path}")

        artifact_path = lab.save_artifact(
            output_path,
            output_filename,
            type="video",
        )

        end_time = datetime.now()
        duration = end_time - start_time

        lab.update_progress(100)
        lab.finish(f"✅ Done in {duration}")

        return {
            "status": "success",
            "artifact": artifact_path,
            "duration": str(duration),
        }

    except Exception as e:
        lab.error(str(e))
        return {
            "status": "error",
            "error": str(e),
        }

if __name__ == "__main__":
    print(main())
