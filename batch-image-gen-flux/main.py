#!/usr/bin/env python3
import os
import shlex
import subprocess
import shutil
from time import sleep, perf_counter
from lab import lab

def slugify(s: str, maxlen: int = 64) -> str:
    keep = []
    for ch in s.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")

    return "".join(keep)[:maxlen].rstrip("_")

def main():
    lab.init()
    config = lab.get_config()
    model_dir = config.get("model_dir", "flux-klein-model")
    model_dir = os.path.expanduser(model_dir)
    output_dir = config.get("output_dir", "./generated-images")
    prompts = config.get(
        "prompts",
        [
            "A fluffy orange cat sitting on a windowsill",
            "A serene mountain lake at sunrise",
            "A futuristic city skyline at night with neon lights",
            "A vintage steam train crossing a snowy bridge",
        ],
    )

    os.makedirs(output_dir, exist_ok=True)
    lab.log(f"Running batch image generation with model with ({len(prompts)} prompts): {model_dir}")

    results = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        safe_name = f"{i:02d}_{slugify(prompt)}.png"
        out_path = os.path.join(output_dir, safe_name)
        env_dir = os.environ.get("SKY_FLUX_DIR")
        skypilot_path = os.path.expanduser("/home/sky/sky_workdir/flux2.c")
        local_dir = "flux2.c"

        flux_dir = None
        if env_dir:
            flux_dir = env_dir
        elif os.path.isdir(skypilot_path):
            flux_dir = skypilot_path
        elif os.path.isdir(local_dir):
            flux_dir = local_dir

        if flux_dir and os.path.isdir(flux_dir):
            cmd = f"cd {shlex.quote(flux_dir)} && ./flux -d {shlex.quote(model_dir)} -p {shlex.quote(prompt)} -o {shlex.quote(out_path)}"
        else:
            flux_path = shutil.which("flux")
            if flux_path:
                cmd = f"{shlex.quote(flux_path)} -d {shlex.quote(model_dir)} -p {shlex.quote(prompt)} -o {shlex.quote(out_path)}"
            else:
                # Fallback: try running ./flux from current working directory
                cmd = f"./flux -d {shlex.quote(model_dir)} -p {shlex.quote(prompt)} -o {shlex.quote(out_path)}"
        lab.log(f"[{i}/{total}] Running: {cmd}")

        start_time = perf_counter()
        try:
            proc = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            duration = perf_counter() - start_time
            lab.log(proc.stdout or "")
            lab.log(f"⏱️ Generation time for prompt #{i}: {duration:.2f}s")
            if proc.returncode != 0:
                lab.log(f"⚠️ flux exited with code {proc.returncode} for prompt #{i}")
                results.append({"prompt": prompt, "output": None, "status": "error", "rc": proc.returncode, "duration_s": duration})
            else:
                if os.path.exists(out_path):
                    saved = lab.save_artifact(out_path, name=os.path.basename(out_path), type="image")
                    lab.log(f"✅ Saved image: {saved}")
                    results.append({"prompt": prompt, "output": out_path, "status": "success", "artifact": saved, "duration_s": duration})
                else:
                    lab.log(f"⚠️ Output image not found for prompt #{i}")
                    results.append({"prompt": prompt, "output": None, "status": "missing", "duration_s": duration})
        except Exception as e:
            duration = perf_counter() - start_time
            lab.log(f"⚠️ Exception occurred for prompt #{i}: {e}")
            results.append({"prompt": prompt, "output": None, "status": "exception", "error": str(e), "duration_s": duration})

        lab.update_progress(int((i / total) * 100))
        sleep(0.2)
    
    lab.log("Batch inference complete")
    lab.finish("Batch flux inference finished")
    return {"status": "success", "generated": results, "output_dir": output_dir}

if __name__ == "__main__":
    print("🚀 Running batch flux inference...")
    res = main()
    print(res)