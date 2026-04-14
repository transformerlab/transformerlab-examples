"""Launch MLX audio TTS Gradio app."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    try:
        espeak_prefix = subprocess.check_output(["brew", "--prefix", "espeak-ng"], text=True).strip()
        env["ESPEAK_DATA_PATH"] = f"{espeak_prefix}/share/espeak-ng-data"
    except Exception:
        env.setdefault("ESPEAK_DATA_PATH", "/opt/homebrew/share/espeak-ng-data")

    print(f"ESPEAK_DATA_PATH={env.get('ESPEAK_DATA_PATH', '')}", flush=True)

    script_dir = pathlib.Path(__file__).resolve().parent
    gradio_script = script_dir / "gradio_tts.py"
    process = subprocess.Popen(["python", str(gradio_script)], env=env)
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
