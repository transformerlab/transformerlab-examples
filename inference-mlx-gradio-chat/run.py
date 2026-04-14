"""Launch the MLX Gradio chat entrypoint."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def main() -> int:
    model_name = os.environ.get("MODEL_NAME", "mlx-community/Llama-3.2-3B-Instruct-4bit")

    try:
        import gradio  # pylint: disable=import-outside-toplevel

        print(f"gradio found at: {gradio.__file__}", flush=True)
    except Exception:
        print("gradio NOT found in run python", flush=True)

    script_dir = pathlib.Path(__file__).resolve().parent
    gradio_script = script_dir / "gradio_chat.py"
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name

    process = subprocess.Popen(["python", str(gradio_script)], env=env)
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
