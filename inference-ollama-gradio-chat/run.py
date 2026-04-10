"""Start Ollama + Gradio chat and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

LOG_FILES = [
    pathlib.Path("/tmp/ollama.log"),
    pathlib.Path("/tmp/ollama-pull.log"),
    pathlib.Path("/tmp/gradio.log"),
    pathlib.Path("/tmp/ngrok.log"),
]


def _touch_logs() -> None:
    for path in LOG_FILES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def _line_prefix(path: pathlib.Path) -> str:
    return f"[{path.name}] "


def _tail_forever() -> None:
    offsets = {path: 0 for path in LOG_FILES}
    while True:
        saw_data = False
        for path in LOG_FILES:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offsets[path])
                    chunk = handle.read()
                    offsets[path] = handle.tell()
            except FileNotFoundError:
                continue

            if chunk:
                saw_data = True
                prefix = _line_prefix(path)
                for line in chunk.splitlines():
                    print(f"{prefix}{line}", flush=True)

        if not saw_data:
            time.sleep(0.25)


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "llama2")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"

    _touch_logs()

    with open("/tmp/ollama.log", "w", encoding="utf-8") as ollama_log:
        subprocess.Popen(["ollama", "serve"], stdout=ollama_log, stderr=subprocess.STDOUT, env=env)
    time.sleep(3)

    with open("/tmp/ollama-pull.log", "w", encoding="utf-8") as pull_log:
        subprocess.Popen(["ollama", "pull", model_name], stdout=pull_log, stderr=subprocess.STDOUT, env=env)
    time.sleep(5)

    try:
        import gradio  # pylint: disable=import-outside-toplevel

        print(f"gradio found at: {gradio.__file__}", flush=True)
    except Exception:
        print("gradio NOT found in run python", flush=True)

    script_dir = pathlib.Path(__file__).resolve().parent
    gradio_script = script_dir / "gradio_chat.py"
    gradio_env = env.copy()
    gradio_env["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    gradio_env["MODEL_NAME"] = model_name
    with open("/tmp/gradio.log", "w", encoding="utf-8") as gradio_log:
        subprocess.Popen(
            ["python", str(gradio_script)],
            stdout=gradio_log,
            stderr=subprocess.STDOUT,
            env=gradio_env,
        )
    time.sleep(5)

    _tail_forever()


if __name__ == "__main__":
    main()
