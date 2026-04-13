"""Start Ollama + Gradio chat and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

LOG_FILES = [
    pathlib.Path("/tmp/ollama.log"),
    pathlib.Path("/tmp/ollama-pull.log"),
    pathlib.Path("/tmp/gradio.log"),
    pathlib.Path("/tmp/ngrok.log"),
]

CHECK_INTERVAL = 5


def _touch_logs() -> None:
    for path in LOG_FILES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def _line_prefix(path: pathlib.Path) -> str:
    return f"[{path.name}] "


def _dump_log(path: pathlib.Path) -> None:
    """
    If a subprocess fails, dump it's log.
    Report if the log doesn't exist yet.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            print(f"=== START {path.name} ===", file=sys.stderr, flush=True)
            print(text, file=sys.stderr, flush=True)
            print(f"=== END {path.name} ===", file=sys.stderr, flush=True)
    except FileNotFoundError:
        print("Log file not found:", path.name)
        pass


def _check_proc(proc: subprocess.Popen, name: str, log_path: pathlib.Path) -> None:
    """
    Check if a process has exited and, if so, output logs and return.
    """
    if proc.poll() is not None:
        print(
            f"ERROR: {name} failed (exit code {proc.returncode})",
            file=sys.stderr,
            flush=True,
        )
        _dump_log(log_path)
        sys.exit(1)


def _tail_and_monitor(procs: dict[str, tuple[subprocess.Popen, pathlib.Path]]) -> None:
    offsets = {path: 0 for path in LOG_FILES}
    last_check = time.monotonic()

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

        now = time.monotonic()
        if now - last_check >= CHECK_INTERVAL:
            last_check = now
            for name, (proc, log_path) in procs.items():
                _check_proc(proc, name, log_path)

        if not saw_data:
            time.sleep(0.25)


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "llama2")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"

    _touch_logs()

    ollama_log = open("/tmp/ollama.log", "w", encoding="utf-8")
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"], stdout=ollama_log, stderr=subprocess.STDOUT, env=env
    )
    time.sleep(3)
    _check_proc(ollama_proc, "Ollama", pathlib.Path("/tmp/ollama.log"))

    pull_log = open("/tmp/ollama-pull.log", "w", encoding="utf-8")
    pull_proc = subprocess.Popen(
        ["ollama", "pull", model_name],
        stdout=pull_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    time.sleep(5)
    _check_proc(pull_proc, "Ollama pull", pathlib.Path("/tmp/ollama-pull.log"))

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
    gradio_log = open("/tmp/gradio.log", "w", encoding="utf-8")
    gradio_proc = subprocess.Popen(
        ["python", str(gradio_script)],
        stdout=gradio_log,
        stderr=subprocess.STDOUT,
        env=gradio_env,
    )
    time.sleep(5)
    _check_proc(gradio_proc, "Gradio", pathlib.Path("/tmp/gradio.log"))

    _tail_and_monitor(
        {
            "Ollama": (ollama_proc, pathlib.Path("/tmp/ollama.log")),
            "Gradio": (gradio_proc, pathlib.Path("/tmp/gradio.log")),
        }
    )


if __name__ == "__main__":
    main()
