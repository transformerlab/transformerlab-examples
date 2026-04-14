"""Launch Ollama + Open WebUI and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

LOG_FILES = [
    pathlib.Path("/tmp/ollama.log"),
    pathlib.Path("/tmp/ollama-pull.log"),
    pathlib.Path("/tmp/openwebui.log"),
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
    If a subprocess fails, dump its log.
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
    rc = proc.poll()
    if rc is not None:
        print(
            f"ERROR: {name} failed (exit code {rc})",
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
    ollama_process = subprocess.Popen(
        ["ollama", "serve"], stdout=ollama_log, stderr=subprocess.STDOUT, env=env
    )
    time.sleep(3)
    _check_proc(ollama_process, "Ollama", pathlib.Path("/tmp/ollama.log"))

    pull_log = open("/tmp/ollama-pull.log", "w", encoding="utf-8")
    pull_process = subprocess.Popen(
        ["ollama", "pull", model_name],
        stdout=pull_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    time.sleep(5)
    _check_proc(pull_process, "Ollama pull", pathlib.Path("/tmp/ollama-pull.log"))

    webui_env = env.copy()
    webui_env["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    webui_env["WEBUI_AUTH"] = "false"
    webui_log = open("/tmp/openwebui.log", "w", encoding="utf-8")
    webui_process = subprocess.Popen(
        ["open-webui", "serve", "--host", "0.0.0.0", "--port", "8080"],
        stdout=webui_log,
        stderr=subprocess.STDOUT,
        env=webui_env,
    )
    time.sleep(5)
    _check_proc(webui_process, "Open WebUI", pathlib.Path("/tmp/openwebui.log"))

    _tail_and_monitor(
        {
            "Ollama": (ollama_process, pathlib.Path("/tmp/ollama.log")),
            "Open WebUI": (webui_process, pathlib.Path("/tmp/openwebui.log")),
        }
    )


if __name__ == "__main__":
    main()
