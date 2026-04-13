"""Launch Jupyter Lab and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

LOG_FILES = [pathlib.Path("/tmp/jupyter.log"), pathlib.Path("/tmp/ngrok.log")]


def _touch_logs() -> None:
    for path in LOG_FILES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def _tail_forever() -> None:
    offsets = {path: 0 for path in LOG_FILES}
    while True:
        saw_data = False
        for path in LOG_FILES:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offsets[path])
                chunk = handle.read()
                offsets[path] = handle.tell()
            if chunk:
                saw_data = True
                for line in chunk.splitlines():
                    print(f"[{path.name}] {line}", flush=True)
        if not saw_data:
            time.sleep(0.25)


def _ensure_running(process: subprocess.Popen[bytes], name: str) -> None:
    code = process.poll()
    if code is not None:
        raise RuntimeError(f"{name} exited early with code {code}")


def main() -> None:
    _touch_logs()

    with open("/tmp/jupyter.log", "w", encoding="utf-8") as jupyter_log:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--ip=0.0.0.0",
                "--port=8888",
                "--no-browser",
                "--allow-root",
                "--NotebookApp.base_url=''",
                "--NotebookApp.token=",
                "--NotebookApp.password=",
                f"--notebook-dir={os.path.expanduser('~')}",
            ],
            stdout=jupyter_log,
            stderr=subprocess.STDOUT,
        )
    time.sleep(3)
    _ensure_running(jupyter_process, "jupyter lab")

    _tail_forever()


if __name__ == "__main__":
    main()
