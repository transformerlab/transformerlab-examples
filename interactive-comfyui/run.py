"""Launch ComfyUI and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

LOG_FILES = [pathlib.Path("/tmp/comfy-ui.log"), pathlib.Path("/tmp/ngrok.log")]


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

    with open("/tmp/comfy-ui.log", "w", encoding="utf-8") as comfy_log:
        comfy_process = subprocess.Popen(
            ["comfy", "launch", "--", "--listen", "0.0.0.0", "--enable-manager"],
            stdout=comfy_log,
            stderr=subprocess.STDOUT,
        )
    time.sleep(5)
    _ensure_running(comfy_process, "comfy launch")

    _tail_forever()


if __name__ == "__main__":
    main()
