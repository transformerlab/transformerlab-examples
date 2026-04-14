"""Launch ComfyUI and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

LOG_FILES = [pathlib.Path("/tmp/comfy-ui.log"), pathlib.Path("/tmp/ngrok.log")]

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
    _touch_logs()

    comfy_log = open("/tmp/comfy-ui.log", "w", encoding="utf-8")
    comfy_process = subprocess.Popen(
        ["comfy", "launch", "--", "--listen", "0.0.0.0", "--enable-manager"],
        stdout=comfy_log,
        stderr=subprocess.STDOUT,
    )
    time.sleep(5)
    _check_proc(comfy_process, "ComfyUI", pathlib.Path("/tmp/comfy-ui.log"))

    _tail_and_monitor(
        {
            "ComfyUI": (comfy_process, pathlib.Path("/tmp/comfy-ui.log")),
        }
    )


if __name__ == "__main__":
    main()
