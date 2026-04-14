"""Launch SSH tunnel and stream logs."""

from __future__ import annotations

import getpass
import os
import pathlib
import subprocess
import time

LOG_PATH = pathlib.Path("/tmp/ngrok.log")


def _tail_forever(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    offset = 0
    while True:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
        if chunk:
            for line in chunk.splitlines():
                print(f"[{path.name}] {line}", flush=True)
        else:
            time.sleep(0.25)


def _ensure_running(process: subprocess.Popen[bytes], name: str) -> None:
    code = process.poll()
    if code is not None:
        raise RuntimeError(f"{name} exited early with code {code}")


def main() -> None:
    username = getpass.getuser() or os.path.basename(os.path.expanduser("~"))
    print(f"USER_ID={username}", flush=True)

    _tail_forever(LOG_PATH)


if __name__ == "__main__":
    main()
