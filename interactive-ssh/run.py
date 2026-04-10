"""Launch SSH ngrok tunnel and stream logs."""

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


def main() -> None:
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", "")
    if ngrok_auth_token:
        subprocess.run(["ngrok", "config", "add-authtoken", ngrok_auth_token], check=False)

    username = getpass.getuser() or os.path.basename(os.path.expanduser("~"))
    print(f"USER_ID={username}", flush=True)

    with open(LOG_PATH, "w", encoding="utf-8") as ngrok_log:
        subprocess.Popen(["ngrok", "tcp", "22", "--log=stdout"], stdout=ngrok_log, stderr=subprocess.STDOUT)

    _tail_forever(LOG_PATH)


if __name__ == "__main__":
    main()
