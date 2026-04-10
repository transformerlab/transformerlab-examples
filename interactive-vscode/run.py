"""Launch VS Code tunnel."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    process = subprocess.Popen(
        ["code", "tunnel", "--accept-server-license-terms", "--disable-telemetry"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
