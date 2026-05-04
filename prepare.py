#!/usr/bin/env python3
"""
Top-level wrapper so tooling that expects `prepare.py` at repo root
can execute the autoresearch.prepare module which lives in the
`autoresearch` package.

Run as:
  python prepare.py [args]

This forwards execution to `autoresearch.prepare` using runpy so the
module's `if __name__ == "__main__"` block runs as expected.
"""

import runpy
import sys

if __name__ == "__main__":
    # Preserve argv when running the module as a script
    runpy.run_module("autoresearch.prepare", run_name="__main__", alter_sys=True)
