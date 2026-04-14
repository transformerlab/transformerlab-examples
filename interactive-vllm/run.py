"""Launch vLLM + Open WebUI and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import urllib.error

LOG_FILES = [
    pathlib.Path("/tmp/vllm.log"),
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
    except Exception as e:
        print(f"Error reading log file {path.name}: {e}", file=sys.stderr)



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


HEALTH_CHECK_URL = "http://localhost:8080/health"
HEALTH_CHECK_INTERVAL = 10
OPENWEBUI_LOG = pathlib.Path("/tmp/openwebui.log")


def _check_openwebui_health(last_health_check: float, health_logged: bool) -> tuple[float, bool]:
    now = time.monotonic()
    if health_logged or now - last_health_check < HEALTH_CHECK_INTERVAL:
        return last_health_check, health_logged
    try:
        req = urllib.request.Request(HEALTH_CHECK_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                msg = "Local OpenWebUI URL: http://localhost:8080\n"
                with OPENWEBUI_LOG.open("a", encoding="utf-8") as f:
                    f.write(msg)
                return now, True
    except (urllib.error.URLError, OSError):
        pass
    return now, False


def _tail_and_monitor(procs: dict[str, tuple[subprocess.Popen, pathlib.Path]]) -> None:
    offsets = {path: 0 for path in LOG_FILES}
    last_check = time.monotonic()
    last_health_check = time.monotonic()
    health_logged = False

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

        last_health_check, health_logged = _check_openwebui_health(last_health_check, health_logged)

        if not saw_data:
            time.sleep(0.25)


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    tp_size = os.environ.get("TP_SIZE", "1")
    env = os.environ.copy()

    _touch_logs()

    vllm_python = os.path.expanduser("~/vllm-venv/bin/python")
    openwebui_bin = os.path.expanduser("~/vllm-venv/bin/open-webui")

    vllm_log = open("/tmp/vllm.log", "w", encoding="utf-8")
    vllm_process = subprocess.Popen(
        [
            vllm_python,
            "-u",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name,
            "--tensor-parallel-size",
            str(tp_size),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--gpu-memory-utilization",
            "0.9",
        ],
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    time.sleep(10)
    _check_proc(vllm_process, "vLLM API server", pathlib.Path("/tmp/vllm.log"))

    openwebui_env = env.copy()
    openwebui_env["OPENAI_API_BASE_URL"] = "http://127.0.0.1:8000/v1"
    openwebui_env["OPENAI_API_KEY"] = "dummy"
    openwebui_env["WEBUI_AUTH"] = "false"
    webui_log = open("/tmp/openwebui.log", "w", encoding="utf-8")
    webui_process = subprocess.Popen(
        [openwebui_bin, "serve", "--host", "0.0.0.0", "--port", "8080"],
        stdout=webui_log,
        stderr=subprocess.STDOUT,
        env=openwebui_env,
    )
    time.sleep(5)
    _check_proc(webui_process, "Open WebUI", pathlib.Path("/tmp/openwebui.log"))

    _tail_and_monitor(
        {
            "vLLM": (vllm_process, pathlib.Path("/tmp/vllm.log")),
            "Open WebUI": (webui_process, pathlib.Path("/tmp/openwebui.log")),
        }
    )


if __name__ == "__main__":
    main()
