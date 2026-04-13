"""Launch vLLM + Open WebUI and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

LOG_FILES = [
    pathlib.Path("/tmp/vllm.log"),
    pathlib.Path("/tmp/openwebui.log"),
    pathlib.Path("/tmp/ngrok.log"),
]


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
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    tp_size = os.environ.get("TP_SIZE", "1")
    env = os.environ.copy()

    _touch_logs()

    vllm_python = os.path.expanduser("~/vllm-venv/bin/python")
    openwebui_bin = os.path.expanduser("~/vllm-venv/bin/open-webui")

    with open("/tmp/vllm.log", "w", encoding="utf-8") as vllm_log:
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
    _ensure_running(vllm_process, "vLLM API server")

    openwebui_env = env.copy()
    openwebui_env["OPENAI_API_BASE_URL"] = "http://127.0.0.1:8000/v1"
    openwebui_env["OPENAI_API_KEY"] = "dummy"
    openwebui_env["WEBUI_AUTH"] = "false"
    with open("/tmp/openwebui.log", "w", encoding="utf-8") as webui_log:
        webui_process = subprocess.Popen(
            [openwebui_bin, "serve", "--host", "0.0.0.0", "--port", "8080"],
            stdout=webui_log,
            stderr=subprocess.STDOUT,
            env=openwebui_env,
        )
    time.sleep(5)
    _ensure_running(webui_process, "Open WebUI")

    _tail_forever()


if __name__ == "__main__":
    main()
