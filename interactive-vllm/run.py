"""Launch vLLM + Open WebUI + ngrok and stream logs."""

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


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    tp_size = os.environ.get("TP_SIZE", "1")
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", "")
    env = os.environ.copy()

    _touch_logs()
    if ngrok_auth_token:
        subprocess.run(["ngrok", "config", "add-authtoken", ngrok_auth_token], check=False)

    vllm_python = os.path.expanduser("~/vllm-venv/bin/python")
    openwebui_bin = os.path.expanduser("~/vllm-venv/bin/open-webui")
    ngrok_config = os.path.expanduser("~/ngrok-vllm.yml")

    with open("/tmp/vllm.log", "w", encoding="utf-8") as vllm_log:
        subprocess.Popen(
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

    openwebui_env = env.copy()
    openwebui_env["OPENAI_API_BASE_URL"] = "http://127.0.0.1:8000/v1"
    openwebui_env["OPENAI_API_KEY"] = "dummy"
    openwebui_env["WEBUI_AUTH"] = "false"
    with open("/tmp/openwebui.log", "w", encoding="utf-8") as webui_log:
        subprocess.Popen(
            [openwebui_bin, "serve", "--host", "0.0.0.0", "--port", "8080"],
            stdout=webui_log,
            stderr=subprocess.STDOUT,
            env=openwebui_env,
        )
    time.sleep(5)

    pathlib.Path(ngrok_config).write_text(
        "\n".join(
            [
                "version: 2",
                f"authtoken: {ngrok_auth_token}",
                "tunnels:",
                "  vllm:",
                "    proto: http",
                "    addr: 8000",
                "  openwebui:",
                "    proto: http",
                "    addr: 8080",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with open("/tmp/ngrok.log", "w", encoding="utf-8") as ngrok_log:
        subprocess.Popen(
            ["ngrok", "start", "--all", "--config", ngrok_config, "--log=stdout"],
            stdout=ngrok_log,
            stderr=subprocess.STDOUT,
        )

    _tail_forever()


if __name__ == "__main__":
    main()
