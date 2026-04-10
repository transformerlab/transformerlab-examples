"""Launch Ollama + Open WebUI + ngrok and stream logs."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

LOG_FILES = [
    pathlib.Path("/tmp/ollama.log"),
    pathlib.Path("/tmp/ollama-pull.log"),
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
    model_name = os.environ.get("MODEL_NAME", "llama2")
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", "")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"

    _touch_logs()
    if ngrok_auth_token:
        subprocess.run(["ngrok", "config", "add-authtoken", ngrok_auth_token], check=True)

    openwebui_bin = os.path.expanduser("~/ollama-venv/bin/open-webui")
    ngrok_config = os.path.expanduser("~/ngrok-ollama.yml")

    with open("/tmp/ollama.log", "w", encoding="utf-8") as ollama_log:
        ollama_process = subprocess.Popen(["ollama", "serve"], stdout=ollama_log, stderr=subprocess.STDOUT, env=env)
    time.sleep(3)
    _ensure_running(ollama_process, "ollama serve")

    with open("/tmp/ollama-pull.log", "w", encoding="utf-8") as pull_log:
        pull_process = subprocess.Popen(["ollama", "pull", model_name], stdout=pull_log, stderr=subprocess.STDOUT, env=env)
    time.sleep(5)
    _ensure_running(pull_process, "ollama pull")

    webui_env = env.copy()
    webui_env["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    webui_env["WEBUI_AUTH"] = "false"
    with open("/tmp/openwebui.log", "w", encoding="utf-8") as webui_log:
        webui_process = subprocess.Popen(
            [openwebui_bin, "serve", "--host", "0.0.0.0", "--port", "8080"],
            stdout=webui_log,
            stderr=subprocess.STDOUT,
            env=webui_env,
        )
    time.sleep(5)
    _ensure_running(webui_process, "Open WebUI")

    pathlib.Path(ngrok_config).write_text(
        "\n".join(
            [
                "version: 2",
                f"authtoken: {ngrok_auth_token}",
                "tunnels:",
                "  ollama:",
                "    proto: http",
                "    addr: 11434",
                "  openwebui:",
                "    proto: http",
                "    addr: 8080",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with open("/tmp/ngrok.log", "w", encoding="utf-8") as ngrok_log:
        ngrok_process = subprocess.Popen(
            ["ngrok", "start", "--all", "--config", ngrok_config, "--log=stdout"],
            stdout=ngrok_log,
            stderr=subprocess.STDOUT,
        )
    time.sleep(1)
    _ensure_running(ngrok_process, "ngrok tunnels")

    _tail_forever()


if __name__ == "__main__":
    main()
