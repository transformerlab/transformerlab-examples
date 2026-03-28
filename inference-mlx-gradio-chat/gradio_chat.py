"""Gradio chat interface backed by a local MLX LM server."""

import os
import subprocess
import sys
import time

import gradio as gr
from openai import OpenAI

MODEL_NAME = os.environ.get("MODEL_NAME", "mlx-community/Llama-3.2-3B-Instruct-4bit")
MLX_HOST = "127.0.0.1"
MLX_PORT = 8001

client = OpenAI(base_url=f"http://{MLX_HOST}:{MLX_PORT}/v1", api_key="mlx")


def start_mlx_server():
    """Start the MLX LM server in a background thread."""
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", MODEL_NAME,
        "--host", MLX_HOST,
        "--port", str(MLX_PORT),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return process


def wait_for_server(host: str, port: int, timeout: int = 120):
    """Wait until the MLX LM server is accepting connections."""
    import socket

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def chat(message: str, history: list[dict]) -> str:
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return response.choices[0].message.content


# Start MLX LM server in background
mlx_process = start_mlx_server()

print(f"Waiting for MLX LM server to start on {MLX_HOST}:{MLX_PORT}...")
if not wait_for_server(MLX_HOST, MLX_PORT):
    print("ERROR: MLX LM server did not start within the timeout period.", file=sys.stderr)
    sys.exit(1)
print(f"MLX LM server is ready on {MLX_HOST}:{MLX_PORT}")


demo = gr.ChatInterface(
    fn=chat,
    title="MLX LM Chat",
    description=f"Chatting with **{MODEL_NAME}** via MLX LM",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
