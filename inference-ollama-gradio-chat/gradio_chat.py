"""Gradio chat interface backed by a local Ollama server."""

import os

import gradio as gr
import requests
from openai import OpenAI
from sshtunnel import SSHTunnelForwarder

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama2")

# SSH tunneling setup if remote Ollama
SSH_HOST = os.environ.get("SSH_HOST")
SSH_USER = os.environ.get("SSH_USER")
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH")
SSH_PORT = int(os.environ.get("SSH_PORT", 22))

tunnel = None
if SSH_HOST:
    tunnel = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_pkey=SSH_KEY_PATH,
        remote_bind_address=("127.0.0.1", 11434),
        local_bind_address=("127.0.0.1", 0),  # 0 means auto-assign port
    )
    tunnel.start()
    local_port = tunnel.local_bind_port
    OLLAMA_BASE_URL = f"http://127.0.0.1:{local_port}"

client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")


def chat(message: str, history: list[dict]) -> str:
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return response.choices[0].message.content or ""


def tokenize_text(text: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/tokenize",
            json={"model": MODEL_NAME, "prompt": text},
        )
        if response.status_code == 200:
            tokens = response.json()["tokens"]
            return f"Tokens: {tokens}\n\nCount: {len(tokens)}"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


chat_interface = gr.ChatInterface(
    fn=chat,
    title="Ollama Chat",
    description=f"Chatting with **{MODEL_NAME}** via Ollama",
)

tokenization_interface = gr.Interface(
    fn=tokenize_text,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter text to tokenize"),
    outputs=gr.Textbox(label="Tokenization Result"),
    title="Tokenization Preview",
    description=f"Preview how **{MODEL_NAME}** tokenizes text via Ollama",
)

demo = gr.TabbedInterface(
    [chat_interface, tokenization_interface], ["Chat", "Tokenization"]
)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        if tunnel:
            tunnel.close()
