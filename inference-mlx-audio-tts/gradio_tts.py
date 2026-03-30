"""Gradio text-to-speech interface backed by MLX Audio CLI."""

import os
import subprocess
import sys
import tempfile
import time

import gradio as gr

# --- Ensure espeak-ng data path is set for all subprocesses ---
if not os.environ.get("ESPEAK_DATA_PATH"):
    brew_prefix = subprocess.run(
        ["brew", "--prefix", "espeak-ng"],
        capture_output=True, text=True
    )
    if brew_prefix.returncode == 0:
        espeak_data = os.path.join(brew_prefix.stdout.strip(), "share", "espeak-ng-data")
        if os.path.isdir(espeak_data):
            os.environ["ESPEAK_DATA_PATH"] = espeak_data
            print(f"[TTS] Set ESPEAK_DATA_PATH={espeak_data}")

MODEL_NAME = os.environ.get("MODEL_NAME", "mlx-community/Kokoro-82M-bf16")
DEFAULT_VOICE = os.environ.get("VOICE", "") or "af_heart"
DEFAULT_LANG_CODE = os.environ.get("LANG_CODE", "") or "a"

VALID_LANG_CODES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Portuguese",
    "z": "Mandarin Chinese",
}


def synthesize(text: str, voice: str, speed: float, lang_code: str):
    """Shell out to mlx_audio.tts.generate and return the output WAV path."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    voice = (voice or "").strip() or DEFAULT_VOICE
    lang_code = (lang_code or "").strip() or DEFAULT_LANG_CODE

    if lang_code not in VALID_LANG_CODES:
        raise gr.Error(
            f"Invalid language code '{lang_code}'. "
            f"Choose one of: " + ", ".join(f"{k}={v}" for k, v in VALID_LANG_CODES.items())
        )

    # Output audio next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    # --- Resolve venv Python ---
    parent_dir = os.path.dirname(script_dir)
    venv_python = os.path.join(parent_dir, "venv", "bin", "python")
    python_executable = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [
        python_executable, "-m", "mlx_audio.tts.generate",
        "--model", MODEL_NAME,
        "--text", text,
        "--voice", voice,
        "--speed", str(speed),
        "--lang_code", lang_code,
        "--file_prefix", "output",
    ]

    # Build env with ESPEAK_DATA_PATH guaranteed
    env = os.environ.copy()

    print(f"[TTS] Running: {' '.join(cmd)}")
    print(f"[TTS] ESPEAK_DATA_PATH={env.get('ESPEAK_DATA_PATH', '(not set)')}")
    print(f"[TTS] Output directory: {output_dir}")
    t0 = time.time()

    # Run from the output directory so files are created there
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=output_dir)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise gr.Error(f"Generation failed:\n{result.stderr or result.stdout}")

    # Add a small delay to ensure file is written
    time.sleep(0.5)

    wav_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".wav")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    
    print(f"[TTS] WAV files found: {wav_files}")
    
    if not wav_files:
        raise gr.Error("No audio file was generated. Check the model and settings.")

    output_path = os.path.join(output_dir, wav_files[0])
    elapsed = time.time() - t0
    print(f"[TTS] Done in {elapsed:.2f}s → {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

VOICE_PRESETS = [
    "af_heart", "af_bella", "af_nova", "af_sky",
    "am_adam", "am_echo",
    "bf_alice", "bf_emma",
    "bm_daniel", "bm_george",
]

with gr.Blocks(title="MLX Audio TTS") as demo:
    gr.Markdown(
        f"# 🔊 MLX Audio — Text-to-Speech\n"
        f"Model: **{MODEL_NAME}**"
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Type or paste text here…",
                lines=5,
            )
            with gr.Row():
                voice_input = gr.Dropdown(
                    label="Voice",
                    choices=VOICE_PRESETS,
                    value=DEFAULT_VOICE,
                    allow_custom_value=True,
                )
                lang_input = gr.Dropdown(
                    label="Language",
                    choices=[(v, k) for k, v in VALID_LANG_CODES.items()],
                    value=DEFAULT_LANG_CODE,
                    allow_custom_value=True,
                )
                speed_input = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )
            generate_btn = gr.Button("🎙️ Generate Speech", variant="primary")

        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Generated Audio", type="filepath")

    generate_btn.click(
        fn=synthesize,
        inputs=[text_input, voice_input, speed_input, lang_input],
        outputs=audio_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)