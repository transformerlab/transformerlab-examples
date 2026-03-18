"""Gradio text-to-speech interface backed by MLX Audio."""

import io
import os
import sys
import tempfile
import time

import gradio as gr
import numpy as np

MODEL_NAME = os.environ.get("MODEL_NAME", "mlx-community/Kokoro-82M-bf16")
DEFAULT_VOICE = os.environ.get("VOICE", "af_heart")
DEFAULT_LANG_CODE = os.environ.get("LANG_CODE", "a")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading TTS model: {MODEL_NAME} ...")
start = time.time()

from mlx_audio.tts.utils import load_model

model = load_model(MODEL_NAME)
print(f"Model loaded in {time.time() - start:.1f}s")

# Resolve sample rate from the model (most MLX-Audio models expose this).
SAMPLE_RATE = getattr(model, "sample_rate", 24000)


# ---------------------------------------------------------------------------
# TTS generation helper
# ---------------------------------------------------------------------------

def synthesize(text: str, voice: str, speed: float, lang_code: str):
    """Generate speech and return a (sample_rate, numpy_array) tuple for Gradio."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    voice = voice.strip() if voice and voice.strip() else DEFAULT_VOICE
    lang_code = lang_code.strip() if lang_code and lang_code.strip() else DEFAULT_LANG_CODE

    gen_kwargs = dict(
        text=text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        verbose=True,
    )

    print(f"[TTS] Generating: text={text!r}, voice={voice}, speed={speed}, lang={lang_code}")
    t0 = time.time()

    audio_segments = []
    for result in model.generate(**gen_kwargs):
        # result.audio is an mx.array – convert to numpy
        audio_np = np.array(result.audio, copy=False)
        audio_segments.append(audio_np)

    if not audio_segments:
        raise gr.Error("No audio was generated. Please try different text or settings.")

    audio = np.concatenate(audio_segments, axis=0)
    elapsed = time.time() - t0
    duration = len(audio) / SAMPLE_RATE
    print(f"[TTS] Done in {elapsed:.2f}s — {duration:.2f}s of audio generated")

    # Gradio gr.Audio expects (sample_rate, numpy_array)
    return (SAMPLE_RATE, audio)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

VOICE_EXAMPLES = [
    "af_heart", "af_bella", "af_nova", "af_sky",
    "am_adam", "am_echo",
    "bf_alice", "bf_emma",
    "bm_daniel", "bm_george",
]

LANG_CODES = {
    "a": "American English",
    "b": "British English",
    "j": "Japanese",
    "z": "Mandarin Chinese",
    "e": "Spanish",
    "f": "French",
}

with gr.Blocks(title="MLX Audio TTS", theme=gr.themes.Soft()) as demo:
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
                    choices=VOICE_EXAMPLES,
                    value=DEFAULT_VOICE,
                    allow_custom_value=True,
                )
                lang_input = gr.Dropdown(
                    label="Language",
                    choices=[(v, k) for k, v in LANG_CODES.items()],
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
            audio_output = gr.Audio(label="Generated Audio", type="numpy")

    generate_btn.click(
        fn=synthesize,
        inputs=[text_input, voice_input, speed_input, lang_input],
        outputs=audio_output,
    )

    gr.Examples(
        examples=[
            ["Hello! Welcome to Transformer Lab. This is a text-to-speech demo powered by MLX Audio on Apple Silicon.", DEFAULT_VOICE, 1.0, DEFAULT_LANG_CODE],
            ["The quick brown fox jumps over the lazy dog.", "af_bella", 1.0, "a"],
            ["Science is not only a disciple of reason but also one of romance and passion.", "am_adam", 0.9, "a"],
        ],
        inputs=[text_input, voice_input, speed_input, lang_input],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
