"""Gradio text-to-speech interface backed by MLX Audio CLI."""

import os
import subprocess
import sys
import tempfile
import time

import gradio as gr

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


# ---------------------------------------------------------------------------
# TTS generation helper
# ---------------------------------------------------------------------------

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

    output_dir = tempfile.mkdtemp()

    cmd = [
        sys.executable, "-m", "mlx_audio.tts.generate",
        "--model", MODEL_NAME,
        "--text", text,
        "--voice", voice,
        "--speed", str(speed),
        "--lang_code", lang_code,
        "--output_path", output_dir,
        "--file_prefix", "output",
    ]

    print(f"[TTS] Running: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise gr.Error(f"Generation failed:\n{result.stderr or result.stdout}")

    # Find the generated file — mlx_audio writes output_<timestamp>.wav
    wav_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".wav")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
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
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


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

    voice = (voice or "").strip() or DEFAULT_VOICE
    lang_code = (lang_code or "").strip() or DEFAULT_LANG_CODE

    if lang_code not in VALID_LANG_CODES:
        raise gr.Error(
            f"Invalid language code '{lang_code}'. "
            f"Choose one of: a=American English, b=British English, "
            f"e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Mandarin Chinese."
        )

    gen_kwargs = dict(
        text=text,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        verbose=True,
    )

    print(f"[TTS] Generating: text={text!r}, voice={voice}, speed={speed}, lang={lang_code}")
    t0 = time.time()

    try:
        audio_segments = []
        for result in model.generate(**gen_kwargs):
            # result.audio is an mx.array – convert to numpy
            audio_np = np.array(result.audio, copy=False)
            audio_segments.append(audio_np)
    except AssertionError as e:
        raise gr.Error(
            f"Generation failed — invalid voice or language code. "
            f"Details: {e}"
        ) from e
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}") from e

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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
