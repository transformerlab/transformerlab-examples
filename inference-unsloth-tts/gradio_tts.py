"""Gradio text-to-speech interface using Unsloth TTS models (CSM / Orpheus).

Designed to run on GPU servers via SkyPilot or SLURM.
"""

import os
import sys
import tempfile
import time
import uuid

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from abc import ABC, abstractmethod
from transformers import AutoProcessor, CsmForConditionalGeneration

try:
    from snac import SNAC
except ImportError:
    SNAC = None

from unsloth import FastModel

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "sesame/csm-1b")
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "CsmForConditionalGeneration")
DEFAULT_VOICE = os.environ.get("VOICE", "") or "tara"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[TTS] Model: {MODEL_NAME}")
print(f"[TTS] Architecture: {MODEL_ARCHITECTURE}")
print(f"[TTS] Device: {DEVICE}")
print(f"[TTS] CUDA available: {torch.cuda.is_available()}")


# ---------------------------------------------------------------------------
# Audio model classes (adapted from the plugin's audio.py)
# ---------------------------------------------------------------------------

class AudioModelBase(ABC):
    def __init__(self, model_name, device, context_length=2048):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length

    @abstractmethod
    def tokenize(self, text, audio_path=None, sample_rate=24000, voice=None):
        pass

    @abstractmethod
    def generate(self, inputs, **kwargs):
        pass

    @abstractmethod
    def decode(self, generated, **kwargs):
        pass


class CsmAudioModel(AudioModelBase):
    def __init__(self, model_name, device, processor_name=None, context_length=2048):
        super().__init__(model_name, device, context_length)
        processor_name = processor_name or model_name
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model = self.model.to(self.device)
        self.generate_kwargs = {
            "max_new_tokens": 1024,
            "output_audio": True,
        }

    def tokenize(self, text, audio_path=None, sample_rate=24000, voice=None):
        speaker_id = 0
        if audio_path:
            audio_array, _ = librosa.load(audio_path, sr=sample_rate)
            conversation = [
                {
                    "role": f"{speaker_id}",
                    "content": [
                        {"type": "text", "text": "This is how I sound."},
                        {"type": "audio", "path": audio_array},
                    ],
                },
                {
                    "role": f"{speaker_id}",
                    "content": [{"type": "text", "text": text}],
                },
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            )
            return inputs.to(self.device)
        else:
            return self.processor(f"[{speaker_id}]{text}", add_special_tokens=True).to(self.device)

    def generate(self, inputs, **kwargs):
        gen_args = {**inputs, **self.generate_kwargs, **kwargs}
        return self.model.generate(**gen_args)

    def decode(self, generated, **kwargs):
        audio = generated[0].to(torch.float32).cpu().numpy()
        return audio


class OrpheusAudioModel(AudioModelBase):
    SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"

    START_OF_HEADER = 128259
    END_OF_TEXT = 128009
    END_OF_HEADER = 128260
    SPEECH_DELIMITER = 128261
    START_OF_SPEECH = 128257
    END_OF_SPEECH = 128258
    SPEECH_SEPARATOR = 128262
    PAD_TOKEN = 128263
    CODE_TOKEN_OFFSET = 128266

    def __init__(self, model_name, device, context_length=2048):
        super().__init__(model_name, device, context_length)

        if SNAC is None:
            raise ImportError("snac package is required for Orpheus models. Install with: pip install snac")

        self.snac_model = SNAC.from_pretrained(self.SNAC_MODEL_NAME)
        self.snac_model = self.snac_model.to(self.device)

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model = self.model.to(self.device)

        self.generate_kwargs = {
            "max_new_tokens": 10240,
            "eos_token_id": self.END_OF_SPEECH,
            "use_cache": True,
            "repetition_penalty": 1.1,
        }

    def tokenize(self, text, audio_path=None, sample_rate=None, voice=None):
        prompt = f"{voice}: " + text if voice else text
        text_tokens = self.tokenizer(prompt, return_tensors="pt")
        text_input_ids = text_tokens["input_ids"].to(self.device)

        if audio_path:
            sample_rate = sample_rate or 24000
            audio_array, _ = librosa.load(audio_path, sr=sample_rate)
            audio_tokens = self._encode_audio_to_tokens(audio_array)
            return self._create_voice_cloning_input(text_input_ids, audio_tokens)
        else:
            return text_input_ids

    def generate(self, inputs, **kwargs):
        return self.model.generate(inputs, **self.generate_kwargs, **kwargs)

    def decode(self, generated_ids, **kwargs):
        start_indices = (generated_ids == self.START_OF_SPEECH).nonzero(as_tuple=True)
        if len(start_indices[1]) > 0:
            last_start_idx = start_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_start_idx + 1:]
        else:
            cropped_tensor = generated_ids

        processed_tokens = [row[row != self.END_OF_SPEECH] for row in cropped_tensor]
        row = processed_tokens[0]
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]

        codec_codes = [token.item() - self.CODE_TOKEN_OFFSET for token in trimmed_row]
        return self._decode_to_audio(codec_codes).squeeze().to(torch.float32).cpu().detach().numpy()

    def _decode_to_audio(self, code_list):
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list) + 1) // 7):
            base_idx = 7 * i
            layer_1.append(code_list[base_idx])
            layer_2.append(code_list[base_idx + 1] - 4096)
            layer_2.append(code_list[base_idx + 4] - 4 * 4096)
            layer_3.append(code_list[base_idx + 2] - 2 * 4096)
            layer_3.append(code_list[base_idx + 3] - 3 * 4096)
            layer_3.append(code_list[base_idx + 5] - 5 * 4096)
            layer_3.append(code_list[base_idx + 6] - 6 * 4096)

        codes = [
            torch.tensor(layer_1, device=self.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.device).unsqueeze(0),
        ]
        return self.snac_model.decode(codes)

    def _encode_audio_to_tokens(self, waveform):
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)
        waveform_tensor = waveform_tensor.to(dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform_tensor)

        all_tokens = []
        for i in range(codes[0].shape[1]):
            base_idx = 4 * i
            all_tokens.extend([
                codes[0][0][i].item() + self.CODE_TOKEN_OFFSET,
                codes[1][0][2 * i].item() + self.CODE_TOKEN_OFFSET + 4096,
                codes[2][0][base_idx].item() + self.CODE_TOKEN_OFFSET + 2 * 4096,
                codes[2][0][base_idx + 1].item() + self.CODE_TOKEN_OFFSET + 3 * 4096,
                codes[1][0][2 * i + 1].item() + self.CODE_TOKEN_OFFSET + 4 * 4096,
                codes[2][0][base_idx + 2].item() + self.CODE_TOKEN_OFFSET + 5 * 4096,
                codes[2][0][base_idx + 3].item() + self.CODE_TOKEN_OFFSET + 6 * 4096,
            ])
        return all_tokens

    def _create_voice_cloning_input(
        self, target_text_ids, audio_tokens, voice_prompt="This is the way I want you to sound. "
    ):
        voice_prompt_tokens = self.tokenizer(voice_prompt, return_tensors="pt")["input_ids"].to(self.device)

        header_start = torch.tensor([[self.START_OF_HEADER]], dtype=torch.int64).to(self.device)
        header_end = torch.tensor(
            [[self.END_OF_TEXT, self.END_OF_HEADER, self.SPEECH_DELIMITER, self.START_OF_SPEECH]],
            dtype=torch.int64,
        ).to(self.device)
        voice_end = torch.tensor(
            [[self.END_OF_SPEECH, self.SPEECH_SEPARATOR]], dtype=torch.int64
        ).to(self.device)
        target_end = torch.tensor(
            [[self.END_OF_TEXT, self.END_OF_HEADER, self.SPEECH_DELIMITER]], dtype=torch.int64
        ).to(self.device)

        input_sequence = [
            header_start,
            voice_prompt_tokens,
            header_end,
            torch.tensor([audio_tokens], dtype=torch.int64).to(self.device),
            voice_end,
            header_start,
            target_text_ids,
            target_end,
        ]

        return torch.cat(input_sequence, dim=1).to(self.device)


# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

print(f"[TTS] Loading model {MODEL_NAME} ({MODEL_ARCHITECTURE}) ...")
t_load = time.time()

if MODEL_ARCHITECTURE == "CsmForConditionalGeneration":
    audio_model = CsmAudioModel(model_name=MODEL_NAME, device=DEVICE)
    print(
        "⚠️  RECOMMENDATION: For best results with CsmForConditionalGeneration models, set temperature=0!"
    )
elif "orpheus" in MODEL_NAME.lower():
    audio_model = OrpheusAudioModel(model_name=MODEL_NAME, device=DEVICE)
else:
    print(f"[TTS] ERROR: Unsupported architecture '{MODEL_ARCHITECTURE}' for model '{MODEL_NAME}'")
    sys.exit(1)

print(f"[TTS] Model loaded in {time.time() - t_load:.1f}s")


# ---------------------------------------------------------------------------
# Synthesis function
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000


def synthesize(
    text: str,
    voice: str,
    speed: float,
    temperature: float,
    top_p: float,
    reference_audio,
):
    """Generate speech from text using the loaded model."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesize.")

    voice = (voice or "").strip() or None

    # Build generation kwargs
    generate_kwargs = {}
    if temperature == 0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        if top_p < 1.0:
            generate_kwargs["top_p"] = top_p

    # Handle reference audio for voice cloning
    audio_path = None
    if reference_audio is not None:
        # Gradio returns (sample_rate, numpy_array) or a filepath
        if isinstance(reference_audio, str):
            audio_path = reference_audio
        elif isinstance(reference_audio, tuple):
            sr, audio_data = reference_audio
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_data, sr)
            audio_path = tmp.name

    print(f"[TTS] Generating speech for: {text[:80]}...")
    t0 = time.time()

    try:
        inputs = audio_model.tokenize(
            text=text,
            audio_path=audio_path,
            sample_rate=SAMPLE_RATE,
            voice=voice,
        )
        audio_values = audio_model.generate(inputs, **generate_kwargs)
        audio = audio_model.decode(audio_values)

        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)

        # Write to a temp file
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, f"output_{uuid.uuid4().hex[:8]}.wav")
        sf.write(output_path, audio, SAMPLE_RATE)

        elapsed = time.time() - t0
        print(f"[TTS] Done in {elapsed:.2f}s → {output_path}")

        return output_path
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")
    finally:
        # Cleanup temp reference audio if we created one
        if audio_path and reference_audio is not None and isinstance(reference_audio, tuple):
            try:
                os.unlink(audio_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

ORPHEUS_VOICE_PRESETS = [
    "tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe",
]

is_orpheus = "orpheus" in MODEL_NAME.lower()

with gr.Blocks(title="Unsloth TTS") as demo:
    gr.Markdown(
        f"# 🔊 Unsloth Text-to-Speech\n"
        f"Model: **{MODEL_NAME}** &nbsp;|&nbsp; Architecture: **{MODEL_ARCHITECTURE}** &nbsp;|&nbsp; Device: **{DEVICE}**"
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Type or paste text here…",
                lines=5,
            )
            with gr.Row():
                if is_orpheus:
                    voice_input = gr.Dropdown(
                        label="Voice (Orpheus)",
                        choices=ORPHEUS_VOICE_PRESETS,
                        value=DEFAULT_VOICE,
                        allow_custom_value=True,
                    )
                else:
                    voice_input = gr.Textbox(
                        label="Voice",
                        value="",
                        placeholder="(not used for CSM models)",
                    )

                speed_input = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )
            with gr.Row():
                temperature_input = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.0 if MODEL_ARCHITECTURE == "CsmForConditionalGeneration" else 0.6,
                    step=0.05,
                )
                top_p_input = gr.Slider(
                    label="Top P",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                )

            reference_audio_input = gr.Audio(
                label="Reference Audio (optional, for voice cloning)",
                type="filepath",
            )

            generate_btn = gr.Button("🎙️ Generate Speech", variant="primary")

        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Generated Audio", type="filepath")

    generate_btn.click(
        fn=synthesize,
        inputs=[text_input, voice_input, speed_input, temperature_input, top_p_input, reference_audio_input],
        outputs=audio_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
