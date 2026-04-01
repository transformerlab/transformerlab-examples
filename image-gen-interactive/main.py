import os
import sys
import gradio as gr
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. Please set it in app settings under secrets.")

IMAGE_MODELS = {
    "SDXL 1.0 (1024x1024)": ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl", 1024),
    "SD 2.1 (768x768)": ("stabilityai/stable-diffusion-2-1", "sd", 768),
    "SD 1.5 (512x512)": ("runwayml/stable-diffusion-v1-5", "sd", 512),
    "Flux 1.0 Dev (1024x1024)": ("black-forest-labs/FLUX.1-dev", "flux", 1024),
    "SegMoE v2.2 Schnell (1024x1024)": ("segmind/segmoe_02_schnell", "sdxl", 1024),
}

VIDEO_MODELS = {
    "ModelScope 1.7B (256x256)": ("damo-vilab/text-to-video-ms-1.7b", 256),
    "ZeroScope v2 (576x320)": ("cerspense/zeroscope_v2_576w", 576),
    "ModelScope 14B (256x256)": ("damo-vilab/text-to-video-synthesis-14b", 256),
}

pipeline_cache = {}
pipeline_type_cache = {}


def get_pipeline(model_name, model_type):
    cache_key = f"{model_name}_{model_type}"
    if cache_key in pipeline_cache:
        return pipeline_cache[cache_key]

    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
    import torch

    print(f"Loading model: {model_name}...")

    try:
        if model_type == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=HF_TOKEN,
            )
        elif model_type == "flux":
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=HF_TOKEN,
            )
        elif model_type == "video":
            pipeline = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=HF_TOKEN,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=HF_TOKEN,
            )

        pipeline.to("cuda")
        if model_type == "video":
            pipeline.enable_model_cpu_offload()

        pipeline_cache[cache_key] = pipeline
        pipeline_type_cache[cache_key] = model_type
        print("Model loaded!")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def generate_image(model_name, prompt, negative_prompt, width, height, steps, guidance):
    if not HF_TOKEN:
        return None, "Error: HF_TOKEN not set. Please set it in app settings."

    model_info = IMAGE_MODELS.get(model_name)
    if not model_info:
        return None, "Invalid model selected"

    model_id, _, default_size = model_info

    if not width:
        width = default_size
    if not height:
        height = default_size

    pipeline = get_pipeline(model_id, "image")
    if not pipeline:
        return None, "Failed to load pipeline"

    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        return result.images[0], None
    except Exception as e:
        return None, f"Error generating image: {str(e)}"


def generate_video(model_name, prompt, num_frames, width, height, steps, guidance):
    if not HF_TOKEN:
        return None, "Error: HF_TOKEN not set. Please set it in app settings."

    model_info = VIDEO_MODELS.get(model_name)
    if not model_info:
        return None, "Invalid model selected"

    model_id, default_size = model_info

    if not width:
        width = default_size
    if not height:
        height = default_size // 2

    pipeline = get_pipeline(model_id, "video")
    if not pipeline:
        return None, "Failed to load pipeline"

    try:
        result = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        frames = result.frames[0]

        output_dir = Path.home() / "generated_videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = list(output_dir.glob("generated_*.mp4"))
        num = len(existing) + 1
        filepath = output_dir / f"generated_{num:04d}.mp4"

        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(list(frames), fps=8)
        clip.write_videofile(
            str(filepath), codec="libx264", audio=False, verbose=False, logger=None
        )

        return filepath, None
    except Exception as e:
        return None, f"Error generating video: {str(e)}"


def image_tab():
    with gr.Column():
        gr.Markdown("### Image Generation")
        model_dropdown = gr.Dropdown(
            choices=list(IMAGE_MODELS.keys()),
            label="Model",
            value="SDXL 1.0 (1024x1024)",
        )
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt...")
        neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="What to avoid...")

        with gr.Row():
            width = gr.Number(label="Width", value=1024)
            height = gr.Number(label="Height", value=1024)
        with gr.Row():
            steps = gr.Slider(
                minimum=1, maximum=100, value=25, step=1, label="Inference Steps"
            )
            guidance = gr.Slider(
                minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale"
            )

        generate_btn = gr.Button("Generate Image", variant="primary")
        output = gr.Image(label="Generated Image")
        error_msg = gr.Textbox(label="Error", visible=False)

        generate_btn.click(
            generate_image,
            inputs=[model_dropdown, prompt, neg_prompt, width, height, steps, guidance],
            outputs=[output, error_msg],
        )


def video_tab():
    with gr.Column():
        gr.Markdown("### Video Generation")
        model_dropdown = gr.Dropdown(
            choices=list(VIDEO_MODELS.keys()),
            label="Model",
            value="ZeroScope v2 (576x320)",
        )
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt...")

        with gr.Row():
            width = gr.Number(label="Width", value=576)
            height = gr.Number(label="Height", value=320)
        with gr.Row():
            num_frames = gr.Slider(
                minimum=8, maximum=64, value=16, step=1, label="Frames"
            )
            steps = gr.Slider(
                minimum=1, maximum=100, value=25, step=1, label="Inference Steps"
            )
            guidance = gr.Slider(
                minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale"
            )

        generate_btn = gr.Button("Generate Video", variant="primary")
        output = gr.Video(label="Generated Video")
        error_msg = gr.Textbox(label="Error", visible=False)

        generate_btn.click(
            generate_video,
            inputs=[model_dropdown, prompt, num_frames, width, height, steps, guidance],
            outputs=[output, error_msg],
        )


def main():
    with gr.Blocks(title="Image & Video Generation") as demo:
        gr.Markdown("# 🎨 Interactive Image & Video Generation")
        gr.Markdown("Select a tab below to generate images or videos.")

        with gr.Tab("Image"):
            image_tab()

        with gr.Tab("Video"):
            video_tab()

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
