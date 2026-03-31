import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_ROOT = os.path.join(SCRIPT_DIR, "Hunyuan3D-2.1")
HY3D_SHAPE_PATH = os.path.join(HUNYUAN_ROOT, "hy3dshape")
HY3D_PAINT_PATH = os.path.join(HUNYUAN_ROOT, "hy3dpaint")
DEFAULT_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-1"
# Default image to use when no --input is supplied
DEFAULT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "dog.png")
IMAGE_TO_3D_MODES = {"image2text", "image23d", "image2shape", "image-to-3d"}

sys.path.insert(0, HY3D_SHAPE_PATH)
sys.path.insert(0, HY3D_PAINT_PATH)

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def _require_hunyuan_checkout():
    if not os.path.isdir(HUNYUAN_ROOT):
        raise FileNotFoundError(
            f"Hunyuan checkout not found at {HUNYUAN_ROOT}. Run the task setup first."
        )


def _build_paint_config(output_dir):
    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)

    # The upstream paint pipeline expects these paths relative to the Hunyuan repo
    # root, but this example runs from a wrapper directory.
    paint_config.multiview_cfg_path = os.path.join(
        HUNYUAN_ROOT, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
    )
    paint_config.realesrgan_ckpt_path = os.path.join(
        HUNYUAN_ROOT, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"
    )
    paint_config.custom_pipeline = os.path.join(
        HUNYUAN_ROOT, "hy3dpaint", "hunyuanpaintpbr"
    )

    return paint_config


def generate_3d_from_image(
    image_path,
    output_dir,
    model_path,
    low_vram_mode=False,
    shape_subfolder=DEFAULT_SHAPE_SUBFOLDER,
):
    del low_vram_mode

    _require_hunyuan_checkout()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading shape model from {model_path}...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, subfolder=shape_subfolder
    )

    print("Generating 3D shape from image...")
    mesh_untextured = shape_pipeline(image=image_path)[0]

    mesh_path = os.path.join(output_dir, "mesh.obj")
    mesh_untextured.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")

    print("Loading texture model...")
    paint_config = _build_paint_config(output_dir)
    paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

    print("Generating textures...")
    output_path = paint_pipeline(
        mesh_path=mesh_path,
        image_path=image_path,
        output_mesh_path=os.path.join(output_dir, "textured_mesh.obj"),
    )

    print(f"Saved textured 3D model to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 3D Generation")
    parser.add_argument(
        "--mode",
        type=str,
        default="image2text",
        choices=sorted(IMAGE_TO_3D_MODES | {"text2text"}),
        help="Generation mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Input image path or text prompt (defaults to the bundled dog.png)",
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="tencent/Hunyuan3D-2.1",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--shape_subfolder",
        type=str,
        default=DEFAULT_SHAPE_SUBFOLDER,
        help="Shape model subfolder inside the HuggingFace repository",
    )
    parser.add_argument(
        "--low_vram_mode", action="store_true", help="Enable low VRAM mode"
    )

    args = parser.parse_args()

    # Normalize input: if a templated placeholder or empty value is passed,
    # fall back to the bundled default image.
    input_path = args.input
    if not input_path or (isinstance(input_path, str) and input_path.strip() == ""):
        input_path = DEFAULT_IMAGE_PATH

    # Some runners inject templated placeholders like "{{input}}"; treat those
    # as "no input provided" and fall back to the default image.
    if (
        isinstance(input_path, str)
        and input_path.startswith("{{")
        and input_path.endswith("}}")
    ):
        print(
            f"Warning: detected placeholder input '{input_path}', using default image {DEFAULT_IMAGE_PATH}"
        )
        input_path = DEFAULT_IMAGE_PATH

    if args.mode in IMAGE_TO_3D_MODES and not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Running Hunyuan3D-2.1 in {args.mode} mode")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")

    if args.mode in IMAGE_TO_3D_MODES:
        output_path = generate_3d_from_image(
            input_path,
            args.output,
            args.model_path,
            args.low_vram_mode,
            args.shape_subfolder,
        )
    else:
        print("Text-to-3D mode not yet implemented in this script")
        sys.exit(1)

    print(f"Done! 3D model saved to: {output_path}")


if __name__ == "__main__":
    main()
