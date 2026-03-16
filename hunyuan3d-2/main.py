import argparse
import os
import sys
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "Hunyuan3D-2.1", "hy3dshape")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "Hunyuan3D-2.1", "hy3dpaint")
)

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def generate_3d_from_image(image_path, output_dir, model_path, low_vram_mode=False):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading shape model from {model_path}...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    print("Generating 3D shape from image...")
    mesh_untextured = shape_pipeline(image=image_path)[0]

    mesh_path = os.path.join(output_dir, "mesh.obj")
    mesh_untextured.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")

    print("Loading texture model...")
    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

    print("Generating textures...")
    mesh_textured = paint_pipeline(mesh_path, image_path=image_path)

    output_path = os.path.join(output_dir, "textured_mesh.obj")
    mesh_textured.export(output_path)
    print(f"Saved textured 3D model to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 3D Generation")
    parser.add_argument(
        "--mode",
        type=str,
        default="image23d",
        choices=["image2text", "image2text", "image2text", "text2text"],
        help="Generation mode",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input image path or text prompt"
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
        "--low_vram_mode", action="store_true", help="Enable low VRAM mode"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input) and args.mode == "image2text":
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Running Hunyuan3D-2.1 in {args.mode} mode")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    if args.mode in ["image2text", "image2text", "image2text"]:
        output_path = generate_3d_from_image(
            args.input, args.output, args.model_path, args.low_vram_mode
        )
    else:
        print("Text-to-3D mode not yet implemented in this script")
        sys.exit(1)

    print(f"Done! 3D model saved to: {output_path}")


if __name__ == "__main__":
    main()
