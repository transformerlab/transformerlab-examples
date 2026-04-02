import os
import sys
import argparse
import traceback


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def ensure_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_simplify_stub(hunyuan_root: str):
    """Create missing simplify_mesh_utils if not present."""
    utils_dir = os.path.join(hunyuan_root, "hy3dpaint", "utils")
    os.makedirs(utils_dir, exist_ok=True)

    simplify_file = os.path.join(utils_dir, "simplify_mesh_utils.py")

    if not os.path.exists(simplify_file):
        print("⚠️ Missing simplify_mesh_utils.py → creating stub")
        with open(simplify_file, "w") as f:
            f.write(
                "def remesh_mesh(mesh, *args, **kwargs):\n"
                "    return mesh\n"
            )


def ensure_realesrgan(hunyuan_root: str):
    """Ensure ESRGAN weights exist."""
    ckpt_dir = os.path.join(hunyuan_root, "hy3dpaint", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    model_path = os.path.join(ckpt_dir, "RealESRGAN_x4plus.pth")

    if not os.path.exists(model_path):
        print("⚠️ RealESRGAN model missing. Please download it manually.")
        print("Expected at:", model_path)


def load_pipelines(model_path, shape_subfolder):
    import torch

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import (
        Hunyuan3DPaintPipeline,
        Hunyuan3DPaintConfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Loading shape model on {device}...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=shape_subfolder,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    print("🎨 Loading paint model...")
    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)

    return shape_pipeline, Hunyuan3DPaintPipeline(paint_config)


def generate_3d(image_path, output_dir, model_path, shape_subfolder):
    shape_pipeline, paint_pipeline = load_pipelines(model_path, shape_subfolder)

    print("🧠 Generating 3D shape...")
    try:
        result = shape_pipeline(image=image_path)
        mesh = result[0] if isinstance(result, (list, tuple)) else result
    except Exception:
        print("❌ Shape generation failed")
        traceback.print_exc()
        raise

    mesh_path = os.path.join(output_dir, "mesh.obj")
    mesh.export(mesh_path)
    print(f"✅ Mesh saved: {mesh_path}")

    print("🎨 Generating texture...")
    try:
        textured_path = paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=os.path.join(output_dir, "textured_mesh.obj"),
        )
    except Exception:
        print("❌ Texture generation failed")
        traceback.print_exc()
        raise

    print(f"✅ Final output: {textured_path}")
    return textured_path


def main():
    parser = argparse.ArgumentParser(description="Hunyuan3D Inference")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--shape_subfolder",
        type=str,
        default="hunyuan3d-dit-v2-1",
    )

    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output)
    model_path = args.model_path  # can be HF or local

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # If model_path is local → use it for sys.path
    if os.path.exists(model_path):
        ensure_sys_path(model_path)
        hunyuan_root = model_path
    else:
        # HF mode: assume working dir contains repo
        hunyuan_root = resolve_path("Hunyuan3D-2.1")
        ensure_sys_path(hunyuan_root)

    # Fix missing components
    ensure_simplify_stub(hunyuan_root)
    ensure_realesrgan(hunyuan_root)

    print("📦 Starting Hunyuan3D pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path}")

    try:
        output = generate_3d(
            input_path,
            output_dir,
            model_path,
            args.shape_subfolder,
        )
    except Exception:
        print("❌ Pipeline failed")
        traceback.print_exc()
        sys.exit(1)

    print("🎉 Done!")
    print(f"Saved at: {output}")


if __name__ == "__main__":
    main()
