"""Gradio 3D Model Viewer - Interactive visualization of 3D models."""

import gradio as gr
import os


def view_3d_model(model_file):
    if model_file is None:
        return None
    return model_file


demo = gr.Interface(
    fn=view_3d_model,
    inputs=gr.Model3D(label="3D Model", file_types=[".obj", ".glb", ".gltf"]),
    outputs=gr.Model3D(label="Interactive 3D Viewer", interactive=False),
    title="3D Model Viewer",
    description="Upload a 3D model (.obj, .glb, .gltf) to visualize it interactively",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
