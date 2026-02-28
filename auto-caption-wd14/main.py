#!/usr/bin/env python3
"""
Auto-tagging task using Waifu Diffusion v1-4 SwinV2 Tagger with TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Loading and using the WD tagger model for image auto-tagging
- Processing batches of images and saving tag results as artifacts
- Progress tracking and logging with lab SDK
"""

import os
import json
import csv
import numpy as np
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf
from torchvision import datasets

from lab import lab
from huggingface_hub import from_pretrained_keras, hf_hub_download

def preprocess_image(image_path, target_size=(448, 448)):
    """Preprocess image for WD tagger model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def annotate_image_with_tags(image_path, tags, output_path, max_tags=5):
    """Annotate image with top tags."""
    image = Image.open(image_path).convert("RGB")
    # Resize for better viewing
    image = image.resize((448, 448), Image.Resampling.LANCZOS)
    # Apply unsharp mask to reduce blurriness
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Larger font for better readability
    except OSError:
        font = ImageFont.load_default()
    
    y_offset = 10
    for tag_info in tags[:max_tags]:
        tag = tag_info["tag"]
        conf = tag_info["confidence"]
        index = tag_info["index"]
        text = f"{index}: {tag}: {conf:.2f}"
        draw.text((10, y_offset), text, fill="white", font=font, stroke_fill="black", stroke_width=2)
        y_offset += 25  # Adjust spacing for larger font
    
    image.save(output_path)

def main():
    """Main function for auto-tagging images with WD v1-4."""
    try:
        # Initialize lab
        lab.init()
        config = lab.get_config()

        # Get configuration parameters with defaults
        output_dir = config.get("output_dir", "./tagged_images")
        threshold = float(config.get("threshold", 0.35))
        model_name = config.get("model_name", "SmilingWolf/wd-v1-4-swinv2-tagger-v2")
        dataset_name = config.get("dataset_name", "cifar10")
        num_images = int(config.get("num_images", 10))
        split = config.get("split", "train")

        # Load dataset using Hugging Face datasets library
        lab.log(f"Loading dataset: {dataset_name} (split: {split})")
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            
            # Dynamically find the image column (check common names)
            possible_image_columns = ["image", "img", "picture"]
            image_column = None
            for col in possible_image_columns:
                if col in dataset.column_names:
                    image_column = col
                    break
            if image_column is None:
                raise ValueError(f"Dataset '{dataset_name}' does not have a recognized image column ({possible_image_columns}). Please choose a dataset with image data.")
            
            lab.log(f"✅ Loaded dataset with {len(dataset)} examples (using '{image_column}' column for images)")
        except Exception as e:
            lab.log(f"❌ Failed to load dataset '{dataset_name}': {e}")
            raise RuntimeError(f"Could not load dataset '{dataset_name}'. Ensure it's a valid Hugging Face dataset name and 'datasets' library is installed.")

        # Select and save images to temporary files
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        actual_num_images = min(num_images, len(dataset))
        for i in range(actual_num_images):
            img = dataset[i][image_column]  # Use the detected image column
            img_path = os.path.join(temp_dir, f"image_{i}.png")
            img.save(img_path)
            image_paths.append(img_path)
        lab.log(f"✅ Prepared {actual_num_images} images from {dataset_name}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Log start
        start_time = datetime.now()
        lab.log(f"Auto-tagging started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Dataset: {dataset_name} ({split} split)")
        lab.log(f"Number of images: {actual_num_images}")
        lab.log(f"Output directory: {output_dir}")
        lab.log(f"Tag threshold: {threshold}")

        # Load the model
        lab.log("Loading WD tagger model...")
        lab.update_progress(10)
        model = from_pretrained_keras(model_name)
        lab.log("✅ Model loaded successfully")

        # Load tag labels from the model repo
        lab.log("Loading tag labels...")
        tags_file = hf_hub_download(repo_id=model_name, filename="selected_tags.csv")
        tag_names = []
        with open(tags_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    tag_names.append(row[0])
        lab.log(f"✅ Loaded {len(tag_names)} tags")

        results = []
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            lab.log(f"Processing image {i+1}/{total_images}: {image_path}")
            
            try:
                # Preprocess image
                processed_image = preprocess_image(image_path)
                
                # Run inference
                predictions = model.predict(processed_image)[0]  # Assuming output is (batch_size, num_tags)
                
                # Filter tags above threshold
                tags = []
                for j, prob in enumerate(predictions):
                    if prob > threshold:
                        tags.append({"tag": tag_names[j], "confidence": float(prob), "index": j})
                
                # Sort tags by confidence descending
                tags.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Save individual result
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                result_file = os.path.join(output_dir, f"{image_name}_tags.json")
                with open(result_file, "w") as f:
                    json.dump({
                        "image_path": image_path,
                        "tags": tags,
                        "threshold": threshold,
                        "processed_at": datetime.now().isoformat()
                    }, f, indent=2)
                
                # Save original image as artifact
                lab.save_artifact(image_path, f"{image_name}.png", type="image")
                
                # Annotate image with tags
                annotated_image_path = os.path.join(output_dir, f"{image_name}_annotated.png")
                annotate_image_with_tags(image_path, tags, annotated_image_path)
                lab.save_artifact(annotated_image_path, f"{image_name}_annotated.png", type="image")
                
                results.append({
                    "image_path": image_path,
                    "num_tags": len(tags),
                    "top_tags": tags[:10]  # Top 10 for summary
                })
                
            except Exception as e:
                lab.log(f"⚠️ Error processing {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })
            
            # Update progress
            progress = 10 + int((i + 1) / total_images * 80)  # 10% to 90%
            lab.update_progress(progress)

        # Save summary artifact
        summary_file = os.path.join(output_dir, "tagging_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "model": model_name,
                "dataset": dataset_name,
                "split": split,
                "total_images": total_images,
                "threshold": threshold,
                "results": results,
                "completed_at": datetime.now().isoformat()
            }, f, indent=2)
        
        summary_artifact_path = lab.save_artifact(summary_file, "tagging_summary.json")
        lab.log(f"✅ Saved tagging summary: {summary_artifact_path}")

        lab.update_progress(100)
        
        # Finish
        end_time = datetime.now()
        duration = end_time - start_time
        lab.log(f"Auto-tagging completed in {duration}")
        lab.finish("Auto-tagging completed successfully")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(duration),
            "output_dir": output_dir,
            "total_images": total_images,
            "model": model_name,
            "dataset": dataset_name
        }

    except Exception as e:
        error_msg = str(e)
        lab.log(f"❌ Auto-tagging failed: {error_msg}")
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}

if __name__ == "__main__":
    result = main()
    print("Auto-tagging result:", result)