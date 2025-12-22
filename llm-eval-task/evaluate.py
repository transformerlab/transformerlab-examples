#!/usr/bin/env python3
"""
Evaluation script using EleutherAI LM Evaluation Harness.
Runs specified benchmarks on an LLM and saves the results as artifacts.
"""

import os
import json
import torch
from datetime import datetime

# TransformerLab import
from lab import lab

# EleutherAI LM Eval imports
import lm_eval
from lm_eval.utils import make_table

# Hugging Face login
from huggingface_hub import login

def setup_environment():
    """Handle logins and configuration"""
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    # Configuration
    return {
        "model_name": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B"),
        "tasks": os.getenv("EVAL_TASKS", "hellaswag,arc_easy").split(","),
        "batch_size": os.getenv("BATCH_SIZE", "auto"),
        "output_dir": "./output",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

class LabLogWrapper:
    """Redirects print statements to Lab logs if needed, or just helpers"""
    @staticmethod
    def log(msg):
        print(msg)
        lab.log(msg)

def run_evaluation():
    config = setup_environment()
    
    # Initialize Lab
    lab.init()
    lab.set_config(config)
    
    lab.log("🚀 Starting LLM Evaluation Task")
    lab.log(f"Model: {config['model_name']}")
    lab.log(f"Tasks: {', '.join(config['tasks'])}")
    lab.log(f"Device: {config['device']}")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    try:
        start_time = datetime.now()
        lab.update_progress(10)
        
        # 1. Run Evaluation
        lab.log("Loading model and running benchmarks... (This may take time)")
        
        # lm_eval.simple_evaluate is the main entry point
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={config['model_name']},trust_remote_code=True,dtype=auto",
            tasks=config["tasks"],
            batch_size=config["batch_size"],
            device=config["device"],
            limit=100,
        )
        
        lab.update_progress(80)
        lab.log("✅ Evaluation completed.")

        # 2. Process and Display Results
        if results is None:
            raise ValueError("Evaluation returned no results.")

        # Generate a markdown table for the logs
        results_table = make_table(results)
        print("\n" + results_table)
        
        # Log specific metrics to Lab
        # Structure of 'results' dict varies, usually results['results'][task_name]
        if 'results' in results:
            for task, metrics in results['results'].items():
                acc = metrics.get('acc,none') or metrics.get('acc')
                if acc:
                    lab.log(f"📊 {task} Accuracy: {acc:.4f}")

        # 3. Save Artifacts
        lab.log("Saving results artifacts...")
        
        # Save full JSON dump
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"eval_results_{timestamp}.json"
        json_path = os.path.join(config["output_dir"], json_filename)
        
        # Convert non-serializable objects (like functions) to strings if necessary
        # lm_eval usually returns serializable data, but safely dumping is good practice
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        saved_json = lab.save_artifact(json_path, "evaluation_results.json")
        lab.log(f"saved artifact: {saved_json}")

        # Save a readable summary text file
        txt_path = os.path.join(config["output_dir"], "summary.txt")
        with open(txt_path, "w") as f:
            f.write(f"Evaluation Summary for {config['model_name']}\n")
            f.write(f"Date: {start_time}\n")
            f.write("============================================\n\n")
            f.write(results_table)
            
        saved_txt = lab.save_artifact(txt_path, "evaluation_summary.txt")
        lab.log(f"saved artifact: {saved_txt}")
        
        lab.update_progress(100)
        
        duration = datetime.now() - start_time
        lab.finish(f"Evaluation complete in {duration}")
        
        return {
            "status": "success",
            "model": config["model_name"],
            "artifacts": [saved_json, saved_txt]
        }

    except Exception as e:
        lab.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    run_evaluation()