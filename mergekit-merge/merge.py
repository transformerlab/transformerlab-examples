import os
import subprocess
import yaml
from lab import lab


def merge_models():
    """Merge models using mergekit"""

    try:
        lab.init()
        config = lab.get_config()

        # Training configuration
        merge_config = {
            "model1": config.get("model1", "HuggingFaceTB/SmolLM-135M"),
            "model2": config.get("model2", "HuggingFaceTB/SmolLM-135M-Instruct"),
            "base_model": config.get("base_model", "HuggingFaceTB/SmolLM-135M"),
            "weight1": float(config.get("weight1", "0.5")),
            "weight2": float(config.get("weight2", "0.5")),
            "merge_method": config.get("merge_method", "linear"),
            "output_dir": config.get("output_dir", "./merged_model"),
            "dtype": config.get("dtype", "float16"),
            "tokenizer_source": config.get("tokenizer_source", "union"),
        }

        lab.set_config(merge_config)
        lab.log("Starting model merge with mergekit")

        # Create mergekit config
        mergekit_config = {
            "merge_method": merge_config["merge_method"],
            "models": [
                {
                    "model": merge_config["model1"],
                    "parameters": {"weight": merge_config["weight1"]},
                },
                {
                    "model": merge_config["model2"],
                    "parameters": {"weight": merge_config["weight2"]},
                },
            ],
            "dtype": merge_config["dtype"],
            "tokenizer_source": merge_config["tokenizer_source"],
        }

        if merge_config["base_model"]:
            mergekit_config["base_model"] = merge_config["base_model"]

        # Write config to file
        config_path = "merge_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(mergekit_config, f)

        lab.log("Generated mergekit config")

        # Run mergekit
        lab.update_progress(10)
        cmd = ["mergekit-yaml", config_path, merge_config["output_dir"]]
        lab.log(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            lab.log("Merge completed successfully")
            lab.update_progress(90)

            # Save the merged model
            saved_path = lab.save_model(merge_config["output_dir"], name="merged_model")
            lab.log(f"Merged model saved: {saved_path}")

            lab.update_progress(100)
            lab.finish("Merge completed successfully")

            return {
                "status": "success",
                "saved_model_path": saved_path,
                "output_dir": merge_config["output_dir"],
            }
        else:
            error_msg = f"Merge failed: {result.stderr}"
            lab.error(error_msg)
            return {"status": "error", "error": error_msg}

    except Exception as e:
        error_msg = str(e)
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    result = merge_models()
    print(result)
