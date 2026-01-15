import json
import os
import re
import subprocess
import time

from jinja2 import Environment
from lab import lab

# Setup Jinja environment
jinja_environment = Environment()


def train_model():
    """Main training function for AutoTrain SFT"""

    # Training configuration
    training_config = {
        "experiment_name": "autotrain-sft-training",
        "model_name": "microsoft/DialoGPT-small",
        "dataset": "HuggingFaceH4/no_robots",
        "template_name": "autotrain-sft-demo",
        "output_dir": "./output",
        "log_to_wandb": False,
        "_config": {
            "learning_rate": 0.0002,
            "batch_size": 4,
            "num_train_epochs": 1,
            "adaptor_name": "adaptor",
            "formatting_template": "{% for message in messages %}{{ message.role | title }}: {{ message.content }}\n{% endfor %}",
            "dataset_name": "HuggingFaceH4/no_robots",
            "max_steps": 10,
        },
    }

    try:
        # Initialize lab
        lab.init()
        lab.set_config(training_config)

        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")

        # Log start time
        start_time = time.time()
        lab.log(f"Training started at {time.ctime(start_time)}")

        config = training_config["_config"]

        # Parameters to pass to autotrain
        learning_rate = config["learning_rate"]
        batch_size = config.get("batch_size", 4)
        num_train_epochs = config.get("num_train_epochs", 1)

        # Generate a model name using the original model and the passed adaptor
        adaptor_name = config.get("adaptor_name", "default")
        input_model_no_author = training_config["model_name"].split("/")[-1]
        project_name = f"{input_model_no_author}-{adaptor_name}".replace(".", "")

        # Setup directories
        plugin_dir = os.getcwd()
        data_directory = os.path.join(plugin_dir, "data")
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        # Get template from config
        formatting_template = jinja_environment.from_string(config["formatting_template"])

        # Load datasets (train, test) - using datasets library directly
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            # For now, assume dataset is specified in config
            # In a real implementation, this would be passed from the UI
            dataset_name = training_config["dataset"]
            datasets = load_dataset(dataset_name)
            
            # Take only a small subset for fast testing
            train_subset = datasets["train"].select(range(min(100, len(datasets["train"]))))
            test_subset = datasets["test"].select(range(min(20, len(datasets["test"]))))
            
            dataset_types = ["train", "test"]
            dataset = {"train": train_subset, "test": test_subset}

        except Exception as e:
            lab.log(f"❌ Failed to load dataset: {e}")
            lab.finish("Training failed due to dataset loading error.")
            return {"status": "error", "error": str(e)}

        for dataset_type in dataset_types:
            lab.log(f"Loaded {dataset_type} dataset with {len(dataset[dataset_type])} examples (subset for fast testing).")

            # Output training files in templated format
            with open(f"{data_directory}/{dataset_type}.jsonl", "w") as f:
                for i in range(len(dataset[dataset_type])):
                    data_line = dict(dataset[dataset_type][i])
                    line = formatting_template.render(data_line)

                    # Escape newlines for jsonl format
                    line = line.replace("\n", "\\n")
                    line = line.replace("\r", "\\r")
                    o = {"text": line}
                    f.write(json.dumps(o) + "\n")

        # Copy test.jsonl to valid.jsonl (validation = test)
        if os.path.exists(f"{data_directory}/test.jsonl"):
            os.system(f"cp {data_directory}/test.jsonl {data_directory}/valid.jsonl")

        lab.log("Example formatted training example:")
        if "train" in dataset and len(dataset["train"]) > 0:
            example = formatting_template.render(dataset["train"][0])
            lab.log(example)

        lab.update_progress(20)

        # Prepare autotrain command
        popen_command = [
            "autotrain",
            "llm",
            "--train",
            "--model",
            training_config["model_name"],
            "--data-path",
            data_directory,
            "--lr",
            str(learning_rate),
            "--batch-size",
            str(batch_size),
            "--epochs",
            str(num_train_epochs),
            "--trainer",
            "sft",
            "--peft",
            "--merge-adapter",
            "--auto_find_batch_size",
            "--project-name",
            project_name,
        ]

        lab.log("Running command:")
        lab.log(" ".join(popen_command))

        lab.log("Training beginning:")
        lab.update_progress(30)

        # Run the subprocess with output monitoring
        iteration = 0
        it_per_sec = 0
        percent_complete = 0

        with subprocess.Popen(
            popen_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
        ) as process:
            for line in process.stdout:
                # Parse progress from output lines
                # Progress complete pattern
                pattern = r"\s*(\d+)\%\|.+?(?=\d+/)(\d+)/.+?(?=\d+.\d+s/it)(\d+.\d+)s/it"
                match = re.search(pattern, line)
                if match:
                    percent_complete = int(match.group(1))
                    iteration = int(match.group(2))
                    it_per_sec = float(match.group(3))
                    # Update progress in lab
                    lab.update_progress(30 + (percent_complete * 0.6))  # Scale to 30-90%

                # Parse metrics for logging
                pattern = r"INFO.+?{'loss': (\d+\.\d+), 'grad_norm': (\d+\.\d+), 'learning_rate': (\d+\.\d+), 'epoch': (\d+\.\d+)}"
                match = re.search(pattern, line)
                if match:
                    loss = float(match.group(1))
                    grad_norm = float(match.group(2))
                    learning_rate = float(match.group(3))
                    epoch = float(match.group(4))

                    lab.log(f"Progress: {percent_complete}%")
                    lab.log(f"Iteration: {iteration}")
                    lab.log(f"It/sec: {it_per_sec}")
                    lab.log(f"Loss: {loss}")
                    lab.log(f"Epoch: {epoch}")

                # Print the output line
                print(line, end="", flush=True)

        lab.update_progress(90)

        # Clean up and move model
        try:
            # Remove autotrain data as it's not needed anymore
            autotrain_data_path = os.path.join(project_name, "autotrain_data")
            if os.path.exists(autotrain_data_path):
                os.system(f"rm -rf {autotrain_data_path}")
        except Exception as e:
            lab.log(f"Failed to delete unnecessary data: {e}")

        try:
            # Move the model to the lab directory
            if os.path.exists(project_name):
                saved_path = lab.save_model(project_name, name=f"{project_name}_trained")
                lab.log(f"✅ Model saved to lab: {saved_path}")
                
                # Save checkpoint
                saved_checkpoint_path = lab.save_checkpoint(project_name, name=f"{project_name}_checkpoint")
                lab.log(f"✅ Checkpoint saved to lab: {saved_checkpoint_path}")
            else:
                lab.log("⚠️ Model directory not found")
        except Exception as e:
            lab.log(f"Failed to save model: {e}")

        end_time = time.time()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration:.2f} seconds")

        lab.update_progress(100)
        lab.finish("Training completed successfully")

        return {
            "status": "success",
            "duration": training_duration,
            "model_name": project_name,
            "saved_model_path": saved_path if 'saved_path' in locals() else None,
            "saved_checkpoint_path": saved_checkpoint_path if 'saved_checkpoint_path' in locals() else None,
        }

    except Exception as e:
        error_msg = str(e)
        lab.log(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting Autotrain SFT training...")
    result = train_model()
    print("Training result:", result)