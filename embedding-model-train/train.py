#!/usr/bin/env python3
"""
Embedding Model Training Script with TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Using lab.get_hf_callback() for automatic progress tracking and checkpoint saving
- Training embedding models with Sentence Transformers v3
- Automatic wandb URL detection when wandb is initialized within ML frameworks
"""

import os
import random
import torch
import json
from datetime import datetime

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


# --- Utility Functions ---
def normalize_dataset_columns(dataset, dataset_type_str):
    """
    Rename the dataset columns to the lower-case names derived from the dataset_type.
    It excludes any column named 'id' (which is preserved).
    Assumes that the relevant text columns (in order) are the first columns
    that are not 'id'.
    """
    expected_names = [name.strip().lower() for name in dataset_type_str.split("|")]
    # Get all columns except 'id'
    cols = [col for col in dataset.column_names if col.lower() != "id"]
    if len(expected_names) > len(cols):
        raise ValueError(f"Dataset does not have enough columns to match the dataset type '{dataset_type_str}'")
    mapping = {}
    for i, new_name in enumerate(expected_names):
        mapping[cols[i]] = new_name
    return dataset.rename_columns(mapping)


def get_loss_function(loss_name, model):
    """Dynamically import and instantiate the loss function from sentence_transformers.losses."""
    loss_module = __import__("sentence_transformers.losses", fromlist=[loss_name])
    try:
        loss_cls = getattr(loss_module, loss_name)
        return loss_cls(model)
    except AttributeError:
        raise ValueError(f"Loss function '{loss_name}' is not available in sentence_transformers.losses.")


def add_noise(sentence):
    """Randomly removes some words to create a noised version."""
    words = sentence.split()
    if len(words) < 2:
        return sentence  # Skip short sentences
    num_words_to_remove = max(1, len(words) // 4)  # Remove 25% of words
    indices_to_remove = random.sample(range(len(words)), num_words_to_remove)
    noised_words = [w for i, w in enumerate(words) if i not in indices_to_remove]
    return " ".join(noised_words)


def load_dataset_column(dataset, column_name="context"):
    """Load a specific column from a dataset and return the sentences as a list."""
    if column_name not in dataset.column_names:
        raise ValueError(f"Column '{column_name}' not found in dataset. Available columns: {dataset.column_names}")

    sentences = dataset[column_name]
    print(f"Loaded {len(sentences)} sentences from column '{column_name}'.")
    return sentences


def prepare_training_data(sentences):
    """Create dataset pairs with original and noised sentences."""
    data_pairs = [
        {"noised_text": add_noise(s), "original_text": s} for s in sentences if isinstance(s, str) and len(s) > 0
    ]
    return Dataset.from_list(data_pairs)


# Mapping from dataset type to allowed loss functions
ALLOWED_LOSSES = {
    "anchor | positive": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "MegaBatchMarginLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
    "anchor | positive | negative": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "TripletLoss",
        "CachedGISTEmbedLoss",
        "GISTEmbedLoss",
    ],
    "sentence_A | sentence_B | score": ["CoSENTLoss", "AnglELoss", "CosineSimilarityLoss"],
    "single sentences": ["ContrastiveTensionLoss", "DenoisingAutoEncoderLoss"],
    "single sentences | class": [
        "BatchAllTripletLoss",
        "BatchHardSoftMarginTripletLoss",
        "BatchHardTripletLoss",
        "BatchSemiHardTripletLoss",
    ],
    "anchor | anchor": ["ContrastiveTensionLossInBatchNegatives"],
    "damaged_sentence | original_sentence": ["DenoisingAutoEncoderLoss"],
    "sentence_A | sentence_B | class": ["SoftmaxLoss"],
    "anchor | positve/negative | class": ["ContrastiveLoss", "OnlineContrastiveLoss"],
    "anchor | positive | negative_1 | negative_2 | ... | negative_n": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "CachedGISTEmbedLoss",
    ],
    "id | anchor | positive": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "MegaBatchMarginLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
}


def train_embedding_model():
    """Main function to train an embedding model using TransformerLab"""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()

        # Get parameters from task configuration (set via UI)
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        dataset_name = config.get("dataset_name", "sentence-transformers/stsb")
        output_dir = config.get("output_dir", "./output")
        log_to_wandb = config.get("log_to_wandb", False)

        # Dataset and loss configuration
        dataset_type = config.get("dataset_type", "anchor | positive")
        loss_function = config.get("loss_function", "MultipleNegativesRankingLoss")
        loss_modifier_name = config.get("loss_modifier_name", "None")
        text_column_name = config.get("text_column_name", "context")

        # Training hyperparameters
        num_train_epochs_raw = config.get("num_train_epochs", 3)
        num_train_epochs = int(num_train_epochs_raw) if isinstance(num_train_epochs_raw, (str, int, float)) else num_train_epochs_raw

        max_steps = config.get("max_steps", 5)

        batch_size_raw = config.get("batch_size", 16)
        batch_size = int(batch_size_raw) if isinstance(batch_size_raw, (str, int, float)) else batch_size_raw

        learning_rate_raw = config.get("learning_rate", 2e-5)
        learning_rate = float(learning_rate_raw) if isinstance(learning_rate_raw, (str, int, float)) else learning_rate_raw

        warmup_ratio_raw = config.get("warmup_ratio", 0.1)
        warmup_ratio = float(warmup_ratio_raw) if isinstance(warmup_ratio_raw, (str, int, float)) else warmup_ratio_raw

        fp16 = config.get("fp16", False)
        bf16 = config.get("bf16", False)

        max_samples_raw = config.get("max_samples", -1)
        max_samples = int(max_samples_raw) if isinstance(max_samples_raw, (str, int, float)) else max_samples_raw

        matryoshka_dims = config.get("matryoshka_dims", [768, 512, 256, 128, 64])

        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")

        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"Dataset type: {dataset_type}")
        lab.log(f"Loss function: {loss_function}")
        lab.log(f"Learning rate: {learning_rate}")
        lab.log(f"Batch size: {batch_size}")
        lab.log(f"Number of epochs: {num_train_epochs}")
        lab.log(f"Max steps: {max_steps}")
        lab.log(f"Max samples: {max_samples}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Validate loss function against dataset type
        if dataset_type not in ALLOWED_LOSSES:
            raise ValueError(f"Dataset type '{dataset_type}' is not recognized.")

        allowed = ALLOWED_LOSSES[dataset_type]
        if loss_function not in allowed:
            raise ValueError(
                f"Loss function '{loss_function}' is not allowed for dataset type '{dataset_type}'. "
                f"Allowed loss functions: {allowed}"
            )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            full_dataset = load_dataset(dataset_name)
            
            # Try to get the train split
            if "train" in full_dataset:
                full_dataset = full_dataset["train"]
            else:
                # If no train split, use the first available split
                split_name = list(full_dataset.keys())[0]
                lab.log(f"No 'train' split found, using '{split_name}' split instead")
                full_dataset = full_dataset[split_name]

            lab.log(f"Loaded dataset with {len(full_dataset)} examples")

        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            lab.error("Training failed - dataset loading error")
            raise e

        lab.update_progress(20)

        # Apply max_samples limit if specified
        if max_samples > 0 and max_samples < len(full_dataset):
            full_dataset = full_dataset.select(range(max_samples))
            lab.log(f"Limited dataset to {max_samples} samples")

        # Normalize dataset columns according to the dataset type
        lab.log("Preparing dataset...")
        if dataset_type != "single sentences":
            normalized_dataset = normalize_dataset_columns(full_dataset, dataset_type)
        else:
            sentences = load_dataset_column(full_dataset, text_column_name)
            normalized_dataset = prepare_training_data(sentences)

        lab.update_progress(30)

        # Prepare an IR evaluator if the normalized dataset has "id", "anchor", and "positive"
        evaluator = None
        has_evaluator = False
        if all(col in normalized_dataset.column_names for col in ["id", "anchor", "positive"]):
            lab.log("Creating Information Retrieval evaluator...")
            corpus = dict(zip(normalized_dataset["id"], normalized_dataset["positive"]))
            queries = dict(zip(normalized_dataset["id"], normalized_dataset["anchor"]))
            relevant_docs = {q_id: [q_id] for q_id in queries}
            matryoshka_evaluators = []

            for dim in matryoshka_dims:
                ir_eval = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f"dim_{dim}",
                    truncate_dim=dim,
                    score_functions={"cosine": cos_sim},
                )
                matryoshka_evaluators.append(ir_eval)

            if matryoshka_evaluators:
                evaluator = SequentialEvaluator(matryoshka_evaluators)
                has_evaluator = True
                lab.log("✅ IR evaluator created")

        lab.update_progress(40)

        # Load the model
        lab.log(f"Loading Sentence Transformer model: {model_name}")
        try:
            model = SentenceTransformer(
                model_name, 
                device=("cuda" if torch.cuda.is_available() else "cpu"), 
                trust_remote_code=True
            )
            lab.log("✅ Model loaded successfully")
        except Exception as e:
            lab.log(f"Error loading model: {e}")
            lab.error("Training failed - model loading error")
            raise e

        lab.update_progress(50)

        # Configure loss function
        lab.log("Configuring loss function...")
        inner_train_loss = get_loss_function(loss_function, model)

        # Apply loss modifier if specified
        if loss_modifier_name != "None":
            if dataset_type == "single sentences":
                lab.log("Warning: Loss modifier is not supported for single sentences dataset type.")
                lab.log("Using the default loss function instead.")
                train_loss = inner_train_loss
            else:
                loss_modifier_module = __import__("sentence_transformers.losses", fromlist=[loss_modifier_name])
                loss_modifier_cls = getattr(loss_modifier_module, loss_modifier_name)

                if loss_modifier_name == "AdaptiveLayerLoss":
                    # AdaptiveLayerLoss does not take matryoshka_dims as a parameter
                    train_loss = loss_modifier_cls(model=model, loss=inner_train_loss)
                else:
                    train_loss = loss_modifier_cls(model=model, loss=inner_train_loss, matryoshka_dims=matryoshka_dims)
        else:
            train_loss = inner_train_loss

        lab.log("✅ Loss function configured")
        lab.update_progress(60)

        # Configure training arguments
        lab.log("Setting up training arguments...")
        training_args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
            num_train_epochs=num_train_epochs,
            max_steps=max_steps if max_steps > 0 else -1,
            per_device_train_batch_size=batch_size,
            fp16=fp16,
            bf16=bf16,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            load_best_model_at_end=False,
            eval_strategy="epoch" if has_evaluator else "no",
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
            report_to=["wandb"] if log_to_wandb else [],
            run_name=f"embedding-train-{lab.job.id}",
            metric_for_best_model="eval_dim_128_cosine_ndcg@10" if has_evaluator else None,
            greater_is_better=True if has_evaluator else None,
            resume_from_checkpoint=checkpoint if checkpoint else None,
        )

        # Get TransformerLab callback for automatic progress tracking and checkpoint saving
        transformerlab_callback = lab.get_hf_callback()

        # Select appropriate columns for training
        if all(col in normalized_dataset.column_names for col in ["anchor", "positive"]):
            train_data = normalized_dataset.select_columns(["anchor", "positive"])
        else:
            train_data = normalized_dataset

        # Create and run the trainer
        lab.log("Creating trainer...")
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            loss=train_loss,
            evaluator=evaluator,
            callbacks=[transformerlab_callback],
        )
        lab.log("✅ Trainer created")

        lab.update_progress(70)

        # Start training
        lab.log("Starting training...")
        try:
            trainer.train()
            lab.log("✅ Training completed successfully")

            # Save the model
            lab.log("Saving model...")
            trainer.save_model(output_dir)
            lab.log("✅ Model saved")

            # Create training summary artifact
            progress_file = os.path.join(output_dir, "training_summary.json")
            with open(progress_file, "w") as f:
                json.dump(
                    {
                        "training_type": "Sentence Transformers Embedding Model",
                        "model_name": model_name,
                        "dataset": dataset_name,
                        "dataset_type": dataset_type,
                        "loss_function": loss_function,
                        "loss_modifier": loss_modifier_name,
                        "num_train_epochs": num_train_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "warmup_ratio": warmup_ratio,
                        "completed_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            progress_artifact_path = lab.save_artifact(progress_file, "training_summary.json")
            lab.log(f"Saved training summary: {progress_artifact_path}")

        except Exception as e:
            lab.log(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            lab.error("Training failed")
            raise e

        lab.update_progress(90)

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")

        # Save final artifacts
        final_model_file = os.path.join(output_dir, "final_model_summary.txt")
        with open(final_model_file, "w") as f:
            f.write("Final Embedding Model Summary\n")
            f.write("=============================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Dataset Type: {dataset_type}\n")
            f.write(f"Loss Function: {loss_function}\n")
            f.write(f"Epochs: {num_train_epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Completed at: {end_time}\n")

        final_model_path = lab.save_artifact(final_model_file, "final_model_summary.txt")
        lab.log(f"Saved final model summary: {final_model_path}")

        # Save training configuration as artifact
        config_file = os.path.join(output_dir, "training_config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")

        # Save the trained model to TransformerLab's model directory
        saved_path = lab.save_model(output_dir, name="embedding_trained_model")
        lab.log(f"✅ Model saved to job models directory: {saved_path}")


        # Finish wandb run if it was initialized
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
                lab.log("✅ Wandb run finished")
        except Exception:
            pass

        print("Complete")

        # Complete the job in TransformerLab
        lab.finish("Training completed successfully!")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": output_dir,
            "saved_model_path": saved_path,
            "trainer_type": "Sentence Transformers",
            "gpu_used": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        }

    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")

        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting Embedding Model training...")
    result = train_embedding_model()
    print("Training result:", result)
