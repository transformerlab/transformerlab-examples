from unsloth import is_bfloat16_supported
import time
import torch

from transformers import TrainingArguments, Trainer
from datasets import Audio

from trainer import CsmAudioTrainer, OrpheusAudioTrainer

from transformerlab.sdk.v1.train import tlab_trainer  # noqa: E402
from lab import storage

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

def train_model():
    """Train an audio model using unsloth."""

    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training configuration
    training_config = {
        "experiment_name": "unsloth-tts-training",
        "model_name": "unsloth/orpheus-3b-0.1-ft",  # Small model for testing
        "dataset": "bosonai/EmergentTTS-Eval",  # Example dataset
        "template_name": "unsloth-tts-demo",
        "output_dir": "./output",
        "log_to_wandb": True,
        "_config": {
            "dataset_name": "bosonai/EmergentTTS-Eval",
            "audio_column_name": "audio",
            "text_column_name": "text",
            # Get configuration values
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_r": 8,
            "maximum_sequence_length": 1024,
            "max_grad_norm": 1.0,
            "learning_rate": 5e-05,
            "learning_rate_schedule": "constant",
    max_grad_norm = float(tlab_trainer.params.max_grad_norm)
    batch_size = int(tlab_trainer.params.batch_size)
    num_epochs = int(tlab_trainer.params.num_train_epochs)
    weight_decay = float(tlab_trainer.params.weight_decay)
    adam_beta1 = float(tlab_trainer.params.adam_beta1)
    adam_beta2 = float(tlab_trainer.params.adam_beta2)
    adam_epsilon = float(tlab_trainer.params.adam_epsilon)
    output_dir = tlab_trainer.params.output_dir
    report_to = tlab_trainer.report_to
    sampling_rate = int(tlab_trainer.params.get("sampling_rate", 24000))
    max_steps = int(tlab_trainer.params.get("max_steps", -1))
    model_architecture = tlab_trainer.params.get("model_architecture")
    device = "cuda" if torch.cuda.is_available() else "cpu"
            # "lr": 2e-4,
            # "num_train_epochs": 1,
            # "batch_size": 2,
            # "gradient_accumulation_steps": 4,
            # "warmup_steps": 5,
            # "max_steps": 100,
            # "max_seq_length": 2048,
            # "lora_r": 16,
            # "lora_alpha": 16,
            # "lora_dropout": 0.05,
            # "logging_steps": 1,
            # "save_steps": 50,
            # "weight_decay": 0.01,
            # "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        },
    }
    try:
        # Initialize lab with default/simple API
        lab.init()
        lab.set_config(training_config)

        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Create output directory if it doesn't exist
        os.makedirs(training_config["output_dir"], exist_ok=True)

        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset

            datasets = load_dataset(training_config["dataset"])
            dataset = datasets["train"]
            lab.log(f"Loaded dataset with {len(datasets['train'])} training examples.")

            if training_config["_config"]["audio_column_name"] not in dataset.column_names or training_config["_config"]["text_column_name"] not in dataset.column_names:
                lab.log(f"Missing required columns: '{training_config['_config']['audio_column_name']}' and '{training_config['_config']['text_column_name']}'.")
                lab.finish("Training failed due to missing dataset columns.")
                return {"status": "error", "error": "Missing required dataset columns."}
        
        except Exception as e:
            lab.log(f"❌ Failed to load dataset: {e}")
            lab.finish("Training failed due to dataset loading error.")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(20)

        # Load model and tokenizer using
        lab.log("Loading model and tokenizer...")


    # Get configuration values
    lora_alpha = int(tlab_trainer.params.get("lora_alpha", 16))
    lora_dropout = float(tlab_trainer.params.get("lora_dropout", 0))
    lora_r = int(tlab_trainer.params.get("lora_r", 8))
    model_id = tlab_trainer.params.model_name

    max_seq_length = int(tlab_trainer.params.maximum_sequence_length)
    learning_rate = float(tlab_trainer.params.learning_rate)
    learning_rate_schedule = tlab_trainer.params.get("learning_rate_schedule", "constant")
    max_grad_norm = float(tlab_trainer.params.max_grad_norm)
    batch_size = int(tlab_trainer.params.batch_size)
    num_epochs = int(tlab_trainer.params.num_train_epochs)
    weight_decay = float(tlab_trainer.params.weight_decay)
    adam_beta1 = float(tlab_trainer.params.adam_beta1)
    adam_beta2 = float(tlab_trainer.params.adam_beta2)
    adam_epsilon = float(tlab_trainer.params.adam_epsilon)
    output_dir = tlab_trainer.params.output_dir
    report_to = tlab_trainer.report_to
    sampling_rate = int(tlab_trainer.params.get("sampling_rate", 24000))
    max_steps = int(tlab_trainer.params.get("max_steps", -1))
    model_architecture = tlab_trainer.params.get("model_architecture")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Getting the speaker id is important for multi-speaker models and speaker consistency
    speaker_key = "source"
    if "source" not in dataset.column_names and "speaker_id" not in dataset.column_names:
        print('No speaker found, adding default "source" of 0 for all examples')
        new_column = ["0"] * len(dataset)
        dataset = dataset.add_column("source", new_column)
    elif "source" not in dataset.column_names and "speaker_id" in dataset.column_names:
        speaker_key = "speaker_id"

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=sampling_rate))
    max_audio_length = max(len(example[audio_column_name]["array"]) for example in dataset)

    if model_architecture == "CsmForConditionalGeneration":
        model_trainer = CsmAudioTrainer(
            model_name=model_name,
            speaker_key=speaker_key,
            context_length=max_seq_length,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sampling_rate=sampling_rate,
            max_audio_length=max_audio_length,
            audio_column_name=audio_column_name,
            text_column_name=text_column_name,
        )
    elif "orpheus" in model_name:
        model_trainer = OrpheusAudioTrainer(
            model_name=model_name,
            speaker_key=speaker_key,
            context_length=max_seq_length,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            sampling_rate=sampling_rate,
            max_audio_length=max_audio_length,
            batch_size=batch_size,
            audio_column_name=audio_column_name,
            text_column_name=text_column_name,
        )
    else:
        raise ValueError(f"Model architecture {model_architecture} is not supported for audio training.")

    processed_ds = dataset.map(
        model_trainer.preprocess_dataset,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
    )

    processed_ds = processed_ds.filter(lambda x: x is not None)

    print(f"Processed dataset length: {len(processed_ds)}")

    # Create progress callback using tlab_trainer
    progress_callback = tlab_trainer.create_progress_callback(framework="huggingface")

    # Training run name
    today = time.strftime("%Y%m%d-%H%M%S")
    run_suffix = tlab_trainer.params.get("template_name", today)
    trainer = Trainer(
        model=model_trainer.model,
        train_dataset=processed_ds,
        callbacks=[progress_callback],
        args=TrainingArguments(
            logging_dir=storage.join(output_dir, f"job_{tlab_trainer.params.job_id}_{run_suffix}"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            warmup_ratio=0.03,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            save_strategy="epoch",
            weight_decay=weight_decay,
            lr_scheduler_type=learning_rate_schedule,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            disable_tqdm=False,
            seed=3407,
            output_dir=output_dir,
            run_name=f"job_{tlab_trainer.params.job_id}_{run_suffix}",
            report_to=report_to,
        ),
    )
    # Train the model
    try:
        trainer.train()
    except Exception as e:
        raise e

    # Save the model
    try:
        trainer.save_model(output_dir=tlab_trainer.params.adaptor_output_dir)
    except Exception as e:
        raise e

    # Return success message
    return "Audio model trained successfully."


train_model()