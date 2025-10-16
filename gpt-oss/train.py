import argparse
import os
from datetime import datetime

from accelerate import Accelerator
from accelerate import ProfileKwargs
from datasets import load_dataset
from peft import get_peft_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Mxfp4Config
from transformers import TrainerCallback
from trl import SFTConfig
from trl import SFTTrainer

from lab import lab
import json


class TransformerLabCallback(TrainerCallback):
    """Callback to integrate with TransformerLab SDK for progress tracking and logging"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics"""
        if logs:
            # Log key metrics to TransformerLab
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    lab.log(f"{key}: {value:.4f}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        self.current_step = state.global_step
        # Update progress (0-100%)
        progress = min(int((self.current_step / self.total_steps) * 100), 100)
        lab.job.update_progress(progress)

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        if state.global_step > 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                # Use SDK to save checkpoint
                lab.save_checkpoint(checkpoint_dir, name=f"checkpoint-{state.global_step}")
                lab.log(f"Checkpoint saved at step {state.global_step}")


class ProfilingSFTTrainer(SFTTrainer):

    def __init__(self, *args, accelerator_profiler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator_profiler = accelerator_profiler

    def training_step(self, *args, **kwargs):
        result = super().training_step(*args, **kwargs)
        if self.accelerator_profiler is not None:
            self.accelerator_profiler.step()
        return result


def main():
    """Main training function with TransformerLab SDK integration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a model using SFT on Codeforces dataset")
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/gpt-oss-20b",
        help="The model ID to use for training (default: openai/gpt-oss-20b)")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="HuggingFaceH4/Multilingual-Thinking",
        help="The dataset ID to use for training (default: HuggingFaceH4/Multilingual-Thinking)")
    parser.add_argument("--enable_lora",
                        action="store_true",
                        default=False,
                        help="Enable LoRA")
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        default=False,
        help="Enable accelerate profiling with chrome trace export")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)")
    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=1,
                        help="Training batch size per device (default: 1)")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints")
    args = parser.parse_args()
    
    try:
        # Initialize TransformerLab SDK
        lab.init()
        lab.log("TransformerLab SDK initialized")
        
        # Prepare training configuration for TransformerLab
        training_config = {
            "model_name": args.model_id,
            "dataset": args.dataset_id,
            "output_dir": args.output_dir or f"{args.model_id}-checkpoint",
            "_config": {
                "enable_lora": args.enable_lora,
                "enable_profiling": args.enable_profiling,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_seq_length": 1024,
                "warmup_ratio": 0.03,
            },
        }
        lab.set_config(training_config)
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
    
    except Exception as e:
        # If SDK initialization fails, continue without it
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        start_time = datetime.now()

    # Setup profiling if enabled
    accelerator_kwargs = {}
    if args.enable_profiling:
        lab.log("Setting up profiling with chrome trace export")

        def trace_handler(p):
            trace_path = f"/tmp/trace_{p.step_num}.json"
            p.export_chrome_trace(trace_path)
            lab.log(f"Exported trace to {trace_path}")

        profile_kwargs = ProfileKwargs(activities=["cpu", "cuda"],
                                       schedule_option={
                                           "wait": 1,
                                           "warmup": 1,
                                           "active": 1,
                                           "repeat": 0,
                                           "skip_first": 1,
                                       },
                                       on_trace_ready=trace_handler)
        accelerator_kwargs['kwargs_handlers'] = [profile_kwargs]

    accelerator = Accelerator(**accelerator_kwargs)
    model_id = args.model_id

    # Load dataset
    lab.log(f"Loading dataset: {args.dataset_id}")
    num_proc = int(os.cpu_count() / 2)
    train_dataset = load_dataset(args.dataset_id,
                                 split="train",
                                 num_proc=num_proc)
    lab.log(f"Dataset loaded: {len(train_dataset)} training examples")

    quantization_config = Mxfp4Config(dequantize=True)

    device_map_args = {}
    if args.enable_lora:
        device_map_args = {'device_map': 'auto'}

    # Load model
    lab.log(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=False,
        quantization_config=quantization_config,
        **device_map_args,
    )

    lab.log(f'Model loaded successfully: {args.model_id}')

    if args.enable_lora:
        num_layers = 0
        target_parameters = []
        if args.model_id == 'openai/gpt-oss-120b':
            num_layers = 36
        elif args.model_id == 'openai/gpt-oss-20b':
            num_layers = 24

        for i in range(num_layers):
            target_parameters.append(f'{i}.mlp.experts.gate_up_proj')
            target_parameters.append(f'{i}.mlp.experts.down_proj')

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            target_parameters=target_parameters,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Prepare training configuration
    output_dir = args.output_dir or f"{model_id}-checkpoint"
    os.makedirs(output_dir, exist_ok=True)
    lab.log(f"Output directory: {output_dir}")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=1024,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        dataset_num_proc=num_proc,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
    )
    
    # Calculate total training steps for progress tracking
    total_steps = (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    lab.log(f"Total training steps: {total_steps}")
    
    # Create TransformerLab callback for progress tracking
    tlab_callback = TransformerLabCallback(total_steps)

    # Train model with optional profiling
    lab.log("Starting training...")
    
    trainer_kwargs = {
        'args': training_args,
        'model': model,
        'train_dataset': train_dataset,
        'callbacks': [tlab_callback],
    }

    if args.enable_profiling:
        with accelerator.profile() as prof:
            trainer_kwargs['accelerator_profiler'] = prof
            trainer = ProfilingSFTTrainer(**trainer_kwargs)
            trainer.train()
    else:
        trainer = ProfilingSFTTrainer(**trainer_kwargs)
        trainer.train()
    
    # Training completed
    end_time = datetime.now()
    training_duration = end_time - start_time
    lab.log(f"Training completed in {training_duration}")
    
    # Save final model using SDK directory structure
    final_model_dir = os.path.join(lab.job.get_dir(), "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    lab.log(f"Saving final model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    
    # Also save final model as a checkpoint for SDK tracking
    lab.save_checkpoint(final_model_dir, name="final_model")
    
    # Store model metadata in job data
    lab.job.update_job_data_field("final_model_path", final_model_dir)
    
    # Complete the job
    lab.finish("Training completed successfully")
    
    return {
        "status": "success",
        "job_id": lab.job.id if hasattr(lab.job, 'id') else None,
        "duration": str(training_duration),
        "output_dir": final_model_path,
    }


if __name__ == "__main__":
    try:
        result = main()
        print(f"\nTraining Result: {result}")
    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        print("Training interrupted by user")
    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)