#!/usr/bin/env python3
"""
Inference Metrics Evaluation script using TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Batch inference evaluation with metrics tracking
- Automatic progress tracking and metric logging
- No dependency on FastChat - uses direct HTTP API calls
"""

import os
import json
import pandas as pd
from datetime import datetime

from lab import lab

# Login to huggingface
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))


# Metrics mapping
METRICS_MAP = {
    "Time to First Token (TTFT)": "time_to_first_token",
    "Total Time": "time_total",
    "Prompt Tokens": "prompt_tokens",
    "Completion Tokens": "completion_tokens",
    "Total Tokens": "total_tokens",
    "Tokens per Second": "tokens_per_second",
}


async def generate_batched(
    df: pd.DataFrame,
    batch_size: int,
    model: str,
    inference_url: str,
    api_key: str,
    sys_prompt_col=None,
    input_col="input",
    output_col="output",
    temperature=0.7,
    max_tokens=1024,
    top_p=1.0,
) -> pd.DataFrame:
    """
    Process dataset in batches for inference with metrics collection.
    
    Args:
        df: Input DataFrame
        batch_size: Number of samples per batch
        model: Model name/identifier
        inference_url: API endpoint for inference
        api_key: API key for authentication
        sys_prompt_col: Column name for system prompt (optional)
        input_col: Column name for input text
        output_col: Column name for output text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        
    Returns:
        Updated DataFrame with generated outputs and metrics
    """
    import aiohttp
    import asyncio
    import time
    from aiohttp import ClientTimeout
    
    # Create a timeout object (values in seconds)
    timeout = ClientTimeout(total=420)
    
    def get_prompt(content):
        return [{"role": "user", "content": content}]
    
    async def predict(
        session,
        prompt,
        sys_prompt=None,
    ):
        if sys_prompt is not None:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = get_prompt(prompt)
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        })
        
        start_time = time.monotonic()
        async with session.post(
            inference_url,
            headers=headers,
            data=payload,
            timeout=timeout
        ) as response:
            first_token_time = None
            content_bytes = bytearray()
            async for chunk in response.content.iter_chunked(max_tokens):
                if first_token_time is None:
                    first_token_time = time.monotonic()
                content_bytes.extend(chunk)
            end_time = time.monotonic()
        
        try:
            response_json = json.loads(content_bytes.decode())
            output = response_json["choices"][0]["message"]["content"]
            prompt_tokens = response_json.get("usage", {}).get("prompt_tokens")
            completion_tokens = response_json.get("usage", {}).get("completion_tokens")
            total_tokens = response_json.get("usage", {}).get("total_tokens")
            
            metrics = {
                "time_to_first_token": first_token_time - start_time if first_token_time else None,
                "time_total": end_time - start_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": (
                    total_tokens / (end_time - start_time)
                    if total_tokens and (end_time - start_time) > 0
                    else None
                ),
            }
            return output, metrics
            
        except Exception as e:
            lab.log(f"Exception during inference: {e}")
            return "", {}
    
    async def process_batch(batch):
        prompts = batch[input_col].values
        if sys_prompt_col is not None:
            sys_prompts = batch[sys_prompt_col].values
            tasks = [
                predict(session, prompt, sys_prompt=sys_prompt)
                for prompt, sys_prompt in zip(prompts, sys_prompts)
            ]
        else:
            tasks = [predict(session, prompt) for prompt in prompts]
        
        results = await asyncio.gather(*tasks)
        return results
    
    # Process dataset in batches
    max_idx = len(df)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(0, max_idx, batch_size):
            end = min(start + batch_size, max_idx)
            batch = df.iloc[start:end]
            
            results = await process_batch(batch)
            
            for idx, result in enumerate(results):
                global_idx = start + idx
                df.loc[global_idx, output_col] = result[0]
                df.loc[global_idx, "time_to_first_token"] = result[1].get("time_to_first_token", None)
                df.loc[global_idx, "time_total"] = result[1].get("time_total", None)
                df.loc[global_idx, "prompt_tokens"] = result[1].get("prompt_tokens", None)
                df.loc[global_idx, "completion_tokens"] = result[1].get("completion_tokens", None)
                df.loc[global_idx, "total_tokens"] = result[1].get("total_tokens", None)
                df.loc[global_idx, "tokens_per_second"] = result[1].get("tokens_per_second", None)
            
            # Update progress based on batch completion
            progress = 40 + int((end / max_idx) * 40)  # 40-80% range for inference
            lab.update_progress(progress)
            lab.log(f"Processed batch {start}-{end}/{max_idx}")
    
    return df


async def run_evaluation():
    """Run the inference evaluation with metrics tracking."""
    
    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()
        
        # Get parameters from task configuration
        config = lab.get_config()
        
        # Extract parameters with defaults
        generation_model = config.get("generation_model", "local")
        dataset_name = config.get("dataset", "Trelis/touch-rugby-rules")
        input_column = config.get("input_column", "text")
        output_column = config.get("output_column", "generated_output")
        system_prompt = config.get("system_prompt", "")
        output_dir = config.get("output_dir", "./output")
        
        # Convert string values to appropriate types
        batch_size = int(config.get("batch_size", 128))
        temperature = float(config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens", 1024))
        top_p = float(config.get("top_p", 1.0))
        
        # Parse tasks parameter (can be JSON string or list)
        tasks_param = config.get("tasks", ["Time to First Token (TTFT)", "Total Time", "Tokens per Second"])
        if isinstance(tasks_param, str):
            try:
                tasks = json.loads(tasks_param)
                if not isinstance(tasks, list):
                    raise ValueError("Tasks should be a list of task names.")
            except json.JSONDecodeError:
                # Fallback to comma-separated string
                tasks = [t.strip() for t in tasks_param.split(",")]
        else:
            tasks = tasks_param
        
        # Log start time and configuration
        start_time = datetime.now()
        lab.log(f"Evaluation started at {start_time}")
        lab.log(f"Generation model: {generation_model}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"Input column: {input_column}")
        lab.log(f"Output column: {output_column}")
        lab.log(f"Batch size: {batch_size}")
        lab.log(f"Temperature: {temperature}")
        lab.log(f"Max tokens: {max_tokens}")
        lab.log(f"Metrics to evaluate: {tasks}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        lab.update_progress(10)
        
        # Load the evaluation model
        lab.log("Loading evaluation model...")
        try:
            # Use lab's model provider system
            from transformerlab.sdk.v1.evals import ModelProvider
            
            # Create model provider instance
            model_provider = ModelProvider(generation_model)
            
            # Get model details
            model_name = model_provider.generation_model_name
            inference_url = model_provider.chat_completions_url
            api_key = model_provider.api_key
            
            lab.log(f"Model loaded: {model_name}")
            lab.log(f"Inference URL: {inference_url}")
            
        except Exception as e:
            lab.log(f"Error loading model: {e}")
            lab.error("Evaluation failed - model loading error")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(20)
        
        # Load dataset
        lab.log("Loading dataset...")
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name)
            df = dataset["train"].to_pandas()
            
            lab.log(f"Loaded dataset with {len(df)} examples")
            
            # Verify required columns exist
            if input_column not in df.columns:
                raise ValueError(f"Input column '{input_column}' not found in dataset")
            
        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            lab.error("Evaluation failed - dataset loading error")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(30)
        
        # Add system prompt column if specified
        sys_prompt_col = None
        if system_prompt and system_prompt.strip():
            df["system_prompt"] = system_prompt
            sys_prompt_col = "system_prompt"
            lab.log(f"Using system prompt: {system_prompt}")
        
        # Run batch generation with metrics
        lab.log("Starting batch generation with metrics collection...")
        try:
            df = await generate_batched(
                df=df,
                batch_size=batch_size,
                model=model_name,
                inference_url=inference_url,
                api_key=api_key,
                sys_prompt_col=sys_prompt_col,
                input_col=input_column,
                output_col=output_column,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            lab.log("Batched generation completed successfully")
            
        except Exception as e:
            lab.log(f"Error during batch generation: {e}")
            import traceback
            traceback.print_exc()
            lab.error("Evaluation failed - batch generation error")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(80)
        
        # Process and save metrics
        lab.log("Processing metrics...")
        metrics_list = []
        
        for metric_name in tasks:
            if metric_name not in METRICS_MAP:
                lab.log(f"Warning: Unknown metric '{metric_name}', skipping")
                continue
            
            metric_col = METRICS_MAP[metric_name]
            if metric_col not in df.columns:
                lab.log(f"Warning: Metric column '{metric_col}' not found in results, skipping")
                continue
            
            # Calculate average metric value
            metric_values = df[metric_col].dropna()
            if len(metric_values) > 0:
                metric_avg = metric_values.mean()
                lab.log(f"{metric_name}: {metric_avg:.4f}")
            else:
                metric_avg = 0.0
                lab.log(f"{metric_name}: No valid values")
        
        # Create detailed metrics DataFrame for saving
        for idx, row in df.iterrows():
            for metric_name in tasks:
                if metric_name not in METRICS_MAP:
                    continue
                
                metric_col = METRICS_MAP[metric_name]
                if metric_col not in df.columns:
                    continue
                
                metric_value = row[metric_col]
                if metric_value is not None and pd.notna(metric_value):
                    score = round(float(metric_value), 4)
                else:
                    score = 0.0
                
                metrics_list.append({
                    "test_case_id": f"test_case_{idx}",
                    "metric_name": metric_name,
                    "score": score,
                    "input": row[input_column],
                    "output": row[output_column] if output_column in row and pd.notna(row[output_column]) else "",
                })
        
        lab.update_progress(90)
        
        # Save metrics as eval artifact
        try:
            metrics_df = pd.DataFrame(metrics_list)
            
            # Save as CSV artifact with eval type
            saved_metrics_path = lab.save_artifact(
                metrics_df,
                name="inference_metrics.csv",
                type="eval",
                config={
                    "evals": {
                        "input": "input",
                        "output": "output",
                        "score": "score",
                    }
                }
            )
            
            lab.log(f"✅ Metrics saved as eval artifact: {saved_metrics_path}")
            
        except Exception as e:
            lab.log(f"Warning: Could not save metrics as eval artifact: {e}")
            # Fallback: save as regular artifact
            metrics_file = os.path.join(output_dir, "metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            saved_metrics_path = lab.save_artifact(metrics_file, "metrics.csv")
            lab.log(f"✅ Metrics saved to: {saved_metrics_path}")
        
        # Save evaluation summary
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "evaluation_type": "Inference Metrics",
                "model": generation_model,
                "dataset": dataset_name,
                "num_examples": len(df),
                "metrics": tasks,
                "batch_size": batch_size,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "completed_at": datetime.now().isoformat(),
            }, f, indent=2)
        
        summary_path = lab.save_artifact(summary_file, "evaluation_summary.json")
        lab.log(f"✅ Evaluation summary saved: {summary_path}")
        
        # Calculate evaluation time
        end_time = datetime.now()
        eval_duration = end_time - start_time
        lab.log(f"Evaluation completed in {eval_duration}")
        
        lab.update_progress(100)
        
        # Finish the job
        lab.finish("Evaluation completed successfully!")
        
        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(eval_duration),
            "output_dir": output_dir,
            "metrics_path": saved_metrics_path,
            "num_examples": len(df),
            "metrics_evaluated": tasks,
        }
        
    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}
    
    except Exception as e:
        error_msg = str(e)
        print(f"Evaluation failed: {error_msg}")
        
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    import asyncio
    
    print("🚀 Starting inference metrics evaluation...")
    result = asyncio.run(run_evaluation())
    print("Evaluation result:", result)
