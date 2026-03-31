import os
import time
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import psutil
import GPUtil

from lab import lab

# Login to huggingface if token is provided
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# Check if accelerate is available for device_map
try:
    import accelerate
    has_accelerate = True
except ImportError:
    has_accelerate = False


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed, gpus[0].memoryTotal
        return None, None
    except Exception:
        return None, None


class TimingStreamer(TextStreamer):
    """Custom streamer to capture token generation times"""
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.token_times = []
        self.start_time = None
        self.first_token_time = None

    def on_finalized_text(self, text, stream_end=False):
        if self.start_time is None:
            self.start_time = time.time()
        current_time = time.time()
        if self.first_token_time is None:
            self.first_token_time = current_time
        self.token_times.append(current_time)
        super().on_finalized_text(text, stream_end)


def run_inference_metrics():
    """Run comprehensive LLM inference and performance metrics"""
    
    try:
        # Initialize lab
        lab.init()
        
        # Get parameters from task configuration
        config = lab.get_config()
        
        # Extract parameters with defaults
        model_name = config.get("model_name", "HuggingFaceTB/SmolLM-135M-Instruct")
        sample_prompts = config.get("sample_prompts", [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about AI."
        ])
        max_new_tokens = int(config.get("max_new_tokens", 50))
        num_runs = int(config.get("num_runs", 5))  # Number of inference runs for averaging
        device = config.get("device", "auto")
        concurrency = int(config.get("concurrency", 1))  # Number of concurrent requests
        
        # Log start
        start_time = datetime.now()
        lab.log(f"Inference metrics started at {start_time}")
        lab.log(f"Model: {model_name}")
        lab.log(f"Device: {device}")
        lab.log(f"Sample prompts: {len(sample_prompts)}")
        lab.log(f"Max new tokens: {max_new_tokens}")
        lab.log(f"Number of runs: {num_runs}")
        lab.log(f"Concurrency: {concurrency}")
        
        lab.update_progress(10)
        
        # Load model and tokenizer
        lab.log("Loading model and tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Use device_map only if accelerate is available and device is "auto"
            device_map = "auto" if (device == "auto" and has_accelerate) else None
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map
            )
            
            # If not using device_map, move model to appropriate device
            if device_map is None:
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            
            lab.log("✅ Model loaded successfully")
            
        except Exception as e:
            lab.log(f"❌ Failed to load model: {e}")
            lab.error(f"Model loading failed: {str(e)}")
            return {"status": "error", "error": str(e)}
        
        lab.update_progress(30)
        
        # Prepare sample prompts
        if isinstance(sample_prompts, str):
            try:
                sample_prompts = json.loads(sample_prompts)
            except Exception:
                sample_prompts = [sample_prompts]
        
        # Initialize metrics collection
        all_ttfts = []
        all_e2e_latencies = []
        all_itls = []
        all_token_counts = []
        all_tps_per_request = []
        memory_usages = []
        gpu_memory_usages = []
        per_prompt_metrics = []
        
        lab.log("Running inference measurements...")
        
        for run in range(num_runs):
            lab.log(f"Run {run + 1}/{num_runs}")
            
            for prompt in sample_prompts:
                # Prepare inputs
                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Record memory before inference
                mem_before = get_memory_usage()
                gpu_mem_before, _ = get_gpu_memory_usage()
                
                # Create timing streamer
                streamer = TimingStreamer(tokenizer, skip_prompt=True)
                
                # Run inference with timing
                inference_start = time.time()
                
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,  # Deterministic for consistent measurements
                            pad_token_id=tokenizer.eos_token_id,
                            streamer=streamer,
                            use_cache=True
                        )
                    
                    inference_end = time.time()
                    
                    # Extract generated text
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated_text[len(prompt):].strip()
                    
                    # Calculate metrics
                    ttft = streamer.first_token_time - inference_start if streamer.first_token_time else None
                    e2e_latency = inference_end - inference_start
                    
                    # Calculate inter-token latencies
                    if len(streamer.token_times) > 1:
                        itls = [streamer.token_times[i] - streamer.token_times[i-1] for i in range(1, len(streamer.token_times))]
                        avg_itl = sum(itls) / len(itls)
                    else:
                        itls = []
                        avg_itl = 0
                    
                    token_count = len(tokenizer.encode(response, add_special_tokens=False))
                    
                    # TPS per request
                    tps_per_request = token_count / e2e_latency if e2e_latency > 0 else 0
                    
                    # Store metrics
                    if ttft is not None:
                        all_ttfts.append(ttft)
                    all_e2e_latencies.append(e2e_latency)
                    if itls:
                        all_itls.extend(itls)
                    all_token_counts.append(token_count)
                    all_tps_per_request.append(tps_per_request)
                    
                    # Record memory after inference
                    mem_after = get_memory_usage()
                    gpu_mem_after, gpu_mem_total = get_gpu_memory_usage()
                    
                    memory_usages.append(mem_after - mem_before)
                    if gpu_mem_after is not None and gpu_mem_before is not None:
                        gpu_memory_usages.append(gpu_mem_after - gpu_mem_before)
                    
                    # Store per-prompt metrics
                    per_prompt_metrics.append({
                        "run": run + 1,
                        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        "ttft_seconds": ttft,
                        "e2e_latency_seconds": e2e_latency,
                        "avg_itl_seconds": avg_itl,
                        "tokens_generated": token_count,
                        "tps_per_request": tps_per_request,
                        "memory_delta_mb": mem_after - mem_before,
                        "gpu_memory_delta_mb": gpu_mem_after - gpu_mem_before if gpu_mem_after is not None else None
                    })
                    
                    lab.log(f"  Prompt: '{prompt[:50]}...' -> {token_count} tokens, TTFT: {ttft:.3f}s, E2E: {e2e_latency:.3f}s, TPS: {tps_per_request:.2f}")
                    
                except Exception as e:
                    lab.log(f"❌ Inference failed for prompt '{prompt}': {e}")
                    continue
            
            lab.update_progress(30 + (run + 1) * 50 // num_runs)
        
        # Calculate aggregate metrics
        if all_ttfts and all_e2e_latencies and all_token_counts:
            avg_ttft = sum(all_ttfts) / len(all_ttfts)
            min_ttft = min(all_ttfts)
            max_ttft = max(all_ttfts)
            
            avg_e2e_latency = sum(all_e2e_latencies) / len(all_e2e_latencies)
            min_e2e_latency = min(all_e2e_latencies)
            max_e2e_latency = max(all_e2e_latencies)
            
            avg_itl = sum(all_itls) / len(all_itls) if all_itls else 0
            min_itl = min(all_itls) if all_itls else 0
            max_itl = max(all_itls) if all_itls else 0
            
            avg_tokens = sum(all_token_counts) / len(all_token_counts)
            total_tokens = sum(all_token_counts)
            
            avg_tps_per_request = sum(all_tps_per_request) / len(all_tps_per_request)
            
            # System-level TPS (total tokens per second across all requests)
            total_time = sum(all_e2e_latencies)
            system_tps = total_tokens / total_time if total_time > 0 else 0
            
            # RPS (requests per second)
            total_requests = len(all_e2e_latencies)
            system_rps = total_requests / total_time if total_time > 0 else 0
            
            avg_memory_delta = sum(memory_usages) / len(memory_usages) if memory_usages else 0
            
            # Model size
            model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Rough estimate in MB
            
            # GPU memory usage
            avg_gpu_memory_delta = sum(gpu_memory_usages) / len(gpu_memory_usages) if gpu_memory_usages else 0
            
            metrics = {
                "average_ttft_seconds": avg_ttft,
                "min_ttft_seconds": min_ttft,
                "max_ttft_seconds": max_ttft,
                "average_e2e_latency_seconds": avg_e2e_latency,
                "min_e2e_latency_seconds": min_e2e_latency,
                "max_e2e_latency_seconds": max_e2e_latency,
                "average_itl_seconds": avg_itl,
                "min_itl_seconds": min_itl,
                "max_itl_seconds": max_itl,
                "average_tokens_per_response": avg_tokens,
                "total_tokens_generated": total_tokens,
                "average_tps_per_request": avg_tps_per_request,
                "system_tps": system_tps,
                "system_rps": system_rps,
                "average_memory_delta_mb": avg_memory_delta,
                "average_gpu_memory_delta_mb": avg_gpu_memory_delta,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs,
                "num_prompts": len(sample_prompts),
                "total_measurements": len(all_ttfts),
                "concurrency": concurrency
            }
            
            lab.log("📊 Inference Metrics:")
            lab.log(f"  Average TTFT: {avg_ttft:.3f}s")
            lab.log(f"  Average E2E Latency: {avg_e2e_latency:.3f}s")
            lab.log(f"  Average ITL: {avg_itl:.3f}s")
            lab.log(f"  System TPS: {system_tps:.2f}")
            lab.log(f"  System RPS: {system_rps:.2f}")
            lab.log(f"  Average TPS per Request: {avg_tps_per_request:.2f}")
            lab.log(f"  Model Size: {model_size_mb:.1f} MB")
            lab.log(f"  Memory Delta: {avg_memory_delta:.1f} MB")
            if avg_gpu_memory_delta > 0:
                lab.log(f"  GPU Memory Delta: {avg_gpu_memory_delta:.1f} MB")
        
        else:
            lab.log("⚠️ No successful inferences to calculate metrics")
            metrics = {"error": "No successful inferences"}
        
        lab.update_progress(90)
        
        # Save metrics as artifact
        end_time = datetime.now()
        duration = end_time - start_time
        
        results = {
            "metrics": metrics,
            "per_prompt_metrics": per_prompt_metrics,
            "config": {
                "model_name": model_name,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "num_runs": num_runs,
                "num_prompts": len(sample_prompts),
                "concurrency": concurrency
            },
            "timestamp": end_time.isoformat(),
            "duration": str(duration)
        }
        
        # Save as JSON artifact
        metrics_file = "inference_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        
        saved_path = lab.save_artifact(metrics_file, "inference_metrics.json")
        lab.log(f"✅ Saved metrics: {saved_path}")
        
        lab.update_progress(100)
        
        # Complete the job
        lab.finish("Inference metrics completed successfully!")
        
        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(duration),
            "metrics": metrics,
            "saved_metrics_path": saved_path
        }
    
    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id if hasattr(lab, "job") else None}
    
    except Exception as e:
        error_msg = str(e)
        lab.log(f"❌ Inference metrics failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id if hasattr(lab, "job") else None, "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting LLM Inference Metrics...")
    result = run_inference_metrics()
    print("Result:", result)
