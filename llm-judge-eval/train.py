#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation Script using DeepEval with TransformerLab integration.

This script demonstrates:
- Using lab.get_config() to read parameters from task configuration
- Using DeepEval for LLM-as-Judge evaluation
- Saving evaluation results as artifacts
- Supporting both predefined metrics and custom GEval metrics
"""

import os
import json
import importlib
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

# Import DeepEval dependencies
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from lab import lab


def get_metric_class(metric_name: str):
    """
    Import the metric class based on the metric name
    :param metric_name: Name of the metric
    :return: Metric class
    """
    module = importlib.import_module("deepeval.metrics")
    try:
        metric_class = getattr(module, metric_name)
        return metric_class
    except AttributeError:
        print(f"Metric {metric_name} not found in deepeval.metrics")
        sys.exit(1)


def run_evaluation():
    """Run DeepEval metrics for LLM-as-judge evaluation"""
    
    # Configure GPU usage - use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Initialize lab (auto-loads parameters from job_data if available)
        lab.init()

        # Get parameters from task configuration
        config = lab.get_config()

        # Extract parameters with defaults
        dataset_name = config.get("dataset", None)
        dataset_split = config.get("dataset_split", "train")
        generation_model = config.get("generation_model", "HuggingFaceTB/SmolLM2-135M")
        predefined_tasks_raw = config.get("predefined_tasks", '[\"Toxicity\"]')
        tasks_raw = config.get("tasks", "[]")
        limit = float(config.get("limit", 1.0))
        threshold = float(config.get("threshold", 0.5))
        output_dir = config.get("output_dir", "./output")

        # Log start time
        start_time = datetime.now()
        lab.log(f"Evaluation started at {start_time}")
        lab.log(f"Judge Model: {generation_model}")
        lab.log(f"Dataset: {dataset_name}")
        lab.log(f"Dataset Split: {dataset_split}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        lab.update_progress(5)

        # Parse metrics and tasks
        if isinstance(predefined_tasks_raw, str):
            try:
                predefined_tasks = json.loads(predefined_tasks_raw)
            except json.JSONDecodeError:
                lab.log(f"Invalid JSON format for predefined tasks: {predefined_tasks_raw}")
                predefined_tasks = (
                    predefined_tasks_raw.split(",") if predefined_tasks_raw else []
                )
        else:
            predefined_tasks = predefined_tasks_raw if predefined_tasks_raw else []

        if len(predefined_tasks) == 0:
            lab.log("No valid predefined tasks found.")
        
        formatted_predefined_tasks = [task.strip().replace(" ", "") + "Metric" for task in predefined_tasks]

        try:
            geval_tasks = json.loads(tasks_raw) if tasks_raw and tasks_raw != "[]" else []
        except Exception as e:
            lab.log(f"Error parsing tasks JSON: {e}")
            geval_tasks = []

        lab.log(f"Predefined metrics: {formatted_predefined_tasks}")
        lab.log(f"Custom GEval tasks: {len(geval_tasks)}")

        # Classification of metrics
        two_input_metrics = ["AnswerRelevancyMetric", "BiasMetric", "ToxicityMetric"]
        three_input_metrics = [
            "FaithfulnessMetric",
            "ContextualPrecisionMetric",
            "ContextualRecallMetric",
            "ContextualRelevancyMetric",
            "HallucinationMetric",
        ]

        # Analyze custom metrics requirements
        three_input_custom_metric = []
        two_input_custom_metric = []

        for task in geval_tasks:
            if task.get("include_context") == "Yes":
                three_input_custom_metric.append(task["name"])
            else:
                two_input_custom_metric.append(task["name"])

        lab.update_progress(10)

        # Load the model for evaluation
        try:
            from deepeval.models import DeepEvalBaseLLM

            # Check if using a local model or API-based model
            if generation_model.lower().startswith("gpt-") or "openai" in generation_model.lower():
                # Use OpenAI models
                from langchain_openai import ChatOpenAI
                trlab_model = ChatOpenAI(model=generation_model)
                lab.log(f"Using OpenAI model: {generation_model}")
            elif "claude" in generation_model.lower() or "anthropic" in generation_model.lower():
                # Use Anthropic models
                from langchain_anthropic import ChatAnthropic
                trlab_model = ChatAnthropic(model=generation_model)
                lab.log(f"Using Anthropic model: {generation_model}")
            else:
                # Assume it's a local HuggingFace model
                lab.log(f"Loading local model: {generation_model}")
                
                # Import PyTorch dependencies only when needed for local models
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                except ImportError as import_err:
                    error_msg = (
                        "PyTorch and Transformers are required for local model inference. "
                        "Please install them using: pip install torch transformers"
                    )
                    lab.log(f"Import error: {import_err}")
                    raise ImportError(error_msg)
                
                # Check if accelerate is available for device_map support
                try:
                    import accelerate
                    has_accelerate = True
                except ImportError:
                    has_accelerate = False
                    lab.log("Note: accelerate not installed, using default device placement")
                
                # Create a custom DeepEval model wrapper
                class LocalLLM(DeepEvalBaseLLM):
                    def __init__(self, model_name):
                        self.model_name = model_name
                        
                        # Prepare model loading arguments
                        model_kwargs = {
                            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        }
                        
                        # Only use device_map if accelerate is available
                        if has_accelerate:
                            model_kwargs["device_map"] = "auto"
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs
                        )
                        
                        # Move model to device if accelerate is not available
                        if not has_accelerate:
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            self.model = self.model.to(device)
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token

                    def load_model(self):
                        return self.model

                    def generate(self, prompt: str) -> str:
                        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=512,
                                temperature=0.7,
                                do_sample=True
                            )
                        
                        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the prompt from the response
                        response = response[len(prompt):].strip()
                        return response

                    async def a_generate(self, prompt: str) -> str:
                        return self.generate(prompt)

                    def get_model_name(self):
                        return self.model_name

                trlab_model = LocalLLM(generation_model)
                lab.log("Local model loaded successfully")

        except Exception as e:
            lab.log(f"An error occurred while loading the model: {e}")
            traceback.print_exc()
            lab.error(f"Failed to load model: {str(e)}")
            return {"status": "error", "error": str(e)}

        lab.update_progress(20)

        # Load the dataset
        try:
            if dataset_name:
                from datasets import load_dataset
                dataset_dict = load_dataset(dataset_name)
                df = dataset_dict[dataset_split].to_pandas()
                lab.log(f"Dataset loaded successfully: {len(df)} examples")
            else:
                lab.log("⚠️ No dataset specified. Creating sample dataset with 5 examples...")
                # Create a sample dataset with 5 datapoints
                sample_data = {
                    "input": [
                        "What is the capital of France?",
                        "Explain quantum computing",
                        "What is the largest planet in our solar system?",
                        "Define machine learning in simple terms",
                        "What is the chemical symbol for gold?"
                    ],
                    "output": [
                        "Paris is the capital of France.",
                        "Quantum computing uses quantum mechanics principles to process information.",
                        "Jupiter is the largest planet in our solar system.",
                        "Machine learning is a type of artificial intelligence that learns from data.",
                        "The chemical symbol for gold is Au."
                    ],
                    "expected_output": [
                        "Paris",
                        "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement.",
                        "Jupiter",
                        "Machine learning enables computers to learn from examples without explicit programming.",
                        "Au"
                    ]
                }
                df = pd.DataFrame(sample_data)
                lab.log(f"✅ Created sample dataset with {len(df)} examples")
        except Exception as e:
            lab.log(f"Error loading dataset: {e}")
            traceback.print_exc()
            lab.error(f"Failed to load dataset: {str(e)}")
            return {"status": "error", "error": str(e)}

        lab.update_progress(30)

        # Verify required columns exist
        required_columns = ["input", "output", "expected_output"]
        if not all(col in df.columns for col in required_columns):
            error_msg = (
                "The dataset should have the columns `input`, `output` and `expected_output`. "
                f"Current columns: {df.columns.tolist()}"
            )
            lab.log(error_msg)
            lab.error(error_msg)
            return {"status": "error", "error": error_msg}

        # Check context requirements
        if any(elem in three_input_metrics for elem in formatted_predefined_tasks) or len(three_input_custom_metric) > 0:
            if "context" not in df.columns:
                lab.log("Using expected_output column as the context")
                df["context"] = df["expected_output"]

            # Verify non-null values
            if not df["context"].notnull().all():
                error_msg = (
                    f"The dataset should have all non-null values in the 'context' column for metrics: "
                    f"{formatted_predefined_tasks + three_input_custom_metric}"
                )
                lab.log(error_msg)
                lab.error(error_msg)
                return {"status": "error", "error": error_msg}

        # Verify non-null values in required columns
        for col in required_columns:
            if not df[col].notnull().all():
                error_msg = f"The dataset should have all non-null values in the '{col}' column"
                lab.log(error_msg)
                lab.error(error_msg)
                return {"status": "error", "error": error_msg}

        lab.update_progress(40)

        # Initialize metrics
        metrics_arr = []
        try:
            # Initialize predefined metrics
            for met in formatted_predefined_tasks:
                lab.log(f"Initializing metric: {met}")
                metric_class = get_metric_class(met)
                metric = metric_class(
                    model=trlab_model, 
                    threshold=threshold, 
                    include_reason=True
                )
                metrics_arr.append(metric)

            # Initialize custom GEval metrics
            for met in geval_tasks:
                lab.log(f"Initializing custom GEval metric: {met['name']}")
                evaluation_params = [
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ]
                if met.get("include_context") == "Yes":
                    evaluation_params.append(LLMTestCaseParams.CONTEXT)

                evaluation_steps = None

                if isinstance(met.get("evaluation_steps"), str):
                    try:
                        met["evaluation_steps"] = json.loads(met["evaluation_steps"])
                    except json.JSONDecodeError:
                        lab.log(
                            f"Invalid JSON format for evaluation steps: {met.get('evaluation_steps')}. Using description field only."
                        )

                if isinstance(met.get("evaluation_steps"), list):
                    evaluation_steps = met["evaluation_steps"]
                    if len(evaluation_steps) == 0:
                        evaluation_steps = None
                    elif len(evaluation_steps) > 0 and evaluation_steps[0] == "":
                        evaluation_steps = None

                if evaluation_steps is not None:
                    lab.log(f"Using evaluation steps: {evaluation_steps}")
                    metric = GEval(
                        name=met["name"],
                        evaluation_steps=evaluation_steps,
                        evaluation_params=evaluation_params,
                        model=trlab_model,
                    )
                else:
                    lab.log("No evaluation steps provided, using description.")
                    metric = GEval(
                        name=met["name"],
                        criteria=met.get("description", ""),
                        evaluation_params=evaluation_params,
                        model=trlab_model,
                    )

                metrics_arr.append(metric)
            
            lab.log(f"Metrics loaded successfully: {len(metrics_arr)} metrics")
        except Exception as e:
            lab.log(f"An error occurred while loading the metrics: {e}")
            traceback.print_exc()
            lab.error(f"Failed to initialize metrics: {str(e)}")
            return {"status": "error", "error": str(e)}

        lab.update_progress(50)

        # Create test cases
        test_cases = []
        try:
            if (
                all(elem in two_input_metrics for elem in formatted_predefined_tasks)
                and len(three_input_custom_metric) == 0
            ):
                # Two-input test cases (single-turn)
                lab.log("Creating two-input test cases")
                for _, row in df.iterrows():
                    test_cases.append(
                        LLMTestCase(
                            input=row["input"], 
                            actual_output=row["output"], 
                            expected_output=row["expected_output"]
                        )
                    )
            elif (
                any(elem in three_input_metrics for elem in formatted_predefined_tasks)
                or len(three_input_custom_metric) > 0
            ):
                # Three-input test cases
                lab.log("Creating three-input test cases")
                if "HallucinationMetric" not in formatted_predefined_tasks:
                    for _, row in df.iterrows():
                        if isinstance(row["context"], list):
                            context = row["context"]
                        elif isinstance(row["context"], np.ndarray):
                            context = row["context"].tolist()
                        elif (
                            isinstance(row["context"], str)
                            and row["context"].startswith("[")
                            and row["context"].endswith("]")
                        ):
                            try:
                                context = eval(row["context"])
                            except Exception:
                                context = [row["context"]]
                        else:
                            context = [row["context"]]
                        test_cases.append(
                            LLMTestCase(
                                input=row["input"],
                                actual_output=row["output"],
                                expected_output=row["expected_output"],
                                retrieval_context=context,
                            )
                        )
                else:
                    # Special case for HallucinationMetric
                    for _, row in df.iterrows():
                        if isinstance(row["context"], list):
                            context = row["context"]
                        elif (
                            isinstance(row["context"], str)
                            and row["context"].startswith("[")
                            and row["context"].endswith("]")
                        ):
                            try:
                                context = eval(row["context"])
                            except Exception:
                                context = [row["context"]]
                        else:
                            context = [row["context"]]
                        test_cases.append(
                            LLMTestCase(
                                input=row["input"],
                                actual_output=row["output"],
                                expected_output=row["expected_output"],
                                retrieval_context=context,
                            )
                        )
        except Exception as e:
            lab.log(f"An error occurred while creating test cases: {e}")
            traceback.print_exc()
            lab.error(f"Failed to create test cases: {str(e)}")
            return {"status": "error", "error": str(e)}

        # Apply limit if specified
        if limit and float(limit) != 1.0:
            num_samples = max(int(len(test_cases) * float(limit)), 1)
            test_cases = test_cases[:num_samples]
            lab.log(f"Limited to {num_samples} test cases (limit={limit})")

        lab.log(f"Test cases created: {len(test_cases)}")
        lab.update_progress(60)

        # Create evaluation dataset and run evaluation
        dataset = EvaluationDataset()
        for test_case in test_cases:
            dataset.add_test_case(test_case)        
        try:
            # Set the plugin to use sync mode if on macOS
            # as MLX doesn't support async mode currently
            async_mode = True
            if "local" in generation_model.lower() or not any(
                provider in generation_model.lower() 
                for provider in ["gpt-", "claude", "openai", "anthropic"]
            ):
                async_mode = sys.platform != "darwin"
            
            lab.log(f"Running evaluation with async_mode={async_mode}")
            
            # Run the evaluation
            async_config = AsyncConfig(run_async=async_mode)
            output = evaluate(dataset.test_cases, metrics_arr, async_config=async_config)
            
            lab.update_progress(85)

            # Process results
            metrics_data = []
            for test_case in output.test_results:
                for metric in test_case.metrics_data:
                    metrics_data.append(
                        {
                            "test_case_id": test_case.name,
                            "metric_name": metric.name,
                            "score": metric.score,
                            "input": test_case.input,
                            "output": test_case.actual_output,
                            "expected_output": test_case.expected_output,
                            "reason": metric.reason if hasattr(metric, 'reason') else "",
                        }
                    )

            # Create metrics DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            lab.log(f"Evaluation completed: {len(metrics_data)} results")

            # Save evaluation results as artifacts
            metrics_csv_path = os.path.join(output_dir, "eval_results.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            
            # Save as eval artifact with proper config
            saved_metrics_path = lab.save_artifact(
                metrics_df,
                name="llm_judge_eval_results.csv",
                type="eval",
                config={
                    "evals": {
                        "input": "input",
                        "output": "output",
                        "expected_output": "expected_output",
                        "score": "score",
                    }
                },
            )
            lab.log(f"✅ Saved evaluation results as artifact: {saved_metrics_path}")

            # Calculate and log aggregated metrics
            aggregated_metrics = []
            for metric in metrics_df["metric_name"].unique():
                avg_score = metrics_df[metrics_df["metric_name"] == metric]["score"].mean()
                aggregated_metrics.append({
                    "metric_name": metric,
                    "average_score": avg_score
                })
                lab.log(f"📊 {metric}: {avg_score:.4f}")

            # Save aggregated metrics
            agg_df = pd.DataFrame(aggregated_metrics)
            agg_csv_path = os.path.join(output_dir, "aggregated_metrics.csv")
            agg_df.to_csv(agg_csv_path, index=False)
            
            agg_artifact_path = lab.save_artifact(agg_csv_path, "aggregated_metrics.csv")
            lab.log(f"✅ Saved aggregated metrics: {agg_artifact_path}")

            lab.update_progress(95)

        except Exception as e:
            traceback.print_exc()
            lab.log(f"An error occurred during evaluation: {e}")
            lab.error(f"Evaluation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

        # Calculate evaluation time
        end_time = datetime.now()
        eval_duration = end_time - start_time
        lab.log(f"Evaluation completed in {eval_duration}")

        # Save final summary
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        summary = {
            "evaluation_type": "LLM-as-Judge (DeepEval)",
            "judge_model": generation_model,
            "dataset": dataset_name,
            "dataset_split": dataset_split,
            "num_test_cases": len(test_cases),
            "metrics": formatted_predefined_tasks + [t["name"] for t in geval_tasks],
            "aggregated_metrics": aggregated_metrics,
            "duration": str(eval_duration),
            "completed_at": end_time.isoformat(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        summary_artifact_path = lab.save_artifact(summary_file, "evaluation_summary.json")
        lab.log(f"✅ Saved evaluation summary: {summary_artifact_path}")

        lab.update_progress(100)

        # Complete the job
        lab.finish("Evaluation completed successfully!")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(eval_duration),
            "output_dir": output_dir,
            "num_test_cases": len(test_cases),
            "metrics": len(metrics_arr),
            "gpu_used": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        }

    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        print(f"Evaluation failed: {error_msg}")
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    print("🚀 Starting LLM-as-Judge evaluation...")
    result = run_evaluation()
    print("Evaluation result:", result)
