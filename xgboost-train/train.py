#!/usr/bin/env python3
"""
XGBoost Training Script with TransformerLab Integration

This script demonstrates:
- Using lab.init() to initialize TransformerLab SDK
- Using lab.get_config() to read parameters from task configuration
- Using lab.update_progress() for real-time progress tracking
- Using lab.log() for structured logging
- Using lab.save_model() to save trained models
- Automatic checkpoint resumption with lab.get_checkpoint_to_resume()
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

from lab import lab

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_dataset(dataset_name):
    """Load dataset from scikit-learn or create sample data"""
    lab.log(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        lab.log(f"✅ Loaded breast cancer dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
    elif dataset_name == "iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        lab.log(f"✅ Loaded iris dataset with {X.shape[0]} samples and {X.shape[1]} features")
        
    elif dataset_name == "synthetic":
        # Create synthetic dataset for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series((X.iloc[:, 0] + X.iloc[:, 1] > 0).astype(int), name="target")
        lab.log(f"✅ Created synthetic dataset with {X.shape[0]} samples and {X.shape[1]} features")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y


def train_xgboost_classifier(X_train, y_train, X_val, y_val, config):
    """Train XGBoost classifier with specified parameters"""
    lab.log("Setting up XGBoost classifier...")
    
    # Extract hyperparameters from config
    max_depth = config.get("max_depth", 6)
    learning_rate = config.get("learning_rate", 0.1)
    n_estimators = config.get("n_estimators", 100)
    subsample = config.get("subsample", 0.8)
    colsample_bytree = config.get("colsample_bytree", 0.8)
    lambda_param = config.get("lambda", 1.0)
    alpha_param = config.get("alpha", 0.0)
    
    # Convert string values to appropriate types (parameters from config may come as strings)
    max_depth = int(max_depth) if isinstance(max_depth, (str, int)) else max_depth
    learning_rate = float(learning_rate) if isinstance(learning_rate, (str, int, float)) else learning_rate
    n_estimators = int(n_estimators) if isinstance(n_estimators, (str, int)) else n_estimators
    subsample = float(subsample) if isinstance(subsample, (str, int, float)) else subsample
    colsample_bytree = float(colsample_bytree) if isinstance(colsample_bytree, (str, int, float)) else colsample_bytree
    lambda_param = float(lambda_param) if isinstance(lambda_param, (str, int, float)) else lambda_param
    alpha_param = float(alpha_param) if isinstance(alpha_param, (str, int, float)) else alpha_param
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set parameters
    params = {
        "objective": "binary:logistic",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": lambda_param,
        "alpha": alpha_param,
        "eval_metric": "logloss",
        "seed": 42,
    }
    
    lab.log(f"XGBoost Parameters:")
    lab.log(f"  max_depth: {max_depth}")
    lab.log(f"  learning_rate: {learning_rate}")
    lab.log(f"  n_estimators: {n_estimators}")
    lab.log(f"  subsample: {subsample}")
    lab.log(f"  colsample_bytree: {colsample_bytree}")
    lab.log(f"  lambda: {lambda_param}")
    lab.log(f"  alpha: {alpha_param}")
    
    # Training with evaluation set
    evals = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}
    
    lab.log("Starting training...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=10,
    )
    
    lab.log("✅ Training completed")
    return model, evals_result


def evaluate_model(model, X_test, y_test, dataset_name):
    """Evaluate trained model on test set"""
    lab.log("Evaluating model on test set...")
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    
    lab.log(f"📊 Test Set Evaluation Results:")
    lab.log(f"  Accuracy:  {accuracy:.4f}")
    lab.log(f"  Precision: {precision:.4f}")
    lab.log(f"  Recall:    {recall:.4f}")
    lab.log(f"  F1-Score:  {f1:.4f}")
    
    # Log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    lab.log(f"\n📋 Classification Report:")
    lab.log(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report,
    }


def plot_feature_importance(model, X_train, output_dir):
    """Generate feature importance plot"""
    try:
        import matplotlib.pyplot as plt
        
        lab.log("Generating feature importance plot...")
        
        # Get feature importance
        importance = model.get_score(importance_type="weight")
        
        if not importance:
            lab.log("⚠️  No feature importance data available")
            return None
        
        # Create DataFrame for plotting
        features = list(importance.keys())
        scores = list(importance.values())
        df_importance = pd.DataFrame({
            "feature": features,
            "importance": scores
        }).sort_values("importance", ascending=False).head(20)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_importance["feature"], df_importance["importance"])
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 20 Feature Importance (XGBoost)")
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        
        lab.log(f"✅ Feature importance plot saved to {plot_path}")
        return plot_path
        
    except ImportError:
        lab.log("⚠️  matplotlib not available, skipping feature importance plot")
        return None
    except Exception as e:
        lab.log(f"⚠️  Error generating feature importance plot: {e}")
        return None


def save_training_report(model, eval_results, test_metrics, output_dir, config, dataset_name):
    """Save comprehensive training report"""
    lab.log("Saving training report...")
    
    report_path = os.path.join(output_dir, "training_report.json")
    
    # Extract training history from evals_result
    train_losses = eval_results.get("train", {}).get("logloss", [])
    val_losses = eval_results.get("validation", {}).get("logloss", [])
    
    report = {
        "model_type": "XGBoost Classifier",
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": config,
        "training_history": {
            "num_boosting_rounds": len(train_losses),
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
            "best_val_loss": float(min(val_losses)) if val_losses else None,
        },
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1_score": float(test_metrics["f1_score"]),
        },
        "model_info": {
            "n_trees": model.num_boosted_rounds(),
            "max_depth": model.get_params().get("max_depth", "N/A"),
        }
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    lab.log(f"✅ Training report saved to {report_path}")
    return report_path


def main():
    """Main training function with TransformerLab integration"""
    
    start_time = datetime.now()
    
    try:
        # Initialize TransformerLab SDK (auto-loads parameters from job_data if available)
        lab.init()
        lab.log("🚀 Starting XGBoost Training with TransformerLab SDK")
        
        # Get parameters from task configuration (set via UI)
        config = lab.get_config()
        
        # Extract parameters with defaults
        dataset_name = config.get("dataset", "breast_cancer")
        test_size = config.get("test_size", 0.2)
        val_size = config.get("val_size", 0.1)
        max_depth = config.get("max_depth", 6)
        learning_rate = config.get("learning_rate", 0.1)
        n_estimators = config.get("n_estimators", 100)
        subsample = config.get("subsample", 0.8)
        colsample_bytree = config.get("colsample_bytree", 0.8)
        
        # Convert string values to appropriate types
        test_size = float(test_size) if isinstance(test_size, (str, int, float)) else test_size
        val_size = float(val_size) if isinstance(val_size, (str, int, float)) else val_size
        
        lab.log(f"Configuration loaded:")
        lab.log(f"  Dataset: {dataset_name}")
        lab.log(f"  Test size: {test_size}")
        lab.log(f"  Validation size: {val_size}")
        lab.log(f"Training started at {start_time}")
        
    except Exception as e:
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        import traceback
        traceback.print_exc()
        config = {}
        dataset_name = "breast_cancer"
        test_size = 0.2
        val_size = 0.1
    
    try:
        # Setup output directory
        output_dir = os.path.expanduser("~/xgboost_output")
        os.makedirs(output_dir, exist_ok=True)
        lab.log(f"Output directory: {output_dir}")
        
        lab.update_progress(5)
        
        # Load dataset
        lab.log("📥 Loading dataset...")
        X, y = load_dataset(dataset_name)
        lab.update_progress(10)
        
        # Split data into train, validation, and test sets
        lab.log("📊 Splitting dataset...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split temp into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        lab.log(f"  Training set: {X_train.shape[0]} samples")
        lab.log(f"  Validation set: {X_val.shape[0]} samples")
        lab.log(f"  Test set: {X_test.shape[0]} samples")
        lab.update_progress(20)
        
        # Check if we should resume from a checkpoint
        checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Attempting to resume from checkpoint: {checkpoint}")
            try:
                with open(checkpoint, "rb") as f:
                    model = pickle.load(f)
                lab.log("✅ Checkpoint loaded successfully")
            except Exception as e:
                lab.log(f"⚠️  Failed to load checkpoint: {e}, starting fresh")
                checkpoint = None
        
        if not checkpoint:
            # Train model
            lab.log("🎯 Training XGBoost model...")
            model, evals_result = train_xgboost_classifier(X_train, y_train, X_val, y_val, config)
            lab.update_progress(70)
        else:
            lab.log("Using loaded checkpoint model")
            evals_result = {}
        
        # Evaluate model
        lab.log("📊 Evaluating model...")
        test_metrics = evaluate_model(model, X_test, y_test, dataset_name)
        lab.update_progress(80)
        
        # Generate visualizations
        lab.log("📈 Generating visualizations...")
        importance_plot = plot_feature_importance(model, X_train, output_dir)
        lab.update_progress(85)
        
        # Save training report
        lab.log("📝 Saving training report...")
        report_path = save_training_report(model, evals_result, test_metrics, output_dir, config, dataset_name)
        lab.update_progress(90)
        
        # Save model
        lab.log("💾 Saving trained model...")
        model_path = os.path.join(output_dir, "xgboost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        lab.log(f"✅ Model saved to {model_path}")
        
        # Save model to TransformerLab
        saved_model_path = lab.save_model(model_path, name="xgboost_classifier")
        lab.log(f"✅ Model saved to TransformerLab: {saved_model_path}")
        
        # Save training report as artifact
        lab.save_artifact(report_path, "training_report.json")
        lab.log(f"✅ Training report saved as artifact")
        
        # Save feature importance plot as artifact
        if importance_plot:
            lab.save_artifact(importance_plot, "feature_importance.png")
            lab.log(f"✅ Feature importance plot saved as artifact")
        
        lab.update_progress(95)
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"🎉 Training completed in {training_duration}")
        
        lab.update_progress(100)
        lab.finish("XGBoost training completed successfully!")
        
        return {
            "status": "success",
            "duration": str(training_duration),
            "dataset": dataset_name,
            "model_path": model_path,
            "accuracy": test_metrics["accuracy"],
            "f1_score": test_metrics["f1_score"],
            "test_metrics": test_metrics,
        }
    
    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        print("Training interrupted by user")
        return {"status": "stopped"}
    
    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n✅ XGBoost Training Result:")
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import sys
        sys.exit(1)
