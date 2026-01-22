#!/usr/bin/env python3
"""
Advanced Parameter Demo Task
Demonstrates how to access parameters with different types in a task script.
"""

from lab import lab
import json

def main():
    print("=" * 60)
    print("Advanced Parameter Demo Task")
    print("=" * 60)
    
    # Get all configuration parameters
    config = lab.get_config()
    
    print("\n📋 Configuration Parameters:")
    print(json.dumps(config, indent=2))
    
    # Access individual parameters
    print("\n🔍 Accessing Individual Parameters:")
    print(f"  Project Name: {config.get('project_name', 'N/A')}")
    print(f"  Batch Size: {config.get('batch_size', 'N/A')} (type: {type(config.get('batch_size')).__name__})")
    print(f"  Learning Rate: {config.get('learning_rate', 'N/A')} (type: {type(config.get('learning_rate')).__name__})")
    print(f"  Use AMP: {config.get('use_amp', 'N/A')} (type: {type(config.get('use_amp')).__name__})")
    print(f"  Optimizer: {config.get('optimizer', 'N/A')}")
    print(f"  Scheduler: {config.get('scheduler', 'N/A')}")
    print(f"  API Key: {'***' if config.get('api_key') else '(not set)'}")
    print(f"  Num Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"  Temperature: {config.get('temperature', 'N/A')}")
    print(f"  Enable Logging: {config.get('enable_logging', 'N/A')}")
    print(f"  Max Tokens: {config.get('max_tokens', 'N/A')}")
    
    print("\n✅ Task completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
