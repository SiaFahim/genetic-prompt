#!/usr/bin/env python3
"""
Download and prepare the GSM8K dataset for the genetic algorithm project.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets import load_dataset
from src.utils.config import config
import json


def download_gsm8k():
    """Download the GSM8K dataset and save it locally."""
    print("Downloading GSM8K dataset...")
    
    try:
        # Download the dataset
        dataset = load_dataset("openai/gsm8k", "main")
        
        # Create data directory if it doesn't exist
        data_dir = config.get_data_dir()
        gsm8k_dir = data_dir / "gsm8k_raw"
        gsm8k_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the dataset
        dataset.save_to_disk(str(gsm8k_dir))
        
        # Print dataset info
        print(f"Dataset saved to: {gsm8k_dir}")
        print(f"Train set size: {len(dataset['train'])}")
        print(f"Test set size: {len(dataset['test'])}")
        
        # Save a sample for inspection
        sample_file = gsm8k_dir / "sample.json"
        sample_data = {
            "train_sample": dataset['train'][0],
            "test_sample": dataset['test'][0]
        }
        
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Sample data saved to: {sample_file}")
        print("\nSample train problem:")
        print(f"Question: {dataset['train'][0]['question']}")
        print(f"Answer: {dataset['train'][0]['answer']}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_gsm8k()
    if success:
        print("\n✅ Dataset download completed successfully!")
    else:
        print("\n❌ Dataset download failed!")
        sys.exit(1)
