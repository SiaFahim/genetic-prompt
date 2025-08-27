#!/usr/bin/env python3
"""
Create evaluation subsets from the GSM8K dataset for the genetic algorithm project.
"""

import sys
import random
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets import load_from_disk
from src.utils.config import config
from src.utils.answer_extraction import extract_final_answer


def create_evaluation_subsets():
    """Create evaluation subsets with deterministic random sampling."""
    print("Creating evaluation subsets...")
    
    try:
        # Load the dataset
        data_dir = config.get_data_dir()
        gsm8k_dir = data_dir / "gsm8k_raw"
        dataset = load_from_disk(str(gsm8k_dir))
        
        train_data = list(dataset['train'])
        test_data = list(dataset['test'])
        
        print(f"Loaded {len(train_data)} training problems")
        print(f"Loaded {len(test_data)} test problems")
        
        # Create evaluation subsets with deterministic sampling
        subsets = {}
        
        # Primary evaluation set: 100 problems from test set (seed=42)
        random.seed(42)
        primary_eval = random.sample(test_data, 100)
        subsets['primary_eval'] = primary_eval
        
        # Validation set: 100 problems from train set (seed=43)
        random.seed(43)
        validation_set = random.sample(train_data, 100)
        subsets['validation'] = validation_set
        
        # Final test set: 200 problems from test set, non-overlapping with primary (seed=44)
        remaining_test = [item for item in test_data if item not in primary_eval]
        random.seed(44)
        final_test = random.sample(remaining_test, min(200, len(remaining_test)))
        subsets['final_test'] = final_test
        
        # Process and save each subset
        for subset_name, subset_data in subsets.items():
            processed_subset = []
            
            for i, item in enumerate(subset_data):
                # Extract the final answer
                final_answer = extract_final_answer(item['answer'])
                
                processed_item = {
                    'id': f"gsm8k_{subset_name}_{i:04d}",
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': final_answer
                }
                processed_subset.append(processed_item)
            
            # Save subset
            subset_file = data_dir / f"{subset_name}_set.json"
            with open(subset_file, 'w') as f:
                json.dump(processed_subset, f, indent=2)
            
            print(f"Created {subset_name} set: {len(processed_subset)} problems -> {subset_file}")
            
            # Show sample
            if processed_subset:
                sample = processed_subset[0]
                print(f"  Sample question: {sample['question'][:100]}...")
                print(f"  Sample final answer: {sample['final_answer']}")
        
        # Create a summary file
        summary = {
            'primary_eval': {
                'size': len(subsets['primary_eval']),
                'source': 'test_set',
                'seed': 42,
                'description': 'Primary evaluation set for fitness calculation during evolution'
            },
            'validation': {
                'size': len(subsets['validation']),
                'source': 'train_set',
                'seed': 43,
                'description': 'Validation set to prevent overfitting'
            },
            'final_test': {
                'size': len(subsets['final_test']),
                'source': 'test_set',
                'seed': 44,
                'description': 'Final test set for comprehensive evaluation'
            }
        }
        
        summary_file = data_dir / "evaluation_subsets_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        return True
        
    except Exception as e:
        print(f"Error creating evaluation subsets: {e}")
        return False


if __name__ == "__main__":
    success = create_evaluation_subsets()
    if success:
        print("\n✅ Evaluation subsets created successfully!")
    else:
        print("\n❌ Failed to create evaluation subsets!")
        sys.exit(1)
