"""
Dataset utilities for loading and processing GSM8K data.
"""

import json
import random
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import load_from_disk

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    # Add project root to path for standalone execution
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.utils.answer_extraction import extract_final_answer
else:
    # Relative imports for module usage
    from .config import config
    from .answer_extraction import extract_final_answer


class GSM8KDataset:
    """Utility class for loading and managing GSM8K dataset."""
    
    def __init__(self):
        """Initialize the dataset manager."""
        self.data_dir = config.get_data_dir()
        self._raw_dataset = None
        self._evaluation_sets = {}
    
    def load_raw_dataset(self):
        """Load the raw GSM8K dataset from disk."""
        if self._raw_dataset is None:
            gsm8k_dir = self.data_dir / "gsm8k_raw"
            if not gsm8k_dir.exists():
                raise FileNotFoundError(
                    f"GSM8K dataset not found at {gsm8k_dir}. "
                    "Please run scripts/download_dataset.py first."
                )
            self._raw_dataset = load_from_disk(str(gsm8k_dir))
        return self._raw_dataset
    
    def load_evaluation_set(self, set_name: str) -> List[Dict[str, Any]]:
        """
        Load an evaluation set.
        
        Args:
            set_name: Name of the evaluation set ('primary_eval', 'validation', 'final_test')
            
        Returns:
            List of problem dictionaries
        """
        if set_name not in self._evaluation_sets:
            set_file = self.data_dir / f"{set_name}_set.json"
            if not set_file.exists():
                raise FileNotFoundError(
                    f"Evaluation set {set_name} not found at {set_file}. "
                    "Please run scripts/create_evaluation_subsets.py first."
                )
            
            with open(set_file, 'r') as f:
                self._evaluation_sets[set_name] = json.load(f)
        
        return self._evaluation_sets[set_name]
    
    def get_primary_eval_set(self) -> List[Dict[str, Any]]:
        """Get the primary evaluation set (100 problems for fitness calculation)."""
        return self.load_evaluation_set('primary_eval')
    
    def get_validation_set(self) -> List[Dict[str, Any]]:
        """Get the validation set (100 problems to prevent overfitting)."""
        return self.load_evaluation_set('validation')
    
    def get_final_test_set(self) -> List[Dict[str, Any]]:
        """Get the final test set (200 problems for comprehensive evaluation)."""
        return self.load_evaluation_set('final_test')
    
    def sample_problems(self, set_name: str, n_problems: int, 
                       seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample n problems from an evaluation set.
        
        Args:
            set_name: Name of the evaluation set
            n_problems: Number of problems to sample
            seed: Random seed for reproducible sampling
            
        Returns:
            List of sampled problem dictionaries
        """
        problems = self.load_evaluation_set(set_name)
        
        if n_problems >= len(problems):
            return problems
        
        if seed is not None:
            random.seed(seed)
        
        return random.sample(problems, n_problems)
    
    def get_adaptive_eval_problems(self, generation: int) -> List[Dict[str, Any]]:
        """
        Get evaluation problems with adaptive size based on generation.
        
        Args:
            generation: Current generation number
            
        Returns:
            List of problems for evaluation
        """
        if generation <= 10:
            # Early generations: 50 problems for quick iteration
            return self.sample_problems('primary_eval', 50, seed=generation)
        elif generation <= 20:
            # Middle generations: 75 problems for better signal
            return self.sample_problems('primary_eval', 75, seed=generation)
        else:
            # Late generations: 100 problems for fine-tuning
            return self.get_primary_eval_set()
    
    def get_problem_statistics(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about a set of problems.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not problems:
            return {}
        
        # Calculate answer statistics
        answers = [p['final_answer'] for p in problems if p['final_answer'] is not None]
        
        stats = {
            'total_problems': len(problems),
            'problems_with_answers': len(answers),
            'problems_without_answers': len(problems) - len(answers),
        }
        
        if answers:
            stats.update({
                'min_answer': min(answers),
                'max_answer': max(answers),
                'mean_answer': sum(answers) / len(answers),
            })
        
        # Calculate question length statistics
        question_lengths = [len(p['question'].split()) for p in problems]
        if question_lengths:
            stats.update({
                'min_question_length': min(question_lengths),
                'max_question_length': max(question_lengths),
                'mean_question_length': sum(question_lengths) / len(question_lengths),
            })
        
        return stats
    
    def validate_dataset(self) -> bool:
        """
        Validate that all required dataset files exist and are properly formatted.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check raw dataset
            self.load_raw_dataset()
            print("✅ Raw dataset loaded successfully")
            
            # Check evaluation sets
            for set_name in ['primary_eval', 'validation', 'final_test']:
                problems = self.load_evaluation_set(set_name)
                stats = self.get_problem_statistics(problems)
                print(f"✅ {set_name} set: {stats['total_problems']} problems, "
                      f"{stats['problems_with_answers']} with valid answers")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return False


# Global dataset instance
gsm8k_dataset = GSM8KDataset()


def load_problems(set_name: str, n_problems: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to load problems from an evaluation set.
    
    Args:
        set_name: Name of the evaluation set
        n_problems: Number of problems to sample (None for all)
        
    Returns:
        List of problem dictionaries
    """
    if n_problems is None:
        return gsm8k_dataset.load_evaluation_set(set_name)
    else:
        return gsm8k_dataset.sample_problems(set_name, n_problems)


if __name__ == "__main__":
    # Validate the dataset
    print("Validating GSM8K dataset...")
    success = gsm8k_dataset.validate_dataset()
    
    if success:
        print("\n✅ Dataset validation completed successfully!")
        
        # Show some statistics
        print("\nDataset Statistics:")
        for set_name in ['primary_eval', 'validation', 'final_test']:
            problems = gsm8k_dataset.load_evaluation_set(set_name)
            stats = gsm8k_dataset.get_problem_statistics(problems)
            print(f"\n{set_name.upper()} SET:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print("\n❌ Dataset validation failed!")
        exit(1)
