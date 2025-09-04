"""
Dataset preparation and management for GSM8K genetic algorithm system.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from datasets import load_dataset
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def extract_final_answer(answer_text: str) -> Optional[float]:
    """
    Extract numeric answer after #### marker from GSM8K answer format.
    
    Args:
        answer_text: The answer text containing step-by-step solution
        
    Returns:
        Float value of the final answer, or None if no answer found
    """
    # Pattern to match numbers after #### marker
    # Handles: integers, decimals, negative numbers, commas
    pattern = r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, answer_text)
    
    if match:
        number_str = match.group(1).replace(',', '')
        try:
            return float(number_str)
        except ValueError:
            logger.warning(f"Could not convert extracted answer to float: {number_str}")
            return None
    
    logger.warning(f"No answer found in text: {answer_text[:100]}...")
    return None


class GSM8KDataset:
    """Manager for GSM8K dataset preparation and access."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize dataset manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.data_dir = Path(self.config.get('paths.data_dir', './data'))
        self.raw_data_dir = self.data_dir / 'gsm8k_raw'
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._train_data = None
        self._test_data = None
        self._evaluation_sets = {}
    
    def download_dataset(self, force_reload: bool = False) -> None:
        """
        Download and cache GSM8K dataset locally.
        
        Args:
            force_reload: Whether to force re-download even if cached
        """
        if self.raw_data_dir.exists() and not force_reload:
            logger.info(f"Dataset already exists at {self.raw_data_dir}")
            return
        
        logger.info("Downloading GSM8K dataset...")
        
        try:
            # Download dataset
            dataset_config = self.config.get('dataset', {})
            dataset_name = dataset_config.get('name', 'openai/gsm8k')
            split_name = dataset_config.get('split', 'main')
            
            dataset = load_dataset(dataset_name, split_name)
            
            # Save to disk
            dataset.save_to_disk(str(self.raw_data_dir))
            
            logger.info(f"Dataset saved to {self.raw_data_dir}")
            logger.info(f"Train set: {len(dataset['train'])} problems")
            logger.info(f"Test set: {len(dataset['test'])} problems")
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load the GSM8K dataset from disk.
        
        Returns:
            Tuple of (train_data, test_data) as lists of dictionaries
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {self.raw_data_dir}. Run download_dataset() first.")
        
        if self._train_data is None or self._test_data is None:
            logger.info("Loading GSM8K dataset from disk...")
            
            from datasets import load_from_disk
            dataset = load_from_disk(str(self.raw_data_dir))
            
            # Convert to list of dictionaries with processed answers
            self._train_data = self._process_split(dataset['train'])
            self._test_data = self._process_split(dataset['test'])
            
            logger.info(f"Loaded {len(self._train_data)} training problems")
            logger.info(f"Loaded {len(self._test_data)} test problems")
        
        return self._train_data, self._test_data
    
    def _process_split(self, split_data) -> List[Dict]:
        """
        Process a dataset split to extract final answers.
        
        Args:
            split_data: HuggingFace dataset split
            
        Returns:
            List of processed problem dictionaries
        """
        processed = []
        
        for i, item in enumerate(split_data):
            final_answer = extract_final_answer(item['answer'])
            
            processed_item = {
                'id': f"gsm8k_{i:06d}",
                'question': item['question'],
                'answer': item['answer'],
                'final_answer': final_answer
            }
            
            processed.append(processed_item)
        
        return processed
    
    def create_evaluation_sets(self, force_recreate: bool = False) -> Dict[str, List[Dict]]:
        """
        Create evaluation subsets with deterministic sampling.
        
        Args:
            force_recreate: Whether to recreate sets even if they exist
            
        Returns:
            Dictionary of evaluation sets
        """
        if self._evaluation_sets and not force_recreate:
            return self._evaluation_sets
        
        train_data, test_data = self.load_dataset()
        
        eval_config = self.config.get('dataset.evaluation_sets', {})
        
        # Create evaluation sets
        evaluation_sets = {}
        
        for set_name, set_config in eval_config.items():
            size = set_config.get('size', 100)
            source = set_config.get('source', 'test')
            seed = set_config.get('seed', 42)
            
            # Select source data
            source_data = train_data if source == 'train' else test_data
            
            # Deterministic sampling
            random.seed(seed)
            if size >= len(source_data):
                sampled_data = source_data.copy()
            else:
                sampled_data = random.sample(source_data, size)
            
            evaluation_sets[set_name] = sampled_data
            
            logger.info(f"Created {set_name} evaluation set: {len(sampled_data)} problems (seed={seed})")
        
        self._evaluation_sets = evaluation_sets
        
        # Save evaluation sets to disk
        self._save_evaluation_sets()
        
        return evaluation_sets
    
    def _save_evaluation_sets(self) -> None:
        """Save evaluation sets to JSON files."""
        for set_name, set_data in self._evaluation_sets.items():
            file_path = self.data_dir / f"{set_name}_evaluation_set.json"
            
            with open(file_path, 'w') as f:
                json.dump(set_data, f, indent=2)
            
            logger.info(f"Saved {set_name} evaluation set to {file_path}")
    
    def load_evaluation_set(self, set_name: str) -> List[Dict]:
        """
        Load a specific evaluation set.
        
        Args:
            set_name: Name of the evaluation set
            
        Returns:
            List of problems in the evaluation set
        """
        if set_name in self._evaluation_sets:
            return self._evaluation_sets[set_name]
        
        # Try to load from disk
        file_path = self.data_dir / f"{set_name}_evaluation_set.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self._evaluation_sets[set_name] = data
            return data
        
        # Create if not exists
        evaluation_sets = self.create_evaluation_sets()
        
        if set_name not in evaluation_sets:
            raise ValueError(f"Evaluation set '{set_name}' not found")
        
        return evaluation_sets[set_name]
    
    def get_problems_for_generation(self, generation: int) -> List[Dict]:
        """
        Get the appropriate number of problems for a given generation.
        
        Args:
            generation: Current generation number
            
        Returns:
            List of problems to evaluate on
        """
        eval_config = self.config.get('evaluation.progressive_evaluation', {})
        
        # Determine which configuration to use based on generation
        if generation <= eval_config.get('early_generations', {}).get('range', [1, 10])[1]:
            problems_count = eval_config.get('early_generations', {}).get('problems_per_genome', 50)
        elif generation <= eval_config.get('middle_generations', {}).get('range', [11, 20])[1]:
            problems_count = eval_config.get('middle_generations', {}).get('problems_per_genome', 100)
        else:
            problems_count = eval_config.get('late_generations', {}).get('problems_per_genome', 150)
        
        # Get primary evaluation set
        primary_set = self.load_evaluation_set('primary')
        
        # Sample the required number of problems
        if problems_count >= len(primary_set):
            return primary_set
        
        # Use generation as seed for reproducible sampling within generation
        random.seed(self.config.get('experiment.random_seed', 42) + generation)
        return random.sample(primary_set, problems_count)
    
    def get_full_test_set(self) -> List[Dict]:
        """Get the complete test set for final evaluation."""
        _, test_data = self.load_dataset()
        return test_data
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate the dataset and return statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        train_data, test_data = self.load_dataset()
        
        stats = {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_valid_answers': sum(1 for item in train_data if item['final_answer'] is not None),
            'test_valid_answers': sum(1 for item in test_data if item['final_answer'] is not None),
        }
        
        stats['train_answer_extraction_rate'] = stats['train_valid_answers'] / stats['train_size']
        stats['test_answer_extraction_rate'] = stats['test_valid_answers'] / stats['test_size']
        
        logger.info(f"Dataset validation: {stats}")
        
        return stats
