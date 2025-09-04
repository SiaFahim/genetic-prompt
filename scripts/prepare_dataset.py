#!/usr/bin/env python3
"""
Script to download and prepare the GSM8K dataset for genetic algorithm experiments.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.dataset import GSM8KDataset
from src.utils.config import get_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to prepare the dataset."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GSM8K dataset preparation...")
    
    try:
        # Initialize dataset manager
        dataset = GSM8KDataset()
        
        # Download dataset
        logger.info("Downloading GSM8K dataset...")
        dataset.download_dataset()
        
        # Load and validate dataset
        logger.info("Loading and validating dataset...")
        train_data, test_data = dataset.load_dataset()
        
        # Validate dataset
        stats = dataset.validate_dataset()
        
        # Create evaluation sets
        logger.info("Creating evaluation sets...")
        evaluation_sets = dataset.create_evaluation_sets()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("DATASET PREPARATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Training problems: {stats['train_size']}")
        logger.info(f"Test problems: {stats['test_size']}")
        logger.info(f"Train answer extraction rate: {stats['train_answer_extraction_rate']:.2%}")
        logger.info(f"Test answer extraction rate: {stats['test_answer_extraction_rate']:.2%}")
        
        logger.info("\nEvaluation sets created:")
        for set_name, set_data in evaluation_sets.items():
            logger.info(f"  {set_name}: {len(set_data)} problems")
        
        logger.info("\nDataset is ready for genetic algorithm experiments!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
