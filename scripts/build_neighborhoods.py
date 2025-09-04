#!/usr/bin/env python3
"""
Script to build semantic neighborhoods for the genetic algorithm system.
This is a computationally intensive process that may take 30-60 minutes.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.neighborhoods import SemanticNeighborhoods
from src.utils.config import get_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to build semantic neighborhoods."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting semantic neighborhoods construction...")
    logger.info("This process may take 30-60 minutes depending on your hardware.")
    
    start_time = time.time()
    
    try:
        # Initialize neighborhoods manager
        neighborhoods_manager = SemanticNeighborhoods()
        
        # Build neighborhoods
        logger.info("Building semantic neighborhoods...")
        neighborhoods = neighborhoods_manager.build_neighborhoods()
        
        # Validate neighborhoods
        logger.info("Validating neighborhoods...")
        stats = neighborhoods_manager.validate_neighborhoods()
        
        # Test functionality
        logger.info("Testing neighborhood functionality...")
        
        test_words = ['good', 'bad', 'math', 'problem', 'solve', 'calculate']
        
        for word in test_words:
            if neighborhoods_manager.contains_word(word):
                neighbors = neighborhoods_manager.get_neighbors(word, count=5)
                logger.info(f"  '{word}' neighbors: {neighbors}")
                
                # Test random neighbor selection
                random_neighbor = neighborhoods_manager.get_random_neighbor(word, semantic_prob=0.9)
                logger.info(f"  '{word}' random semantic neighbor: {random_neighbor}")
        
        # Test similarity
        if (neighborhoods_manager.contains_word('good') and 
            neighborhoods_manager.contains_word('bad')):
            similarity = neighborhoods_manager.get_similarity('good', 'bad')
            logger.info(f"Similarity between 'good' and 'bad': {similarity:.4f}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SEMANTIC NEIGHBORHOODS CONSTRUCTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total words: {stats['total_words']}")
        logger.info(f"Total neighbors: {stats['total_neighbors']}")
        logger.info(f"Average neighbors per word: {stats['avg_neighbors_per_word']:.1f}")
        logger.info(f"Words without neighbors: {stats['words_without_neighbors']}")
        logger.info(f"Construction time: {elapsed_time/60:.1f} minutes")
        
        logger.info("\nSample neighborhoods:")
        for word, neighbors in stats['sample_neighborhoods'].items():
            logger.info(f"  {word}: {neighbors}")
        
        logger.info("\nSemantic neighborhoods are ready for genetic mutations!")
        
    except Exception as e:
        logger.error(f"Neighborhoods construction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
