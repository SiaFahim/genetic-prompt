#!/usr/bin/env python3
"""
Script to download and prepare GloVe embeddings for the genetic algorithm system.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.glove_loader import GloVeLoader
from src.utils.config import get_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to download and prepare embeddings."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GloVe embeddings download and preparation...")
    
    try:
        # Initialize GloVe loader
        glove_loader = GloVeLoader()
        
        # Download embeddings
        logger.info("Downloading GloVe embeddings...")
        glove_loader.download_glove()
        
        # Load and process embeddings
        logger.info("Loading and processing embeddings...")
        word_to_idx, idx_to_word, embeddings = glove_loader.load_embeddings()
        
        # Test the embeddings
        logger.info("Testing embeddings...")
        
        # Test basic functionality
        test_words = ['the', 'and', 'is', 'good', 'bad', 'math', 'problem', 'solve']
        found_words = []
        
        for word in test_words:
            embedding = glove_loader.get_embedding(word)
            if embedding is not None:
                found_words.append(word)
                logger.info(f"  '{word}': embedding shape {embedding.shape}")
        
        # Test similarity
        if 'good' in found_words and 'bad' in found_words:
            similarity = glove_loader.compute_similarity('good', 'bad')
            logger.info(f"Similarity between 'good' and 'bad': {similarity:.4f}")
        
        if 'math' in found_words and 'problem' in found_words:
            similarity = glove_loader.compute_similarity('math', 'problem')
            logger.info(f"Similarity between 'math' and 'problem': {similarity:.4f}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EMBEDDINGS DOWNLOAD COMPLETE")
        logger.info("="*50)
        logger.info(f"Vocabulary size: {glove_loader.get_vocab_size()}")
        logger.info(f"Embedding dimension: {glove_loader.get_embedding_dim()}")
        logger.info(f"Test words found: {len(found_words)}/{len(test_words)}")
        logger.info(f"Embeddings matrix shape: {embeddings.shape}")
        
        logger.info("\nEmbeddings are ready for semantic neighborhood construction!")
        
    except Exception as e:
        logger.error(f"Embeddings preparation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
