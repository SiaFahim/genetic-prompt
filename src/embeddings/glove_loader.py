"""
GloVe embeddings loader and manager.
Downloads and manages GloVe word embeddings for semantic similarity operations.
"""

import os
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
from tqdm import tqdm

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class GloVeLoader:
    """Loader and manager for GloVe word embeddings."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize GloVe loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.embeddings_dir = Path(self.config.get('paths.embeddings_dir', './data/embeddings'))
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # GloVe configuration
        self.embedding_dim = 100  # Using 100-dimensional embeddings
        self.vocab_size = self.config.get('embeddings.vocab_size', 10000)
        
        # File paths
        self.glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        self.glove_file = "glove.6B.100d.txt"
        self.zip_path = self.embeddings_dir / "glove.6B.zip"
        self.txt_path = self.embeddings_dir / self.glove_file
        self.processed_path = self.embeddings_dir / "glove_processed.pkl"
        
        # Loaded data
        self._word_to_idx = None
        self._idx_to_word = None
        self._embeddings = None
    
    def download_glove(self, force_download: bool = False) -> None:
        """
        Download GloVe embeddings if not already present.
        
        Args:
            force_download: Whether to force re-download even if files exist
        """
        if self.txt_path.exists() and not force_download:
            logger.info(f"GloVe embeddings already exist at {self.txt_path}")
            return
        
        logger.info("Downloading GloVe embeddings...")
        
        # Download zip file
        if not self.zip_path.exists() or force_download:
            logger.info(f"Downloading from {self.glove_url}")
            
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) // total_size)
                    print(f"\rDownloading: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='')
            
            try:
                urllib.request.urlretrieve(self.glove_url, self.zip_path, progress_hook)
                print()  # New line after progress
                logger.info(f"Downloaded GloVe zip to {self.zip_path}")
            except Exception as e:
                logger.error(f"Failed to download GloVe embeddings: {e}")
                raise
        
        # Extract the specific file we need
        if not self.txt_path.exists() or force_download:
            logger.info("Extracting GloVe embeddings...")
            
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    # Extract only the 100d file
                    zip_ref.extract(self.glove_file, self.embeddings_dir)
                
                logger.info(f"Extracted {self.glove_file}")
                
                # Clean up zip file to save space
                self.zip_path.unlink()
                logger.info("Cleaned up zip file")
                
            except Exception as e:
                logger.error(f"Failed to extract GloVe embeddings: {e}")
                raise
    
    def load_embeddings(self, force_reload: bool = False) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
        """
        Load GloVe embeddings into memory.
        
        Args:
            force_reload: Whether to force reload even if already cached
            
        Returns:
            Tuple of (word_to_idx, idx_to_word, embeddings_matrix)
        """
        if (self._word_to_idx is not None and 
            self._idx_to_word is not None and 
            self._embeddings is not None and 
            not force_reload):
            return self._word_to_idx, self._idx_to_word, self._embeddings
        
        # Try to load from processed cache first
        if self.processed_path.exists() and not force_reload:
            logger.info("Loading processed GloVe embeddings from cache...")
            
            try:
                with open(self.processed_path, 'rb') as f:
                    data = pickle.load(f)
                
                self._word_to_idx = data['word_to_idx']
                self._idx_to_word = data['idx_to_word']
                self._embeddings = data['embeddings']
                
                logger.info(f"Loaded {len(self._word_to_idx)} embeddings from cache")
                return self._word_to_idx, self._idx_to_word, self._embeddings
                
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Download if needed
        self.download_glove()
        
        # Load from text file
        logger.info("Loading GloVe embeddings from text file...")
        
        if not self.txt_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {self.txt_path}")
        
        word_to_idx = {}
        embeddings_list = []
        
        # Count total lines for progress bar
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # Load embeddings
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, total=total_lines, desc="Loading embeddings")):
                if idx >= self.vocab_size:
                    break
                
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if len(vector) != self.embedding_dim:
                    logger.warning(f"Skipping word '{word}' with incorrect dimension: {len(vector)}")
                    continue
                
                word_to_idx[word] = len(embeddings_list)
                embeddings_list.append(vector)
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        
        # Create reverse mapping
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Cache the processed data
        logger.info("Caching processed embeddings...")
        
        cache_data = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'embeddings': embeddings_matrix
        }
        
        with open(self.processed_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Store in instance
        self._word_to_idx = word_to_idx
        self._idx_to_word = idx_to_word
        self._embeddings = embeddings_matrix
        
        logger.info(f"Loaded {len(word_to_idx)} GloVe embeddings")
        
        return word_to_idx, idx_to_word, embeddings_matrix
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector or None if word not found
        """
        if self._word_to_idx is None:
            self.load_embeddings()
        
        if word in self._word_to_idx:
            idx = self._word_to_idx[word]
            return self._embeddings[idx]
        
        return None
    
    def get_word_from_idx(self, idx: int) -> Optional[str]:
        """
        Get word from index.
        
        Args:
            idx: Index to get word for
            
        Returns:
            Word or None if index not found
        """
        if self._idx_to_word is None:
            self.load_embeddings()
        
        return self._idx_to_word.get(idx)
    
    def get_idx_from_word(self, word: str) -> Optional[int]:
        """
        Get index from word.
        
        Args:
            word: Word to get index for
            
        Returns:
            Index or None if word not found
        """
        if self._word_to_idx is None:
            self.load_embeddings()
        
        return self._word_to_idx.get(word)
    
    def compute_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity or None if either word not found
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_vocab_size(self) -> int:
        """Get the size of the loaded vocabulary."""
        if self._word_to_idx is None:
            self.load_embeddings()
        
        return len(self._word_to_idx)
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.embedding_dim
    
    def get_all_words(self) -> List[str]:
        """Get list of all words in vocabulary."""
        if self._word_to_idx is None:
            self.load_embeddings()
        
        return list(self._word_to_idx.keys())
