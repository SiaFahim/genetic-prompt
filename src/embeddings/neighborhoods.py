"""
Semantic neighborhood construction and management for genetic mutations.
Builds and manages nearest neighbor relationships between words based on embeddings.
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

import numpy as np
from tqdm import tqdm

from src.embeddings.glove_loader import GloVeLoader
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class SemanticNeighborhoods:
    """Manager for semantic neighborhoods based on word embeddings."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize semantic neighborhoods manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.glove_loader = GloVeLoader(config_path)
        
        # Configuration
        self.neighborhood_size = self.config.get('embeddings.neighborhood_size', 50)
        self.fallback_enabled = self.config.get('embeddings.fallback_enabled', True)
        
        # Paths
        self.embeddings_dir = Path(self.config.get('paths.embeddings_dir', './data/embeddings'))
        self.neighborhoods_path = self.embeddings_dir / 'neighborhoods.pkl'
        
        # Loaded data
        self._neighborhoods = None
        self._word_to_idx = None
        self._idx_to_word = None
        self._embeddings = None
    
    def build_neighborhoods(self, force_rebuild: bool = False) -> Dict[str, List[str]]:
        """
        Build semantic neighborhoods for all words in vocabulary.
        
        Args:
            force_rebuild: Whether to force rebuild even if cached neighborhoods exist
            
        Returns:
            Dictionary mapping each word to its nearest neighbors
        """
        if self.neighborhoods_path.exists() and not force_rebuild:
            logger.info("Loading cached semantic neighborhoods...")
            return self.load_neighborhoods()
        
        logger.info("Building semantic neighborhoods...")
        
        # Load embeddings
        word_to_idx, idx_to_word, embeddings = self.glove_loader.load_embeddings()
        self._word_to_idx = word_to_idx
        self._idx_to_word = idx_to_word
        self._embeddings = embeddings
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        neighborhoods = {}
        vocab_size = len(word_to_idx)
        
        logger.info(f"Computing neighborhoods for {vocab_size} words...")
        
        # Compute neighborhoods for each word
        for word, word_idx in tqdm(word_to_idx.items(), desc="Building neighborhoods"):
            # Compute cosine similarities with all other words
            word_embedding = normalized_embeddings[word_idx:word_idx+1]  # Keep 2D shape
            similarities = np.dot(normalized_embeddings, word_embedding.T).flatten()
            
            # Get top k+1 most similar (including the word itself)
            top_indices = np.argsort(similarities)[::-1][:self.neighborhood_size + 1]
            
            # Convert indices to words, excluding the word itself
            neighbors = []
            for idx in top_indices:
                neighbor_word = idx_to_word[idx]
                if neighbor_word != word:  # Exclude self
                    neighbors.append(neighbor_word)
                
                if len(neighbors) >= self.neighborhood_size:
                    break
            
            neighborhoods[word] = neighbors
        
        # Cache the neighborhoods
        self._save_neighborhoods(neighborhoods)
        self._neighborhoods = neighborhoods
        
        logger.info(f"Built neighborhoods for {len(neighborhoods)} words")
        return neighborhoods
    
    def load_neighborhoods(self) -> Dict[str, List[str]]:
        """
        Load neighborhoods from cache.
        
        Returns:
            Dictionary mapping each word to its nearest neighbors
        """
        if self._neighborhoods is not None:
            return self._neighborhoods
        
        if not self.neighborhoods_path.exists():
            logger.info("No cached neighborhoods found, building from scratch...")
            return self.build_neighborhoods()
        
        try:
            with open(self.neighborhoods_path, 'rb') as f:
                neighborhoods = pickle.load(f)
            
            self._neighborhoods = neighborhoods
            logger.info(f"Loaded neighborhoods for {len(neighborhoods)} words")
            return neighborhoods
            
        except Exception as e:
            logger.error(f"Failed to load cached neighborhoods: {e}")
            logger.info("Building neighborhoods from scratch...")
            return self.build_neighborhoods(force_rebuild=True)
    
    def _save_neighborhoods(self, neighborhoods: Dict[str, List[str]]) -> None:
        """
        Save neighborhoods to cache.
        
        Args:
            neighborhoods: Dictionary of neighborhoods to save
        """
        try:
            with open(self.neighborhoods_path, 'wb') as f:
                pickle.dump(neighborhoods, f)
            
            logger.info(f"Saved neighborhoods to {self.neighborhoods_path}")
            
        except Exception as e:
            logger.error(f"Failed to save neighborhoods: {e}")
    
    def get_neighbors(self, word: str, count: Optional[int] = None) -> List[str]:
        """
        Get semantic neighbors for a word.
        
        Args:
            word: Word to get neighbors for
            count: Number of neighbors to return (default: all available)
            
        Returns:
            List of neighbor words
        """
        if self._neighborhoods is None:
            self.load_neighborhoods()
        
        neighbors = self._neighborhoods.get(word, [])
        
        if count is not None:
            neighbors = neighbors[:count]
        
        return neighbors
    
    def get_random_neighbor(self, word: str, semantic_prob: float = 0.9) -> str:
        """
        Get a random neighbor for a word, with probability of semantic vs random selection.
        
        Args:
            word: Word to get neighbor for
            semantic_prob: Probability of selecting semantic neighbor vs random word
            
        Returns:
            Neighbor word (semantic or random)
        """
        if random.random() < semantic_prob:
            # Try to get semantic neighbor
            neighbors = self.get_neighbors(word)
            if neighbors:
                return random.choice(neighbors)
        
        # Fallback to random word from vocabulary
        if self.fallback_enabled:
            if self._word_to_idx is None:
                self.glove_loader.load_embeddings()
                self._word_to_idx = self.glove_loader._word_to_idx
            
            return random.choice(list(self._word_to_idx.keys()))
        
        # If no fallback, return original word
        return word
    
    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Get cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity or None if either word not found
        """
        return self.glove_loader.compute_similarity(word1, word2)
    
    def find_most_similar(self, word: str, candidates: List[str], count: int = 1) -> List[str]:
        """
        Find most similar words from a list of candidates.
        
        Args:
            word: Reference word
            candidates: List of candidate words
            count: Number of most similar words to return
            
        Returns:
            List of most similar words from candidates
        """
        similarities = []
        
        for candidate in candidates:
            sim = self.get_similarity(word, candidate)
            if sim is not None:
                similarities.append((candidate, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return [word for word, _ in similarities[:count]]
    
    def get_diverse_words(self, reference_words: List[str], candidates: List[str], 
                         count: int = 10) -> List[str]:
        """
        Get diverse words that are least similar to reference words.
        
        Args:
            reference_words: List of reference words to be different from
            candidates: List of candidate words to choose from
            count: Number of diverse words to return
            
        Returns:
            List of most diverse words
        """
        diversity_scores = []
        
        for candidate in candidates:
            if candidate in reference_words:
                continue
            
            # Calculate minimum similarity to any reference word
            min_similarity = float('inf')
            
            for ref_word in reference_words:
                sim = self.get_similarity(candidate, ref_word)
                if sim is not None:
                    min_similarity = min(min_similarity, sim)
            
            if min_similarity != float('inf'):
                diversity_scores.append((candidate, min_similarity))
        
        # Sort by minimum similarity (ascending = most diverse)
        diversity_scores.sort(key=lambda x: x[1])
        
        # Return most diverse words
        return [word for word, _ in diversity_scores[:count]]
    
    def validate_neighborhoods(self) -> Dict[str, any]:
        """
        Validate the neighborhoods and return statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        if self._neighborhoods is None:
            self.load_neighborhoods()
        
        total_words = len(self._neighborhoods)
        total_neighbors = sum(len(neighbors) for neighbors in self._neighborhoods.values())
        avg_neighbors = total_neighbors / total_words if total_words > 0 else 0
        
        # Check for words with no neighbors
        words_without_neighbors = sum(1 for neighbors in self._neighborhoods.values() if not neighbors)
        
        # Sample some neighborhoods for quality check
        sample_words = random.sample(list(self._neighborhoods.keys()), min(5, total_words))
        sample_neighborhoods = {word: self._neighborhoods[word][:5] for word in sample_words}
        
        stats = {
            'total_words': total_words,
            'total_neighbors': total_neighbors,
            'avg_neighbors_per_word': avg_neighbors,
            'words_without_neighbors': words_without_neighbors,
            'sample_neighborhoods': sample_neighborhoods
        }
        
        logger.info(f"Neighborhoods validation: {stats}")
        return stats
    
    def get_vocabulary(self) -> List[str]:
        """Get the complete vocabulary."""
        if self._neighborhoods is None:
            self.load_neighborhoods()
        
        return list(self._neighborhoods.keys())
    
    def contains_word(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        if self._neighborhoods is None:
            self.load_neighborhoods()
        
        return word in self._neighborhoods
