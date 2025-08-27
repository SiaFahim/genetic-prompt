"""
Semantic utilities for word embeddings and similarity calculations.
Alternative implementation without gensim for Python 3.13 compatibility.
"""

import numpy as np
import pickle
import random
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import urllib.request
import gzip
import os

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
else:
    from ..utils.config import config


class SimpleWordEmbeddings:
    """Simple word embeddings loader and manager."""
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize word embeddings.
        
        Args:
            embedding_dim: Dimension of word embeddings
        """
        self.embedding_dim = embedding_dim
        self.word_to_vec = {}
        self.vocab = set()
        self.vocab_size = 0
    
    def load_glove_embeddings(self, glove_file: Path, vocab_limit: int = 10000):
        """
        Load GloVe embeddings from file.
        
        Args:
            glove_file: Path to GloVe embeddings file
            vocab_limit: Maximum number of words to load
        """
        print(f"Loading GloVe embeddings from {glove_file}...")
        
        if not glove_file.exists():
            print(f"GloVe file not found: {glove_file}")
            print("Creating simple random embeddings for demonstration...")
            self._create_demo_embeddings(vocab_limit)
            return
        
        word_count = 0
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                if word_count >= vocab_limit:
                    break
                
                parts = line.strip().split()
                if len(parts) != self.embedding_dim + 1:
                    continue
                
                word = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    self.word_to_vec[word] = vector
                    self.vocab.add(word)
                    word_count += 1
                except ValueError:
                    continue
        
        self.vocab_size = len(self.vocab)
        print(f"Loaded {self.vocab_size} word embeddings")
    
    def _create_demo_embeddings(self, vocab_size: int):
        """Create simple random embeddings for demonstration purposes."""
        # Common English words for demonstration
        demo_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
            'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
            'what', 'which', 'who', 'whom', 'whose', 'all', 'any', 'some', 'many', 'much',
            'more', 'most', 'less', 'least', 'few', 'several', 'each', 'every', 'both', 'either',
            'neither', 'one', 'two', 'three', 'four', 'five', 'first', 'second', 'last', 'next',
            'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'new', 'old',
            'young', 'right', 'wrong', 'true', 'false', 'same', 'different', 'easy', 'hard',
            'make', 'take', 'give', 'get', 'go', 'come', 'see', 'know', 'think', 'say',
            'tell', 'ask', 'work', 'play', 'run', 'walk', 'sit', 'stand', 'look', 'find',
            'problem', 'solution', 'answer', 'question', 'step', 'method', 'way', 'approach',
            'calculate', 'solve', 'find', 'determine', 'compute', 'figure', 'work', 'out',
            'number', 'value', 'result', 'total', 'sum', 'difference', 'product', 'quotient',
            'add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'times', 'equals'
        ]
        
        # Extend with more words if needed
        while len(demo_words) < vocab_size:
            demo_words.append(f"word_{len(demo_words)}")
        
        # Create random embeddings
        np.random.seed(42)  # For reproducibility
        for word in demo_words[:vocab_size]:
            vector = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            # Normalize vector
            vector = vector / np.linalg.norm(vector)
            self.word_to_vec[word] = vector
            self.vocab.add(word)
        
        self.vocab_size = len(self.vocab)
        print(f"Created {self.vocab_size} demo word embeddings")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word."""
        return self.word_to_vec.get(word.lower())
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_words(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Find n most similar words to the given word.
        
        Args:
            word: Target word
            n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        target_vec = self.get_vector(word)
        if target_vec is None:
            return []
        
        similarities = []
        for other_word, other_vec in self.word_to_vec.items():
            if other_word != word.lower():
                sim = self.cosine_similarity(target_vec, other_vec)
                similarities.append((other_word, sim))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def save_embeddings(self, filepath: Path):
        """Save embeddings to file."""
        data = {
            'word_to_vec': self.word_to_vec,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: Path):
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word_to_vec = data['word_to_vec']
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.embedding_dim = data['embedding_dim']
        print(f"Loaded {self.vocab_size} embeddings from {filepath}")


def download_glove_embeddings(embedding_dim: int = 100) -> Path:
    """
    Download GloVe embeddings if not already present.
    
    Args:
        embedding_dim: Dimension of embeddings (50, 100, 200, or 300)
        
    Returns:
        Path to the embeddings file
    """
    data_dir = config.get_data_dir()
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"glove.6B.{embedding_dim}d.txt"
    filepath = embeddings_dir / filename
    
    if filepath.exists():
        print(f"GloVe embeddings already exist at {filepath}")
        return filepath
    
    # Note: In a real implementation, you would download from:
    # https://nlp.stanford.edu/data/glove.6B.zip
    # For now, we'll create a placeholder
    print(f"GloVe embeddings not found. Would download from Stanford NLP.")
    print(f"For this demo, using random embeddings instead.")
    
    return filepath


if __name__ == "__main__":
    # Test the embeddings system
    print("Testing semantic embeddings system...")
    
    # Initialize embeddings
    embeddings = SimpleWordEmbeddings(embedding_dim=100)
    
    # Try to load or create embeddings
    glove_file = download_glove_embeddings(100)
    embeddings.load_glove_embeddings(glove_file, vocab_limit=1000)
    
    # Test similarity
    test_words = ['good', 'problem', 'calculate', 'number']
    for word in test_words:
        if word in embeddings.vocab:
            similar = embeddings.find_similar_words(word, 5)
            print(f"\nWords similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")
    
    # Save embeddings
    embeddings_file = config.get_data_dir() / "embeddings" / "word_embeddings.pkl"
    embeddings.save_embeddings(embeddings_file)
    
    print("\n✅ Semantic embeddings system test completed!")
