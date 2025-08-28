"""
Semantic neighborhoods for intelligent mutations in genetic algorithm.
"""

import pickle
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from ..config.hyperparameters import get_hyperparameter_config

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.embeddings.semantic_utils import SimpleWordEmbeddings
    from src.embeddings.vocabulary import TokenVocabulary
else:
    from ..utils.config import config
    from .semantic_utils import SimpleWordEmbeddings
    from .vocabulary import TokenVocabulary


class SemanticNeighborhoods:
    """Manages semantic neighborhoods for token mutations."""
    
    def __init__(self, n_neighbors: Optional[int] = None):
        """
        Initialize semantic neighborhoods.

        Args:
            n_neighbors: Number of neighbors to store for each token (uses hyperparameter config if None)
        """
        config = get_hyperparameter_config()
        self.n_neighbors = n_neighbors if n_neighbors is not None else config.n_neighbors
        self.neighborhoods = {}  # token_id -> List[token_id]
        self.embeddings = None
        self.vocabulary = None
        self.neighborhoods_built = False
    
    def build_neighborhoods(self, vocabulary: TokenVocabulary, 
                          embeddings: SimpleWordEmbeddings):
        """
        Build semantic neighborhoods for all tokens in vocabulary.
        
        Args:
            vocabulary: Token vocabulary
            embeddings: Word embeddings
        """
        print("Building semantic neighborhoods...")
        
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        
        if not vocabulary.vocab_built:
            raise ValueError("Vocabulary must be built first")
        
        # Build neighborhoods for each token
        total_tokens = len(vocabulary.token_to_id)
        processed = 0
        
        for token, token_id in vocabulary.token_to_id.items():
            # Skip special tokens
            if token.startswith('<') and token.endswith('>'):
                self.neighborhoods[token_id] = []
                continue
            
            # Get embedding for this token
            token_vec = embeddings.get_vector(token)
            if token_vec is None:
                # No embedding available, use random neighbors
                self.neighborhoods[token_id] = self._get_random_neighbors(
                    token_id, vocabulary
                )
            else:
                # Find semantic neighbors
                neighbors = self._find_semantic_neighbors(
                    token, token_vec, vocabulary, embeddings
                )
                self.neighborhoods[token_id] = neighbors
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed}/{total_tokens} tokens...")
        
        self.neighborhoods_built = True
        print(f"Built neighborhoods for {len(self.neighborhoods)} tokens")
    
    def _find_semantic_neighbors(self, target_token: str, target_vec: np.ndarray,
                               vocabulary: TokenVocabulary, 
                               embeddings: SimpleWordEmbeddings) -> List[int]:
        """Find semantic neighbors for a token."""
        similarities = []
        
        for other_token, other_token_id in vocabulary.token_to_id.items():
            # Skip self and special tokens
            if (other_token == target_token or 
                (other_token.startswith('<') and other_token.endswith('>'))):
                continue
            
            other_vec = embeddings.get_vector(other_token)
            if other_vec is not None:
                sim = embeddings.cosine_similarity(target_vec, other_vec)
                similarities.append((other_token_id, sim))
        
        # Sort by similarity and return top neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbor_ids = [token_id for token_id, _ in similarities[:self.n_neighbors]]
        
        # If we don't have enough semantic neighbors, fill with random ones
        if len(neighbor_ids) < self.n_neighbors:
            random_neighbors = self._get_random_neighbors(
                vocabulary.token_to_id[target_token], vocabulary,
                exclude=set(neighbor_ids)
            )
            neighbor_ids.extend(random_neighbors[:self.n_neighbors - len(neighbor_ids)])
        
        return neighbor_ids
    
    def _get_random_neighbors(self, token_id: int, vocabulary: TokenVocabulary,
                            exclude: Optional[Set[int]] = None) -> List[int]:
        """Get random neighbors for a token."""
        if exclude is None:
            exclude = set()
        
        exclude.add(token_id)  # Don't include self
        
        # Get all possible token IDs
        all_token_ids = list(vocabulary.id_to_token.keys())
        
        # Filter out excluded tokens and special tokens
        candidates = []
        for tid in all_token_ids:
            if tid not in exclude:
                token = vocabulary.id_to_token[tid]
                if not (token.startswith('<') and token.endswith('>')):
                    candidates.append(tid)
        
        # Sample random neighbors
        n_to_sample = min(self.n_neighbors, len(candidates))
        return random.sample(candidates, n_to_sample)
    
    def get_neighbor(self, token_id: int, semantic_prob: Optional[float] = None) -> int:
        """
        Get a neighbor for mutation.

        Args:
            token_id: Original token ID
            semantic_prob: Probability of using semantic neighbor vs random (uses hyperparameter config if None)

        Returns:
            Neighbor token ID
        """
        config = get_hyperparameter_config()
        semantic_prob = semantic_prob if semantic_prob is not None else config.semantic_prob
        if not self.neighborhoods_built:
            raise ValueError("Neighborhoods must be built first")
        
        # Check if we have neighbors for this token
        if token_id not in self.neighborhoods or not self.neighborhoods[token_id]:
            # Return a random token
            return self.vocabulary.get_random_token_id()
        
        # Decide whether to use semantic or random neighbor
        if random.random() < semantic_prob:
            # Use semantic neighbor
            return random.choice(self.neighborhoods[token_id])
        else:
            # Use random token
            return self.vocabulary.get_random_token_id()
    
    def get_multiple_neighbors(self, token_id: int, n: Optional[int] = None) -> List[int]:
        """Get multiple neighbors for a token."""
        config = get_hyperparameter_config()
        n = n if n is not None else config.neighbor_count
        if not self.neighborhoods_built:
            raise ValueError("Neighborhoods must be built first")
        
        if token_id not in self.neighborhoods or not self.neighborhoods[token_id]:
            # Return random tokens
            return [self.vocabulary.get_random_token_id() for _ in range(n)]
        
        neighbors = self.neighborhoods[token_id]
        if len(neighbors) >= n:
            return random.sample(neighbors, n)
        else:
            # Return all neighbors plus some random ones
            result = neighbors.copy()
            while len(result) < n:
                result.append(self.vocabulary.get_random_token_id())
            return result
    
    def get_neighborhood_stats(self) -> Dict[str, float]:
        """Get statistics about the neighborhoods."""
        if not self.neighborhoods_built:
            return {}
        
        neighborhood_sizes = [len(neighbors) for neighbors in self.neighborhoods.values()]
        
        return {
            'total_tokens': len(self.neighborhoods),
            'avg_neighborhood_size': np.mean(neighborhood_sizes),
            'min_neighborhood_size': np.min(neighborhood_sizes),
            'max_neighborhood_size': np.max(neighborhood_sizes),
            'tokens_with_neighbors': sum(1 for size in neighborhood_sizes if size > 0),
            'coverage': sum(1 for size in neighborhood_sizes if size > 0) / len(neighborhood_sizes)
        }
    
    def save_neighborhoods(self, filepath: Path):
        """Save neighborhoods to file."""
        data = {
            'n_neighbors': self.n_neighbors,
            'neighborhoods': self.neighborhoods,
            'neighborhoods_built': self.neighborhoods_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Neighborhoods saved to {filepath}")
    
    def load_neighborhoods(self, filepath: Path):
        """Load neighborhoods from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.n_neighbors = data['n_neighbors']
        self.neighborhoods = data['neighborhoods']
        self.neighborhoods_built = data['neighborhoods_built']
        
        print(f"Neighborhoods loaded from {filepath}")
        print(f"Loaded neighborhoods for {len(self.neighborhoods)} tokens")


# Global neighborhoods instance
semantic_neighborhoods = SemanticNeighborhoods()


if __name__ == "__main__":
    # Test semantic neighborhoods
    print("Testing semantic neighborhoods system...")
    
    # Load vocabulary and embeddings
    data_dir = config.get_data_dir()
    embeddings_dir = data_dir / "embeddings"
    
    # Load vocabulary
    vocabulary = TokenVocabulary()
    vocab_file = embeddings_dir / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        print("Building vocabulary...")
        vocabulary.build_vocabulary_from_dataset()
        vocabulary.save_vocabulary(vocab_file)
    
    # Load embeddings
    embeddings = SimpleWordEmbeddings()
    embeddings_file = embeddings_dir / "word_embeddings.pkl"
    if embeddings_file.exists():
        embeddings.load_embeddings(embeddings_file)
    else:
        print("Building embeddings...")
        embeddings.load_glove_embeddings(embeddings_dir / "glove.6B.100d.txt", 1000)
        embeddings.save_embeddings(embeddings_file)
    
    # Build neighborhoods (using hyperparameter defaults)
    neighborhoods = SemanticNeighborhoods()
    neighborhoods.build_neighborhoods(vocabulary, embeddings)
    
    # Test neighborhood lookup
    test_tokens = ['problem', 'solve', 'calculate', 'answer']
    for token in test_tokens:
        if token in vocabulary.token_to_id:
            token_id = vocabulary.token_to_id[token]
            neighbors = neighborhoods.get_multiple_neighbors(token_id)
            neighbor_tokens = [vocabulary.id_to_token[nid] for nid in neighbors]
            print(f"\nNeighbors of '{token}': {neighbor_tokens}")
    
    # Show statistics
    stats = neighborhoods.get_neighborhood_stats()
    print(f"\nNeighborhood statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save neighborhoods
    neighborhoods_file = embeddings_dir / "semantic_neighborhoods.pkl"
    neighborhoods.save_neighborhoods(neighborhoods_file)
    
    print("\n✅ Semantic neighborhoods system test completed!")
