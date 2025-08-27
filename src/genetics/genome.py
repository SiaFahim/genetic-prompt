"""
PromptGenome class for genetic algorithm representation of prompts.
"""

import random
import uuid
from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy
import numpy as np

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.embeddings.vocabulary import vocabulary
    from src.embeddings.neighborhoods import semantic_neighborhoods
else:
    from ..utils.config import config
    from ..embeddings.vocabulary import vocabulary
    from ..embeddings.neighborhoods import semantic_neighborhoods


class PromptGenome:
    """
    Represents a prompt as a genome for genetic algorithm evolution.
    
    The genome is represented as a sequence of token IDs that can be
    mutated and crossed over to create new prompt variants.
    """
    
    def __init__(self, token_ids: Optional[List[int]] = None, 
                 prompt_text: Optional[str] = None,
                 genome_id: Optional[str] = None):
        """
        Initialize a PromptGenome.
        
        Args:
            token_ids: List of token IDs representing the prompt
            prompt_text: Text to convert to token IDs (alternative to token_ids)
            genome_id: Unique identifier for this genome
        """
        self.genome_id = genome_id or str(uuid.uuid4())
        self.generation = 0
        self.fitness = None
        self.parent_ids = []
        self.mutation_history = []
        
        # Initialize token sequence
        if token_ids is not None:
            self.token_ids = token_ids.copy()
        elif prompt_text is not None:
            self.token_ids = vocabulary.encode_text(prompt_text)
        else:
            # Create empty genome
            self.token_ids = []
        
        # Genome statistics
        self.age = 0
        self.evaluation_count = 0
        self.best_fitness = None
    
    @classmethod
    def from_text(cls, prompt_text: str, genome_id: Optional[str] = None) -> 'PromptGenome':
        """Create a genome from prompt text."""
        return cls(prompt_text=prompt_text, genome_id=genome_id)
    
    @classmethod
    def from_tokens(cls, token_ids: List[int], genome_id: Optional[str] = None) -> 'PromptGenome':
        """Create a genome from token IDs."""
        return cls(token_ids=token_ids, genome_id=genome_id)
    
    def to_text(self) -> str:
        """Convert genome to prompt text."""
        return vocabulary.decode_ids(self.token_ids)
    
    def length(self) -> int:
        """Get the length of the genome in tokens."""
        return len(self.token_ids)
    
    def is_empty(self) -> bool:
        """Check if genome is empty."""
        return len(self.token_ids) == 0
    
    def copy(self) -> 'PromptGenome':
        """Create a deep copy of this genome."""
        new_genome = PromptGenome(
            token_ids=self.token_ids.copy(),
            genome_id=str(uuid.uuid4())
        )
        new_genome.generation = self.generation
        new_genome.parent_ids = self.parent_ids.copy()
        new_genome.mutation_history = self.mutation_history.copy()
        new_genome.age = self.age
        new_genome.evaluation_count = self.evaluation_count
        new_genome.best_fitness = self.best_fitness
        return new_genome
    
    def set_fitness(self, fitness: float):
        """Set the fitness score for this genome."""
        self.fitness = fitness
        self.evaluation_count += 1
        
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
    
    def get_diversity_score(self, other: 'PromptGenome') -> float:
        """
        Calculate diversity score between this genome and another.
        
        Args:
            other: Another PromptGenome
            
        Returns:
            Diversity score between 0 (identical) and 1 (completely different)
        """
        if self.is_empty() or other.is_empty():
            return 1.0 if self.length() != other.length() else 0.0
        
        # Use edit distance normalized by max length
        max_len = max(self.length(), other.length())
        if max_len == 0:
            return 0.0
        
        # Simple edit distance calculation
        edit_distance = self._calculate_edit_distance(self.token_ids, other.token_ids)
        return min(1.0, edit_distance / max_len)
    
    def _calculate_edit_distance(self, seq1: List[int], seq2: List[int]) -> int:
        """Calculate edit distance between two token sequences."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                      dp[i][j-1],     # insertion
                                      dp[i-1][j-1])   # substitution
        
        return dp[m][n]
    
    def validate(self) -> bool:
        """
        Validate that the genome is well-formed.
        
        Returns:
            True if genome is valid, False otherwise
        """
        # Check if all token IDs are valid
        if not vocabulary.vocab_built:
            return True  # Can't validate without vocabulary
        
        max_token_id = len(vocabulary.id_to_token) - 1
        for token_id in self.token_ids:
            if not isinstance(token_id, int) or token_id < 0 or token_id > max_token_id:
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this genome."""
        return {
            'genome_id': self.genome_id,
            'length': self.length(),
            'generation': self.generation,
            'age': self.age,
            'fitness': self.fitness,
            'best_fitness': self.best_fitness,
            'evaluation_count': self.evaluation_count,
            'parent_count': len(self.parent_ids),
            'mutation_count': len(self.mutation_history),
            'is_valid': self.validate(),
            'prompt_text': self.to_text()[:100] + '...' if self.length() > 0 else ''
        }
    
    def __str__(self) -> str:
        """String representation of the genome."""
        text = self.to_text()
        if len(text) > 100:
            text = text[:97] + "..."
        return f"PromptGenome(id={self.genome_id[:8]}, len={self.length()}, fitness={self.fitness}, text='{text}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        return (f"PromptGenome(id={self.genome_id}, token_ids={self.token_ids}, "
                f"generation={self.generation}, fitness={self.fitness})")
    
    def __eq__(self, other) -> bool:
        """Check equality based on token sequence."""
        if not isinstance(other, PromptGenome):
            return False
        return self.token_ids == other.token_ids
    
    def __hash__(self) -> int:
        """Hash based on token sequence."""
        return hash(tuple(self.token_ids))


def create_random_genome(min_length: int = 5, max_length: int = 20) -> PromptGenome:
    """
    Create a random genome for initialization.
    
    Args:
        min_length: Minimum number of tokens
        max_length: Maximum number of tokens
        
    Returns:
        Random PromptGenome
    """
    if not vocabulary.vocab_built:
        raise ValueError("Vocabulary must be built before creating random genomes")
    
    length = random.randint(min_length, max_length)
    token_ids = []
    
    for _ in range(length):
        # Bias towards more frequent tokens
        token_id = vocabulary.get_random_token_id()
        token_ids.append(token_id)
    
    return PromptGenome.from_tokens(token_ids)


if __name__ == "__main__":
    # Test the PromptGenome class
    print("Testing PromptGenome class...")

    # Load vocabulary first
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print("Vocabulary loaded successfully")
    else:
        print("Vocabulary not found, creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()

    # Test creation from text
    test_text = "Let's solve this step by step to find the answer."
    genome1 = PromptGenome.from_text(test_text)
    print(f"Created genome from text: {genome1}")
    
    # Test creation from tokens
    token_ids = [1, 2, 3, 4, 5]
    genome2 = PromptGenome.from_tokens(token_ids)
    print(f"Created genome from tokens: {genome2}")
    
    # Test copying
    genome3 = genome1.copy()
    print(f"Copied genome: {genome3}")
    print(f"Are they equal? {genome1 == genome3}")
    print(f"Same ID? {genome1.genome_id == genome3.genome_id}")
    
    # Test diversity calculation
    diversity = genome1.get_diversity_score(genome2)
    print(f"Diversity between genomes: {diversity}")
    
    # Test statistics
    stats = genome1.get_statistics()
    print(f"Genome statistics: {stats}")
    
    # Test validation
    print(f"Genome is valid: {genome1.validate()}")
    
    print("\n✅ PromptGenome class tests completed!")
