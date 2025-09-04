"""
PromptGenome class - the fundamental unit of evolution in the genetic algorithm.
Represents a prompt as a sequence of tokens with associated metadata.
"""

import hashlib
import uuid
from typing import List, Optional, Dict, Any
import logging

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class PromptGenome:
    """
    Represents a prompt genome in the genetic algorithm.
    
    Each genome contains:
    - Token sequence (limited to max_length)
    - Fitness and accuracy scores
    - Generation and genealogy information
    - Mutation and evaluation tracking
    """
    
    def __init__(self, tokens: List[str], max_length: Optional[int] = None,
                 config_path: str = "configs/experiment_config.json"):
        """
        Initialize a PromptGenome.
        
        Args:
            tokens: List of token strings
            max_length: Maximum length limit (from config if None)
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Set max length from config if not provided
        if max_length is None:
            max_length = self.config.get('genome.max_length', 200)
        
        # Core genome data
        self.tokens = tokens[:max_length]  # Truncate if necessary
        self.max_length = max_length
        
        # Performance metrics
        self.fitness: Optional[float] = None
        self.accuracy: Optional[float] = None
        
        # Genealogy and tracking
        self.generation_born: int = 0
        self.parent_ids: List[str] = []
        self.genome_id: str = str(uuid.uuid4())
        
        # Evolution tracking
        self.mutation_count: int = 0
        self.evaluation_count: int = 0
        self.crossover_count: int = 0
        
        # Metadata
        self.created_timestamp: Optional[float] = None
        self.last_evaluated_timestamp: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
    
    def to_text(self) -> str:
        """
        Convert token sequence to text string.
        
        Returns:
            Text representation of the genome
        """
        return " ".join(self.tokens)
    
    def from_text(self, text: str) -> 'PromptGenome':
        """
        Create a new genome from text string.
        
        Args:
            text: Text to convert to tokens
            
        Returns:
            New PromptGenome instance
        """
        # Simple whitespace tokenization
        # In a more sophisticated implementation, you might use a proper tokenizer
        tokens = text.strip().split()
        
        new_genome = PromptGenome(tokens, self.max_length)
        new_genome.generation_born = self.generation_born
        
        return new_genome
    
    def get_hash(self) -> str:
        """
        Generate a hash of the genome for caching and deduplication.
        
        Returns:
            MD5 hash of the token sequence
        """
        token_string = " ".join(self.tokens)
        return hashlib.md5(token_string.encode('utf-8')).hexdigest()
    
    def length(self) -> int:
        """Get the length of the token sequence."""
        return len(self.tokens)
    
    def is_empty(self) -> bool:
        """Check if the genome is empty."""
        return len(self.tokens) == 0
    
    def copy(self) -> 'PromptGenome':
        """
        Create a deep copy of the genome.
        
        Returns:
            New PromptGenome instance with copied data
        """
        new_genome = PromptGenome(self.tokens.copy(), self.max_length)
        
        # Copy all attributes
        new_genome.fitness = self.fitness
        new_genome.accuracy = self.accuracy
        new_genome.generation_born = self.generation_born
        new_genome.parent_ids = self.parent_ids.copy()
        new_genome.mutation_count = self.mutation_count
        new_genome.evaluation_count = self.evaluation_count
        new_genome.crossover_count = self.crossover_count
        new_genome.created_timestamp = self.created_timestamp
        new_genome.last_evaluated_timestamp = self.last_evaluated_timestamp
        new_genome.metadata = self.metadata.copy()
        
        # Generate new ID for the copy
        new_genome.genome_id = str(uuid.uuid4())
        
        return new_genome
    
    def set_fitness(self, fitness: float, accuracy: Optional[float] = None) -> None:
        """
        Set fitness and optionally accuracy scores.
        
        Args:
            fitness: Fitness score
            accuracy: Optional accuracy score
        """
        self.fitness = fitness
        if accuracy is not None:
            self.accuracy = accuracy
    
    def add_parent(self, parent_id: str) -> None:
        """
        Add a parent ID to the genealogy.
        
        Args:
            parent_id: ID of the parent genome
        """
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
    
    def set_parents(self, parent_ids: List[str]) -> None:
        """
        Set the parent IDs.
        
        Args:
            parent_ids: List of parent genome IDs
        """
        self.parent_ids = parent_ids.copy()
    
    def increment_mutation_count(self, count: int = 1) -> None:
        """
        Increment the mutation count.
        
        Args:
            count: Number to add to mutation count
        """
        self.mutation_count += count
    
    def increment_evaluation_count(self, count: int = 1) -> None:
        """
        Increment the evaluation count.
        
        Args:
            count: Number to add to evaluation count
        """
        self.evaluation_count += count
    
    def increment_crossover_count(self, count: int = 1) -> None:
        """
        Increment the crossover count.
        
        Args:
            count: Number to add to crossover count
        """
        self.crossover_count += count
    
    def get_genealogy_depth(self) -> int:
        """
        Get the genealogy depth (number of parent generations).
        
        Returns:
            Depth of genealogy
        """
        return len(self.parent_ids)
    
    def is_evaluated(self) -> bool:
        """Check if the genome has been evaluated."""
        return self.fitness is not None
    
    def get_length_penalty(self) -> float:
        """
        Calculate length penalty based on configuration.
        
        Returns:
            Length penalty factor (1.0 = no penalty, <1.0 = penalty applied)
        """
        penalty_threshold = self.config.get('genome.length_penalty_threshold', 200)
        max_penalty = self.config.get('genome.max_length_penalty', 0.1)
        
        if len(self.tokens) <= penalty_threshold:
            return 1.0
        
        excess_tokens = len(self.tokens) - penalty_threshold
        penalty_factor = min(max_penalty, (excess_tokens / 300) * max_penalty)
        
        return 1.0 - penalty_factor
    
    def apply_length_penalty(self) -> None:
        """Apply length penalty to fitness if fitness is set."""
        if self.fitness is not None:
            penalty_factor = self.get_length_penalty()
            self.fitness *= penalty_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert genome to dictionary representation.
        
        Returns:
            Dictionary representation of the genome
        """
        return {
            'genome_id': self.genome_id,
            'tokens': self.tokens,
            'text': self.to_text(),
            'length': self.length(),
            'fitness': self.fitness,
            'accuracy': self.accuracy,
            'generation_born': self.generation_born,
            'parent_ids': self.parent_ids,
            'mutation_count': self.mutation_count,
            'evaluation_count': self.evaluation_count,
            'crossover_count': self.crossover_count,
            'created_timestamp': self.created_timestamp,
            'last_evaluated_timestamp': self.last_evaluated_timestamp,
            'metadata': self.metadata,
            'hash': self.get_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: str = "configs/experiment_config.json") -> 'PromptGenome':
        """
        Create genome from dictionary representation.
        
        Args:
            data: Dictionary representation
            config_path: Path to configuration file
            
        Returns:
            PromptGenome instance
        """
        genome = cls(data['tokens'], config_path=config_path)
        
        # Restore all attributes
        genome.genome_id = data.get('genome_id', genome.genome_id)
        genome.fitness = data.get('fitness')
        genome.accuracy = data.get('accuracy')
        genome.generation_born = data.get('generation_born', 0)
        genome.parent_ids = data.get('parent_ids', [])
        genome.mutation_count = data.get('mutation_count', 0)
        genome.evaluation_count = data.get('evaluation_count', 0)
        genome.crossover_count = data.get('crossover_count', 0)
        genome.created_timestamp = data.get('created_timestamp')
        genome.last_evaluated_timestamp = data.get('last_evaluated_timestamp')
        genome.metadata = data.get('metadata', {})
        
        return genome
    
    def __str__(self) -> str:
        """String representation of the genome."""
        status = f"fit={self.fitness:.4f}" if self.fitness is not None else "uneval"
        return f"Genome({self.genome_id[:8]}, len={self.length()}, {status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PromptGenome(id={self.genome_id}, tokens={len(self.tokens)}, "
                f"fitness={self.fitness}, generation={self.generation_born})")
    
    def __eq__(self, other) -> bool:
        """Check equality based on token sequence."""
        if not isinstance(other, PromptGenome):
            return False
        return self.tokens == other.tokens
    
    def __hash__(self) -> int:
        """Hash based on token sequence."""
        return hash(tuple(self.tokens))
