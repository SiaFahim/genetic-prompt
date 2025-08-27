"""
Crossover operations for genetic algorithm prompt evolution.
"""

import random
from typing import List, Tuple, Optional
from enum import Enum

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome
    from src.utils.config import config
    from src.embeddings.vocabulary import vocabulary
else:
    from .genome import PromptGenome
    from ..utils.config import config
    from ..embeddings.vocabulary import vocabulary


class CrossoverType(Enum):
    """Types of crossover operations."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SEMANTIC_BLEND = "semantic_blend"


def single_point_crossover(parent1: PromptGenome, parent2: PromptGenome) -> Tuple[PromptGenome, PromptGenome]:
    """
    Perform single-point crossover between two parent genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        
    Returns:
        Tuple of two offspring genomes
    """
    if parent1.is_empty() or parent2.is_empty():
        # If either parent is empty, return copies
        return parent1.copy(), parent2.copy()
    
    # Choose crossover point
    min_len = min(parent1.length(), parent2.length())
    if min_len <= 1:
        # Too short for meaningful crossover
        return parent1.copy(), parent2.copy()
    
    crossover_point = random.randint(1, min_len - 1)
    
    # Create offspring
    offspring1_tokens = (parent1.token_ids[:crossover_point] + 
                        parent2.token_ids[crossover_point:])
    offspring2_tokens = (parent2.token_ids[:crossover_point] + 
                        parent1.token_ids[crossover_point:])
    
    # Create new genomes
    offspring1 = PromptGenome.from_tokens(offspring1_tokens)
    offspring2 = PromptGenome.from_tokens(offspring2_tokens)
    
    # Set genealogy information
    offspring1.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring2.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring1.generation = max(parent1.generation, parent2.generation) + 1
    offspring2.generation = max(parent1.generation, parent2.generation) + 1
    
    return offspring1, offspring2


def two_point_crossover(parent1: PromptGenome, parent2: PromptGenome) -> Tuple[PromptGenome, PromptGenome]:
    """
    Perform two-point crossover between two parent genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        
    Returns:
        Tuple of two offspring genomes
    """
    if parent1.is_empty() or parent2.is_empty():
        return parent1.copy(), parent2.copy()
    
    min_len = min(parent1.length(), parent2.length())
    if min_len <= 2:
        # Fall back to single-point crossover
        return single_point_crossover(parent1, parent2)
    
    # Choose two crossover points
    point1 = random.randint(1, min_len - 2)
    point2 = random.randint(point1 + 1, min_len - 1)
    
    # Create offspring by swapping middle segment
    offspring1_tokens = (parent1.token_ids[:point1] + 
                        parent2.token_ids[point1:point2] + 
                        parent1.token_ids[point2:])
    offspring2_tokens = (parent2.token_ids[:point1] + 
                        parent1.token_ids[point1:point2] + 
                        parent2.token_ids[point2:])
    
    # Create new genomes
    offspring1 = PromptGenome.from_tokens(offspring1_tokens)
    offspring2 = PromptGenome.from_tokens(offspring2_tokens)
    
    # Set genealogy information
    offspring1.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring2.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring1.generation = max(parent1.generation, parent2.generation) + 1
    offspring2.generation = max(parent1.generation, parent2.generation) + 1
    
    return offspring1, offspring2


def uniform_crossover(parent1: PromptGenome, parent2: PromptGenome, 
                     crossover_prob: float = 0.5) -> Tuple[PromptGenome, PromptGenome]:
    """
    Perform uniform crossover between two parent genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        crossover_prob: Probability of taking gene from parent1 vs parent2
        
    Returns:
        Tuple of two offspring genomes
    """
    if parent1.is_empty() or parent2.is_empty():
        return parent1.copy(), parent2.copy()
    
    # Use the shorter length for uniform crossover
    min_len = min(parent1.length(), parent2.length())
    
    offspring1_tokens = []
    offspring2_tokens = []
    
    for i in range(min_len):
        if random.random() < crossover_prob:
            # Take from parent1 for offspring1, parent2 for offspring2
            offspring1_tokens.append(parent1.token_ids[i])
            offspring2_tokens.append(parent2.token_ids[i])
        else:
            # Take from parent2 for offspring1, parent1 for offspring2
            offspring1_tokens.append(parent2.token_ids[i])
            offspring2_tokens.append(parent1.token_ids[i])
    
    # Handle remaining tokens from longer parent
    if parent1.length() > min_len:
        remaining = parent1.token_ids[min_len:]
        if random.random() < 0.5:
            offspring1_tokens.extend(remaining)
        else:
            offspring2_tokens.extend(remaining)
    elif parent2.length() > min_len:
        remaining = parent2.token_ids[min_len:]
        if random.random() < 0.5:
            offspring1_tokens.extend(remaining)
        else:
            offspring2_tokens.extend(remaining)
    
    # Create new genomes
    offspring1 = PromptGenome.from_tokens(offspring1_tokens)
    offspring2 = PromptGenome.from_tokens(offspring2_tokens)
    
    # Set genealogy information
    offspring1.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring2.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring1.generation = max(parent1.generation, parent2.generation) + 1
    offspring2.generation = max(parent1.generation, parent2.generation) + 1
    
    return offspring1, offspring2


def semantic_blend_crossover(parent1: PromptGenome, parent2: PromptGenome) -> Tuple[PromptGenome, PromptGenome]:
    """
    Perform semantic blend crossover that tries to preserve meaning.
    
    This is a more sophisticated crossover that attempts to maintain
    semantic coherence by preferring crossover points at word boundaries.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        
    Returns:
        Tuple of two offspring genomes
    """
    if parent1.is_empty() or parent2.is_empty():
        return parent1.copy(), parent2.copy()
    
    # For now, implement as single-point crossover with bias towards
    # crossover points that maintain semantic structure
    # This could be enhanced with actual semantic analysis
    
    min_len = min(parent1.length(), parent2.length())
    if min_len <= 1:
        return parent1.copy(), parent2.copy()
    
    # Prefer crossover points in the middle third of the sequence
    start_range = max(1, min_len // 3)
    end_range = min(min_len - 1, 2 * min_len // 3)
    
    if start_range >= end_range:
        crossover_point = min_len // 2
    else:
        crossover_point = random.randint(start_range, end_range)
    
    # Create offspring
    offspring1_tokens = (parent1.token_ids[:crossover_point] + 
                        parent2.token_ids[crossover_point:])
    offspring2_tokens = (parent2.token_ids[:crossover_point] + 
                        parent1.token_ids[crossover_point:])
    
    # Create new genomes
    offspring1 = PromptGenome.from_tokens(offspring1_tokens)
    offspring2 = PromptGenome.from_tokens(offspring2_tokens)
    
    # Set genealogy information
    offspring1.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring2.parent_ids = [parent1.genome_id, parent2.genome_id]
    offspring1.generation = max(parent1.generation, parent2.generation) + 1
    offspring2.generation = max(parent1.generation, parent2.generation) + 1
    
    return offspring1, offspring2


def crossover(parent1: PromptGenome, parent2: PromptGenome, 
              crossover_type: CrossoverType = CrossoverType.SINGLE_POINT) -> Tuple[PromptGenome, PromptGenome]:
    """
    Perform crossover between two parent genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        crossover_type: Type of crossover to perform
        
    Returns:
        Tuple of two offspring genomes
    """
    if crossover_type == CrossoverType.SINGLE_POINT:
        return single_point_crossover(parent1, parent2)
    elif crossover_type == CrossoverType.TWO_POINT:
        return two_point_crossover(parent1, parent2)
    elif crossover_type == CrossoverType.UNIFORM:
        return uniform_crossover(parent1, parent2)
    elif crossover_type == CrossoverType.SEMANTIC_BLEND:
        return semantic_blend_crossover(parent1, parent2)
    else:
        raise ValueError(f"Unknown crossover type: {crossover_type}")


if __name__ == "__main__":
    # Test crossover operations
    print("Testing crossover operations...")
    
    # Load vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print("Vocabulary loaded successfully")
    else:
        print("Vocabulary not found, creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
    
    # Create test parents
    parent1 = PromptGenome.from_text("Let's solve this problem step by step.")
    parent2 = PromptGenome.from_text("Think carefully about the answer.")
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    # Test all crossover types
    crossover_types = [
        CrossoverType.SINGLE_POINT,
        CrossoverType.TWO_POINT,
        CrossoverType.UNIFORM,
        CrossoverType.SEMANTIC_BLEND
    ]
    
    for crossover_type in crossover_types:
        print(f"\n--- {crossover_type.value.upper()} CROSSOVER ---")
        
        for i in range(3):  # Generate 3 examples
            offspring1, offspring2 = crossover(parent1, parent2, crossover_type)
            print(f"  Trial {i+1}:")
            print(f"    Offspring 1: {offspring1}")
            print(f"    Offspring 2: {offspring2}")
    
    print("\n✅ Crossover operations tests completed!")
