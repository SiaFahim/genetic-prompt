"""
Mutation operations for genetic algorithm prompt evolution.
"""

import random
from typing import List, Optional
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
    from src.embeddings.neighborhoods import semantic_neighborhoods
    from src.config.hyperparameters import get_hyperparameter_config
else:
    from .genome import PromptGenome
    from ..utils.config import config
    from ..embeddings.vocabulary import vocabulary
    from ..embeddings.neighborhoods import semantic_neighborhoods
    from ..config.hyperparameters import get_hyperparameter_config


class MutationType(Enum):
    """Types of mutation operations."""
    SEMANTIC = "semantic"
    RANDOM = "random"
    INSERTION = "insertion"
    DELETION = "deletion"
    SWAP = "swap"
    DUPLICATION = "duplication"


def semantic_mutation(genome: PromptGenome, mutation_rate: Optional[float] = None,
                     semantic_prob: Optional[float] = None) -> PromptGenome:
    """
    Perform semantic mutation using neighborhood information.

    Args:
        genome: Genome to mutate
        mutation_rate: Probability of mutating each token (uses hyperparameter config if None)
        semantic_prob: Probability of using semantic neighbor vs random token (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    mutation_rate = mutation_rate if mutation_rate is not None else config.mutation_rate
    semantic_prob = semantic_prob if semantic_prob is not None else config.semantic_prob
    if genome.is_empty():
        return genome.copy()
    
    mutated_genome = genome.copy()
    mutations_made = []
    
    for i in range(len(mutated_genome.token_ids)):
        if random.random() < mutation_rate:
            original_token_id = mutated_genome.token_ids[i]
            
            # Get semantic neighbor
            if semantic_neighborhoods.neighborhoods_built:
                new_token_id = semantic_neighborhoods.get_neighbor(
                    original_token_id, semantic_prob
                )
            else:
                # Fallback to random mutation
                new_token_id = vocabulary.get_random_token_id()
            
            mutated_genome.token_ids[i] = new_token_id
            mutations_made.append({
                'position': i,
                'original': original_token_id,
                'new': new_token_id,
                'type': 'semantic'
            })
    
    # Record mutation history
    if mutations_made:
        mutated_genome.mutation_history.append({
            'type': 'semantic',
            'mutations': mutations_made,
            'generation': mutated_genome.generation
        })
    
    return mutated_genome


def random_mutation(genome: PromptGenome, mutation_rate: Optional[float] = None) -> PromptGenome:
    """
    Perform random mutation by replacing tokens with random ones.

    Args:
        genome: Genome to mutate
        mutation_rate: Probability of mutating each token (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    mutation_rate = mutation_rate if mutation_rate is not None else config.mutation_rate
    if genome.is_empty():
        return genome.copy()
    
    mutated_genome = genome.copy()
    mutations_made = []
    
    for i in range(len(mutated_genome.token_ids)):
        if random.random() < mutation_rate:
            original_token_id = mutated_genome.token_ids[i]
            new_token_id = vocabulary.get_random_token_id()
            
            mutated_genome.token_ids[i] = new_token_id
            mutations_made.append({
                'position': i,
                'original': original_token_id,
                'new': new_token_id,
                'type': 'random'
            })
    
    # Record mutation history
    if mutations_made:
        mutated_genome.mutation_history.append({
            'type': 'random',
            'mutations': mutations_made,
            'generation': mutated_genome.generation
        })
    
    return mutated_genome


def insertion_mutation(genome: PromptGenome, insertion_rate: Optional[float] = None,
                      max_insertions: Optional[int] = None) -> PromptGenome:
    """
    Perform insertion mutation by adding new tokens.

    Args:
        genome: Genome to mutate
        insertion_rate: Probability of insertion at each position (uses hyperparameter config if None)
        max_insertions: Maximum number of insertions (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    insertion_rate = insertion_rate if insertion_rate is not None else config.insertion_rate
    max_insertions = max_insertions if max_insertions is not None else config.max_insertions
    if genome.length() == 0:
        # Insert into empty genome
        mutated_genome = genome.copy()
        new_token_id = vocabulary.get_random_token_id()
        mutated_genome.token_ids = [new_token_id]
        mutated_genome.mutation_history.append({
            'type': 'insertion',
            'mutations': [{'position': 0, 'inserted': new_token_id}],
            'generation': mutated_genome.generation
        })
        return mutated_genome
    
    mutated_genome = genome.copy()
    insertions_made = []
    insertions_count = 0
    
    # Go through positions (including after last token)
    i = 0
    while i <= len(mutated_genome.token_ids) and insertions_count < max_insertions:
        if random.random() < insertion_rate:
            # Insert a new token at position i
            new_token_id = vocabulary.get_random_token_id()
            mutated_genome.token_ids.insert(i, new_token_id)
            
            insertions_made.append({
                'position': i,
                'inserted': new_token_id,
                'type': 'insertion'
            })
            insertions_count += 1
            i += 1  # Skip the inserted token
        
        i += 1
    
    # Record mutation history
    if insertions_made:
        mutated_genome.mutation_history.append({
            'type': 'insertion',
            'mutations': insertions_made,
            'generation': mutated_genome.generation
        })
    
    return mutated_genome


def deletion_mutation(genome: PromptGenome, deletion_rate: Optional[float] = None,
                     min_length: Optional[int] = None) -> PromptGenome:
    """
    Perform deletion mutation by removing tokens.

    Args:
        genome: Genome to mutate
        deletion_rate: Probability of deleting each token (uses hyperparameter config if None)
        min_length: Minimum length to maintain (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    deletion_rate = deletion_rate if deletion_rate is not None else config.deletion_rate
    min_length = min_length if min_length is not None else config.min_genome_length
    if genome.length() <= min_length:
        return genome.copy()
    
    mutated_genome = genome.copy()
    deletions_made = []
    
    # Go through tokens in reverse order to avoid index issues
    for i in range(len(mutated_genome.token_ids) - 1, -1, -1):
        if len(mutated_genome.token_ids) <= min_length:
            break
        
        if random.random() < deletion_rate:
            deleted_token_id = mutated_genome.token_ids[i]
            del mutated_genome.token_ids[i]
            
            deletions_made.append({
                'position': i,
                'deleted': deleted_token_id,
                'type': 'deletion'
            })
    
    # Record mutation history (reverse to maintain original order)
    if deletions_made:
        deletions_made.reverse()
        mutated_genome.mutation_history.append({
            'type': 'deletion',
            'mutations': deletions_made,
            'generation': mutated_genome.generation
        })
    
    return mutated_genome


def swap_mutation(genome: PromptGenome, swap_rate: Optional[float] = None) -> PromptGenome:
    """
    Perform swap mutation by swapping adjacent tokens.

    Args:
        genome: Genome to mutate
        swap_rate: Probability of swapping at each position (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    swap_rate = swap_rate if swap_rate is not None else config.swap_rate
    if genome.length() < 2:
        return genome.copy()
    
    mutated_genome = genome.copy()
    swaps_made = []
    
    for i in range(len(mutated_genome.token_ids) - 1):
        if random.random() < swap_rate:
            # Swap tokens at positions i and i+1
            token1 = mutated_genome.token_ids[i]
            token2 = mutated_genome.token_ids[i + 1]
            
            mutated_genome.token_ids[i] = token2
            mutated_genome.token_ids[i + 1] = token1
            
            swaps_made.append({
                'position1': i,
                'position2': i + 1,
                'token1': token1,
                'token2': token2,
                'type': 'swap'
            })
    
    # Record mutation history
    if swaps_made:
        mutated_genome.mutation_history.append({
            'type': 'swap',
            'mutations': swaps_made,
            'generation': mutated_genome.generation
        })
    
    return mutated_genome


def duplication_mutation(genome: PromptGenome, duplication_rate: Optional[float] = None,
                        max_length: Optional[int] = None) -> PromptGenome:
    """
    Perform duplication mutation by duplicating tokens or segments.

    Args:
        genome: Genome to mutate
        duplication_rate: Probability of duplication (uses hyperparameter config if None)
        max_length: Maximum genome length (uses hyperparameter config if None)

    Returns:
        Mutated genome
    """
    config = get_hyperparameter_config()
    duplication_rate = duplication_rate if duplication_rate is not None else config.duplication_rate
    max_length = max_length if max_length is not None else config.max_genome_length
    if genome.is_empty() or genome.length() >= max_length:
        return genome.copy()
    
    mutated_genome = genome.copy()
    
    if random.random() < duplication_rate:
        # Choose a random segment to duplicate
        start_pos = random.randint(0, len(mutated_genome.token_ids) - 1)
        max_segment_len = min(3, max_length - len(mutated_genome.token_ids))
        
        if max_segment_len > 0:
            segment_len = random.randint(1, max_segment_len)
            end_pos = min(start_pos + segment_len, len(mutated_genome.token_ids))
            
            # Duplicate the segment
            segment = mutated_genome.token_ids[start_pos:end_pos]
            insert_pos = random.randint(0, len(mutated_genome.token_ids))
            
            # Insert the duplicated segment
            for i, token_id in enumerate(segment):
                mutated_genome.token_ids.insert(insert_pos + i, token_id)
            
            # Record mutation history
            mutated_genome.mutation_history.append({
                'type': 'duplication',
                'mutations': [{
                    'original_start': start_pos,
                    'original_end': end_pos,
                    'insert_position': insert_pos,
                    'duplicated_tokens': segment,
                    'type': 'duplication'
                }],
                'generation': mutated_genome.generation
            })
    
    return mutated_genome


def mutate(genome: PromptGenome, mutation_type: MutationType = MutationType.SEMANTIC,
           **kwargs) -> PromptGenome:
    """
    Perform mutation on a genome.
    
    Args:
        genome: Genome to mutate
        mutation_type: Type of mutation to perform
        **kwargs: Additional arguments for specific mutation types
        
    Returns:
        Mutated genome
    """
    if mutation_type == MutationType.SEMANTIC:
        return semantic_mutation(genome, **kwargs)
    elif mutation_type == MutationType.RANDOM:
        return random_mutation(genome, **kwargs)
    elif mutation_type == MutationType.INSERTION:
        return insertion_mutation(genome, **kwargs)
    elif mutation_type == MutationType.DELETION:
        return deletion_mutation(genome, **kwargs)
    elif mutation_type == MutationType.SWAP:
        return swap_mutation(genome, **kwargs)
    elif mutation_type == MutationType.DUPLICATION:
        return duplication_mutation(genome, **kwargs)
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")


if __name__ == "__main__":
    # Test mutation operations
    print("Testing mutation operations...")
    
    # Load vocabulary and neighborhoods
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    neighborhoods_file = config.get_data_dir() / "embeddings" / "semantic_neighborhoods.pkl"
    
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print("Vocabulary loaded successfully")
    else:
        print("Vocabulary not found, creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
    
    if neighborhoods_file.exists():
        semantic_neighborhoods.load_neighborhoods(neighborhoods_file)
        semantic_neighborhoods.vocabulary = vocabulary
        print("Neighborhoods loaded successfully")
    else:
        print("Neighborhoods not found, mutations will be random")
    
    # Create test genome
    original_genome = PromptGenome.from_text("Let's solve this problem step by step carefully.")
    print(f"Original genome: {original_genome}")
    
    # Test all mutation types
    mutation_types = [
        (MutationType.SEMANTIC, {'mutation_rate': 0.3}),
        (MutationType.RANDOM, {'mutation_rate': 0.3}),
        (MutationType.INSERTION, {'insertion_rate': 0.2}),
        (MutationType.DELETION, {'deletion_rate': 0.2}),
        (MutationType.SWAP, {'swap_rate': 0.3}),
        (MutationType.DUPLICATION, {'duplication_rate': 0.5})
    ]
    
    for mutation_type, kwargs in mutation_types:
        print(f"\n--- {mutation_type.value.upper()} MUTATION ---")
        
        for i in range(3):  # Generate 3 examples
            mutated = mutate(original_genome, mutation_type, **kwargs)
            print(f"  Trial {i+1}: {mutated}")
            if mutated.mutation_history:
                last_mutation = mutated.mutation_history[-1]
                print(f"    Mutations: {len(last_mutation['mutations'])}")
    
    print("\n✅ Mutation operations tests completed!")
