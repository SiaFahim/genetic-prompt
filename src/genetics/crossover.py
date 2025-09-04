"""
Crossover operators for genetic algorithm.
Implements semantic-aware crossover with sentence boundary detection.
"""

import random
import re
from typing import List, Tuple, Optional
import logging

from src.genetics.genome import PromptGenome
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class CrossoverOperator:
    """Handles crossover operations between genomes."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize crossover operator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Configuration
        self.boundary_offset_range = self.config.get('crossover.boundary_offset_range', 5)
        self.fallback_to_midpoint = self.config.get('crossover.fallback_to_midpoint', True)
        
        # Sentence boundary patterns
        self.sentence_endings = {'.', '!', '?', ':', ';'}
        self.sentence_starters = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 
                                'i', 'you', 'he', 'she', 'it', 'we', 'they',
                                'what', 'when', 'where', 'why', 'how', 'who',
                                'first', 'second', 'third', 'next', 'then', 'finally',
                                'solve', 'calculate', 'find', 'determine', 'compute'}
    
    def find_sentence_boundaries(self, tokens: List[str]) -> List[int]:
        """
        Find potential sentence boundaries in token sequence.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of indices where sentences might end
        """
        boundaries = []
        
        for i, token in enumerate(tokens):
            # Check if token ends with sentence-ending punctuation
            if any(token.endswith(punct) for punct in self.sentence_endings):
                boundaries.append(i + 1)  # Boundary after this token
            
            # Check if next token starts a new sentence
            if (i + 1 < len(tokens) and 
                tokens[i + 1].lower() in self.sentence_starters and
                i > 0):  # Don't add boundary at the very beginning
                boundaries.append(i + 1)
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        # Filter out boundaries too close to start or end
        min_distance = 3
        filtered_boundaries = []
        
        for boundary in boundaries:
            if (boundary >= min_distance and 
                boundary <= len(tokens) - min_distance):
                filtered_boundaries.append(boundary)
        
        return filtered_boundaries
    
    def find_optimal_crossover_point(self, parent1: PromptGenome, 
                                   parent2: PromptGenome) -> int:
        """
        Find optimal crossover point considering sentence boundaries.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Crossover point index
        """
        len1, len2 = len(parent1.tokens), len(parent2.tokens)
        min_len = min(len1, len2)
        
        if min_len <= 6:  # Too short for sophisticated crossover
            return min_len // 2
        
        # Find sentence boundaries in both parents
        boundaries1 = self.find_sentence_boundaries(parent1.tokens[:min_len])
        boundaries2 = self.find_sentence_boundaries(parent2.tokens[:min_len])
        
        # Find common or nearby boundaries
        good_boundaries = []
        
        for b1 in boundaries1:
            for b2 in boundaries2:
                if abs(b1 - b2) <= self.boundary_offset_range:
                    # Use the average of nearby boundaries
                    avg_boundary = (b1 + b2) // 2
                    if 3 <= avg_boundary <= min_len - 3:
                        good_boundaries.append(avg_boundary)
        
        # If we found good boundaries, choose one randomly
        if good_boundaries:
            return random.choice(good_boundaries)
        
        # Fallback: try boundaries from either parent
        all_boundaries = boundaries1 + boundaries2
        valid_boundaries = [b for b in all_boundaries if 3 <= b <= min_len - 3]
        
        if valid_boundaries:
            return random.choice(valid_boundaries)
        
        # Final fallback: midpoint with some randomness
        if self.fallback_to_midpoint:
            midpoint = min_len // 2
            offset = random.randint(-2, 2)
            return max(3, min(min_len - 3, midpoint + offset))
        
        # Completely random point
        return random.randint(3, min_len - 3)
    
    def crossover(self, parent1: PromptGenome, parent2: PromptGenome) -> Tuple[PromptGenome, PromptGenome]:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of two offspring genomes
        """
        # Find optimal crossover point
        crossover_point = self.find_optimal_crossover_point(parent1, parent2)
        
        # Create offspring by swapping segments
        offspring1_tokens = (parent1.tokens[:crossover_point] + 
                           parent2.tokens[crossover_point:])
        offspring2_tokens = (parent2.tokens[:crossover_point] + 
                           parent1.tokens[crossover_point:])
        
        # Create new genomes
        offspring1 = PromptGenome(offspring1_tokens, parent1.max_length)
        offspring2 = PromptGenome(offspring2_tokens, parent2.max_length)
        
        # Set genealogy information
        offspring1.set_parents([parent1.genome_id, parent2.genome_id])
        offspring2.set_parents([parent1.genome_id, parent2.genome_id])
        
        # Set generation
        max_generation = max(parent1.generation_born, parent2.generation_born)
        offspring1.generation_born = max_generation + 1
        offspring2.generation_born = max_generation + 1
        
        # Update crossover counts
        offspring1.increment_crossover_count()
        offspring2.increment_crossover_count()
        
        # Add metadata about crossover
        offspring1.metadata['crossover_point'] = crossover_point
        offspring1.metadata['crossover_method'] = 'sentence_boundary'
        offspring2.metadata['crossover_point'] = crossover_point
        offspring2.metadata['crossover_method'] = 'sentence_boundary'
        
        logger.debug(f"Crossover at point {crossover_point}: "
                    f"{len(parent1.tokens)},{len(parent2.tokens)} -> "
                    f"{len(offspring1.tokens)},{len(offspring2.tokens)}")
        
        return offspring1, offspring2
    
    def uniform_crossover(self, parent1: PromptGenome, parent2: PromptGenome, 
                         swap_probability: float = 0.5) -> Tuple[PromptGenome, PromptGenome]:
        """
        Perform uniform crossover (token-by-token swapping).
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            swap_probability: Probability of swapping each token
            
        Returns:
            Tuple of two offspring genomes
        """
        len1, len2 = len(parent1.tokens), len(parent2.tokens)
        min_len = min(len1, len2)
        max_len = max(len1, len2)
        
        offspring1_tokens = []
        offspring2_tokens = []
        
        # Process common length
        for i in range(min_len):
            if random.random() < swap_probability:
                # Swap tokens
                offspring1_tokens.append(parent2.tokens[i])
                offspring2_tokens.append(parent1.tokens[i])
            else:
                # Keep original
                offspring1_tokens.append(parent1.tokens[i])
                offspring2_tokens.append(parent2.tokens[i])
        
        # Handle remaining tokens from longer parent
        if len1 > len2:
            offspring1_tokens.extend(parent1.tokens[min_len:])
            # offspring2 gets nothing extra
        elif len2 > len1:
            offspring2_tokens.extend(parent2.tokens[min_len:])
            # offspring1 gets nothing extra
        
        # Create new genomes
        offspring1 = PromptGenome(offspring1_tokens, parent1.max_length)
        offspring2 = PromptGenome(offspring2_tokens, parent2.max_length)
        
        # Set genealogy information
        offspring1.set_parents([parent1.genome_id, parent2.genome_id])
        offspring2.set_parents([parent1.genome_id, parent2.genome_id])
        
        # Set generation
        max_generation = max(parent1.generation_born, parent2.generation_born)
        offspring1.generation_born = max_generation + 1
        offspring2.generation_born = max_generation + 1
        
        # Update crossover counts
        offspring1.increment_crossover_count()
        offspring2.increment_crossover_count()
        
        # Add metadata
        offspring1.metadata['crossover_method'] = 'uniform'
        offspring1.metadata['swap_probability'] = swap_probability
        offspring2.metadata['crossover_method'] = 'uniform'
        offspring2.metadata['swap_probability'] = swap_probability
        
        return offspring1, offspring2
    
    def multi_point_crossover(self, parent1: PromptGenome, parent2: PromptGenome,
                            num_points: int = 2) -> Tuple[PromptGenome, PromptGenome]:
        """
        Perform multi-point crossover.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            num_points: Number of crossover points
            
        Returns:
            Tuple of two offspring genomes
        """
        len1, len2 = len(parent1.tokens), len(parent2.tokens)
        min_len = min(len1, len2)
        
        if min_len <= num_points + 2:
            # Fall back to single-point crossover
            return self.crossover(parent1, parent2)
        
        # Generate random crossover points
        points = sorted(random.sample(range(1, min_len), num_points))
        
        # Create offspring by alternating segments
        offspring1_tokens = []
        offspring2_tokens = []
        
        last_point = 0
        use_parent1 = True
        
        for point in points + [min_len]:
            if use_parent1:
                offspring1_tokens.extend(parent1.tokens[last_point:point])
                offspring2_tokens.extend(parent2.tokens[last_point:point])
            else:
                offspring1_tokens.extend(parent2.tokens[last_point:point])
                offspring2_tokens.extend(parent1.tokens[last_point:point])
            
            use_parent1 = not use_parent1
            last_point = point
        
        # Handle remaining tokens
        if len1 > min_len:
            offspring1_tokens.extend(parent1.tokens[min_len:])
        if len2 > min_len:
            offspring2_tokens.extend(parent2.tokens[min_len:])
        
        # Create new genomes
        offspring1 = PromptGenome(offspring1_tokens, parent1.max_length)
        offspring2 = PromptGenome(offspring2_tokens, parent2.max_length)
        
        # Set genealogy information
        offspring1.set_parents([parent1.genome_id, parent2.genome_id])
        offspring2.set_parents([parent1.genome_id, parent2.genome_id])
        
        # Set generation
        max_generation = max(parent1.generation_born, parent2.generation_born)
        offspring1.generation_born = max_generation + 1
        offspring2.generation_born = max_generation + 1
        
        # Update crossover counts
        offspring1.increment_crossover_count()
        offspring2.increment_crossover_count()
        
        # Add metadata
        offspring1.metadata['crossover_method'] = 'multi_point'
        offspring1.metadata['crossover_points'] = points
        offspring2.metadata['crossover_method'] = 'multi_point'
        offspring2.metadata['crossover_points'] = points
        
        return offspring1, offspring2
    
    def validate_crossover(self, parent1: PromptGenome, parent2: PromptGenome,
                          offspring1: PromptGenome, offspring2: PromptGenome) -> dict:
        """
        Validate crossover results and return statistics.
        
        Args:
            parent1: First parent
            parent2: Second parent
            offspring1: First offspring
            offspring2: Second offspring
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'parent1_length': len(parent1.tokens),
            'parent2_length': len(parent2.tokens),
            'offspring1_length': len(offspring1.tokens),
            'offspring2_length': len(offspring2.tokens),
            'genealogy_set': all([
                len(offspring1.parent_ids) == 2,
                len(offspring2.parent_ids) == 2,
                parent1.genome_id in offspring1.parent_ids,
                parent2.genome_id in offspring1.parent_ids,
                parent1.genome_id in offspring2.parent_ids,
                parent2.genome_id in offspring2.parent_ids
            ]),
            'generation_incremented': all([
                offspring1.generation_born > parent1.generation_born,
                offspring1.generation_born > parent2.generation_born,
                offspring2.generation_born > parent1.generation_born,
                offspring2.generation_born > parent2.generation_born
            ])
        }
        
        return stats
