"""
Mutation operators for genetic algorithm.
Implements two-level mutation system with semantic neighborhoods.
"""

import random
import time
from typing import List, Optional, Dict, Any
import logging

from src.genetics.genome import PromptGenome
from src.embeddings.semantic_utils import SemanticMutator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class MutationOperator:
    """Handles mutation operations on genomes."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize mutation operator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.semantic_mutator = SemanticMutator(config_path)
        
        # Configuration
        self.population_mutation_prob = self.config.get('mutation.population_mutation_prob', 0.8)
        self.token_mutation_prob = self.config.get('mutation.token_mutation_prob', 0.002)
        self.semantic_neighbor_prob = self.config.get('mutation.semantic_neighbor_prob', 0.9)
        self.stagnation_multiplier = self.config.get('mutation.stagnation_mutation_multiplier', 2.0)

        # Random seed and semantic mutation control
        self.use_random_seed_prompts = self.config.get('genetic_algorithm.use_random_seed_prompts', False)
        self.force_disable_semantic_mutation = self.config.get('genetic_algorithm.force_disable_semantic_mutation', False)

        # Determine if semantic mutation should be disabled
        self.semantic_mutation_disabled = self._should_disable_semantic_mutation()
        
        # Tracking
        self.mutation_stats = {
            'total_mutations': 0,
            'semantic_mutations': 0,
            'random_mutations': 0,
            'genomes_mutated': 0
        }

        # Log semantic mutation status
        if self.semantic_mutation_disabled:
            logger.info("Semantic mutation DISABLED - using pure random mutations only")
            if self.use_random_seed_prompts:
                logger.info("  Reason: Random seed prompts enabled (automatic)")
            if self.force_disable_semantic_mutation:
                logger.info("  Reason: Manual override (force_disable_semantic_mutation=true)")
        else:
            logger.info(f"Semantic mutation ENABLED - semantic_prob={self.semantic_neighbor_prob}")

    def _should_disable_semantic_mutation(self) -> bool:
        """
        Determine if semantic mutation should be disabled based on configuration.

        Returns:
            True if semantic mutation should be disabled
        """
        # Automatic disabling: if random seed prompts are used
        if self.use_random_seed_prompts:
            return True

        # Manual override: force disable regardless of seed type
        if self.force_disable_semantic_mutation:
            return True

        # Default: semantic mutation enabled
        return False
    
    def should_mutate_genome(self, is_stagnant: bool = False) -> bool:
        """
        Determine if a genome should be mutated based on population-level probability.
        
        Args:
            is_stagnant: Whether the population is stagnant
            
        Returns:
            True if genome should be mutated
        """
        prob = self.population_mutation_prob
        if is_stagnant:
            prob = min(1.0, prob * self.stagnation_multiplier)
        
        return random.random() < prob
    
    def mutate_genome(self, genome: PromptGenome, is_stagnant: bool = False) -> PromptGenome:
        """
        Mutate a genome using two-level mutation system.
        
        Args:
            genome: Genome to mutate
            is_stagnant: Whether the population is stagnant
            
        Returns:
            Mutated genome (new instance)
        """
        # First level: decide if this genome should be mutated
        if not self.should_mutate_genome(is_stagnant):
            return genome.copy()
        
        # Create a copy for mutation
        mutated_genome = genome.copy()
        
        # Calculate effective token mutation probability
        token_prob = self.token_mutation_prob
        if is_stagnant:
            token_prob = min(0.1, token_prob * self.stagnation_multiplier)
        
        # Second level: mutate individual tokens
        mutations_made = 0
        semantic_mutations = 0
        
        for i, token in enumerate(mutated_genome.tokens):
            if random.random() < token_prob:
                # Mutate this token
                if self.semantic_mutation_disabled:
                    # Pure random mutation - no semantic neighborhoods
                    new_token = self.semantic_mutator.mutate_word(token, semantic_prob=0.0)
                else:
                    # Normal semantic mutation
                    new_token = self.semantic_mutator.mutate_word(
                        token,
                        semantic_prob=self.semantic_neighbor_prob
                    )
                
                if new_token != token:
                    mutated_genome.tokens[i] = new_token
                    mutations_made += 1
                    
                    # Check if it was a semantic mutation
                    if self.semantic_mutator.is_valid_word(token):
                        neighbors = self.semantic_mutator.get_similar_words(token)
                        if new_token in neighbors:
                            semantic_mutations += 1
        
        # Update genome metadata
        mutated_genome.increment_mutation_count(mutations_made)
        mutated_genome.metadata['last_mutation_count'] = mutations_made
        mutated_genome.metadata['last_semantic_mutations'] = semantic_mutations
        mutated_genome.metadata['mutation_timestamp'] = time.time()
        
        # Update statistics
        if mutations_made > 0:
            self.mutation_stats['genomes_mutated'] += 1
            self.mutation_stats['total_mutations'] += mutations_made
            self.mutation_stats['semantic_mutations'] += semantic_mutations
            self.mutation_stats['random_mutations'] += (mutations_made - semantic_mutations)
        
        logger.debug(f"Mutated genome: {mutations_made} changes "
                    f"({semantic_mutations} semantic, {mutations_made - semantic_mutations} random)")
        
        return mutated_genome

    def mutate_population(self, population: List[PromptGenome],
                         is_stagnant: bool = False) -> List[PromptGenome]:
        """
        Mutate an entire population.

        Args:
            population: List of genomes to mutate
            is_stagnant: Whether the population is stagnant

        Returns:
            List of mutated genomes
        """
        mutated_population = []

        for genome in population:
            mutated_genome = self.mutate_genome(genome, is_stagnant)
            mutated_population.append(mutated_genome)

        logger.info(f"Mutated population: {self.mutation_stats['genomes_mutated']} genomes affected")
        return mutated_population

    def targeted_mutation(self, genome: PromptGenome, target_positions: List[int],
                         semantic_prob: Optional[float] = None) -> PromptGenome:
        """
        Perform targeted mutation at specific positions.

        Args:
            genome: Genome to mutate
            target_positions: List of token positions to mutate
            semantic_prob: Override semantic probability

        Returns:
            Mutated genome
        """
        mutated_genome = genome.copy()
        mutations_made = 0
        semantic_mutations = 0

        if semantic_prob is None:
            semantic_prob = self.semantic_neighbor_prob

        for pos in target_positions:
            if 0 <= pos < len(mutated_genome.tokens):
                original_token = mutated_genome.tokens[pos]
                if self.semantic_mutation_disabled:
                    # Pure random mutation - no semantic neighborhoods
                    new_token = self.semantic_mutator.mutate_word(original_token, semantic_prob=0.0)
                else:
                    # Normal semantic mutation
                    new_token = self.semantic_mutator.mutate_word(original_token, semantic_prob)

                if new_token != original_token:
                    mutated_genome.tokens[pos] = new_token
                    mutations_made += 1

                    # Check if semantic
                    if self.semantic_mutator.is_valid_word(original_token):
                        neighbors = self.semantic_mutator.get_similar_words(original_token)
                        if new_token in neighbors:
                            semantic_mutations += 1

        # Update metadata
        mutated_genome.increment_mutation_count(mutations_made)
        mutated_genome.metadata['targeted_mutation'] = True
        mutated_genome.metadata['target_positions'] = target_positions
        mutated_genome.metadata['last_mutation_count'] = mutations_made

        return mutated_genome

    def adaptive_mutation(self, genome: PromptGenome, fitness_history: List[float],
                         generation: int) -> PromptGenome:
        """
        Perform adaptive mutation based on fitness history.

        Args:
            genome: Genome to mutate
            fitness_history: Recent fitness history
            generation: Current generation

        Returns:
            Mutated genome
        """
        # Calculate adaptive parameters
        is_improving = len(fitness_history) >= 2 and fitness_history[-1] > fitness_history[-2]
        stagnation_detected = (len(fitness_history) >= 5 and
                             max(fitness_history[-5:]) - min(fitness_history[-5:]) < 0.001)

        # Adjust mutation rates
        if is_improving:
            # Reduce mutation when improving
            effective_token_prob = self.token_mutation_prob * 0.5
        elif stagnation_detected:
            # Increase mutation when stagnant
            effective_token_prob = min(0.1, self.token_mutation_prob * self.stagnation_multiplier)
        else:
            # Normal mutation
            effective_token_prob = self.token_mutation_prob

        # Create mutated genome
        mutated_genome = genome.copy()
        mutations_made = 0

        for i, token in enumerate(mutated_genome.tokens):
            if random.random() < effective_token_prob:
                if self.semantic_mutation_disabled:
                    # Pure random mutation - no semantic neighborhoods
                    new_token = self.semantic_mutator.mutate_word(token, semantic_prob=0.0)
                else:
                    # Normal semantic mutation
                    new_token = self.semantic_mutator.mutate_word(token, self.semantic_neighbor_prob)
                if new_token != token:
                    mutated_genome.tokens[i] = new_token
                    mutations_made += 1

        # Update metadata
        mutated_genome.increment_mutation_count(mutations_made)
        mutated_genome.metadata['adaptive_mutation'] = True
        mutated_genome.metadata['effective_token_prob'] = effective_token_prob
        mutated_genome.metadata['is_improving'] = is_improving
        mutated_genome.metadata['stagnation_detected'] = stagnation_detected

        return mutated_genome

    def get_mutation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive mutation statistics including semantic mutation status.

        Returns:
            Dictionary with mutation statistics
        """
        stats = self.mutation_stats.copy()

        # Add semantic mutation configuration info
        stats['semantic_mutation_disabled'] = self.semantic_mutation_disabled
        stats['use_random_seed_prompts'] = self.use_random_seed_prompts
        stats['force_disable_semantic_mutation'] = self.force_disable_semantic_mutation

        # Calculate rates
        total_mutations = stats['total_mutations']
        if total_mutations > 0:
            stats['semantic_rate'] = stats['semantic_mutations'] / total_mutations
            stats['random_rate'] = stats['random_mutations'] / total_mutations
        else:
            stats['semantic_rate'] = 0.0
            stats['random_rate'] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset mutation statistics."""
        self.mutation_stats = {
            'total_mutations': 0,
            'semantic_mutations': 0,
            'random_mutations': 0,
            'genomes_mutated': 0
        }
