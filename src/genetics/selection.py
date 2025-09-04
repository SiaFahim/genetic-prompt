"""
Selection strategies for genetic algorithm.
Implements elite, diverse, and random selection methods.
"""

import random
from typing import List, Dict, Any, Optional
import logging

from src.genetics.genome import PromptGenome
from src.embeddings.semantic_utils import SemanticMutator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class SelectionOperator:
    """Handles selection operations for genetic algorithm."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize selection operator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.semantic_mutator = SemanticMutator(config_path)
        
        # Configuration
        selection_config = self.config.get('selection', {})
        self.elite_count = selection_config.get('elite_count', 20)
        self.diverse_count = selection_config.get('diverse_count', 1)  # Updated from config
        self.random_count = selection_config.get('random_count', 1)   # Updated from config
        
        # Statistics
        self.selection_stats = {
            'total_selections': 0,
            'elite_selections': 0,
            'diverse_selections': 0,
            'random_selections': 0
        }
    
    def select_elite(self, population: List[PromptGenome], count: int) -> List[PromptGenome]:
        """
        Select top genomes by fitness.
        
        Args:
            population: Population to select from
            count: Number of genomes to select
            
        Returns:
            List of selected elite genomes
        """
        # Filter evaluated genomes and sort by fitness
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            logger.warning("No evaluated genomes for elite selection")
            return []
        
        # Sort by fitness (descending)
        sorted_genomes = sorted(evaluated_genomes, key=lambda g: g.fitness, reverse=True)
        
        # Select top count
        selected = sorted_genomes[:count]
        
        self.selection_stats['elite_selections'] += len(selected)
        
        logger.debug(f"Selected {len(selected)} elite genomes (fitness range: "
                    f"{selected[-1].fitness:.3f} - {selected[0].fitness:.3f})")
        
        return selected
    
    def select_diverse(self, population: List[PromptGenome], count: int,
                      exclude: Optional[List[PromptGenome]] = None) -> List[PromptGenome]:
        """
        Select diverse genomes based on token diversity.
        
        Args:
            population: Population to select from
            count: Number of genomes to select
            exclude: Genomes to exclude from selection
            
        Returns:
            List of selected diverse genomes
        """
        if exclude is None:
            exclude = []
        
        exclude_ids = {g.genome_id for g in exclude}
        candidates = [g for g in population if g.genome_id not in exclude_ids and g.fitness is not None]
        
        if not candidates:
            logger.warning("No candidates for diverse selection")
            return []
        
        if len(candidates) <= count:
            self.selection_stats['diverse_selections'] += len(candidates)
            return candidates
        
        # Start with a random genome
        selected = [random.choice(candidates)]
        remaining = [g for g in candidates if g.genome_id != selected[0].genome_id]
        
        # Iteratively select most diverse genomes
        while len(selected) < count and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score as minimum Jaccard distance to selected genomes
                min_distance = float('inf')
                candidate_tokens = set(candidate.tokens)
                
                for selected_genome in selected:
                    selected_tokens = set(selected_genome.tokens)
                    
                    intersection = len(candidate_tokens & selected_tokens)
                    union = len(candidate_tokens | selected_tokens)
                    
                    if union > 0:
                        jaccard_distance = 1 - (intersection / union)
                        min_distance = min(min_distance, jaccard_distance)
                
                if min_distance > best_diversity_score:
                    best_diversity_score = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining = [g for g in remaining if g.genome_id != best_candidate.genome_id]
            else:
                break
        
        self.selection_stats['diverse_selections'] += len(selected)
        
        logger.debug(f"Selected {len(selected)} diverse genomes")
        
        return selected
    
    def select_random(self, population: List[PromptGenome], count: int,
                     exclude: Optional[List[PromptGenome]] = None) -> List[PromptGenome]:
        """
        Select random genomes from population.
        
        Args:
            population: Population to select from
            count: Number of genomes to select
            exclude: Genomes to exclude from selection
            
        Returns:
            List of randomly selected genomes
        """
        if exclude is None:
            exclude = []
        
        exclude_ids = {g.genome_id for g in exclude}
        candidates = [g for g in population if g.genome_id not in exclude_ids and g.fitness is not None]
        
        if not candidates:
            logger.warning("No candidates for random selection")
            return []
        
        # Random selection
        selected_count = min(count, len(candidates))
        selected = random.sample(candidates, selected_count)
        
        self.selection_stats['random_selections'] += len(selected)
        
        logger.debug(f"Selected {len(selected)} random genomes")
        
        return selected
    
    def tournament_selection(self, population: List[PromptGenome], count: int,
                           tournament_size: int = 3) -> List[PromptGenome]:
        """
        Select genomes using tournament selection.
        
        Args:
            population: Population to select from
            count: Number of genomes to select
            tournament_size: Size of each tournament
            
        Returns:
            List of selected genomes
        """
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            return []
        
        selected = []
        
        for _ in range(count):
            # Run tournament
            tournament_candidates = random.sample(
                evaluated_genomes, 
                min(tournament_size, len(evaluated_genomes))
            )
            
            # Select best from tournament
            winner = max(tournament_candidates, key=lambda g: g.fitness)
            selected.append(winner)
        
        return selected
    
    def select_parents(self, population: List[PromptGenome]) -> List[PromptGenome]:
        """
        Select parents for next generation using multi-strategy approach.
        
        Args:
            population: Current population
            
        Returns:
            List of selected parent genomes
        """
        logger.info(f"Selecting parents: {self.elite_count} elite, "
                   f"{self.diverse_count} diverse, {self.random_count} random")
        
        selected_parents = []
        
        # Elite selection
        elite_parents = self.select_elite(population, self.elite_count)
        selected_parents.extend(elite_parents)
        
        # Diverse selection (excluding already selected)
        diverse_parents = self.select_diverse(population, self.diverse_count, selected_parents)
        selected_parents.extend(diverse_parents)
        
        # Random selection (excluding already selected)
        random_parents = self.select_random(population, self.random_count, selected_parents)
        selected_parents.extend(random_parents)
        
        self.selection_stats['total_selections'] += len(selected_parents)
        
        logger.info(f"Selected {len(selected_parents)} parents total")
        
        return selected_parents
    
    def select_survivors(self, population: List[PromptGenome], 
                        offspring: List[PromptGenome],
                        target_size: int) -> List[PromptGenome]:
        """
        Select survivors for next generation from population and offspring.
        
        Args:
            population: Current population
            offspring: New offspring
            target_size: Target population size
            
        Returns:
            List of survivors
        """
        # Combine population and offspring
        all_candidates = population + offspring
        
        # Remove duplicates based on genome hash
        unique_candidates = []
        seen_hashes = set()
        
        for genome in all_candidates:
            genome_hash = genome.get_hash()
            if genome_hash not in seen_hashes:
                unique_candidates.append(genome)
                seen_hashes.add(genome_hash)
        
        # Filter evaluated genomes
        evaluated_candidates = [g for g in unique_candidates if g.fitness is not None]
        
        if len(evaluated_candidates) <= target_size:
            return evaluated_candidates
        
        # Select survivors using elite selection
        survivors = self.select_elite(evaluated_candidates, target_size)
        
        logger.info(f"Selected {len(survivors)} survivors from {len(all_candidates)} candidates")
        
        return survivors
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics."""
        stats = self.selection_stats.copy()
        
        if stats['total_selections'] > 0:
            stats['elite_rate'] = stats['elite_selections'] / stats['total_selections']
            stats['diverse_rate'] = stats['diverse_selections'] / stats['total_selections']
            stats['random_rate'] = stats['random_selections'] / stats['total_selections']
        else:
            stats['elite_rate'] = 0.0
            stats['diverse_rate'] = 0.0
            stats['random_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset selection statistics."""
        self.selection_stats = {
            'total_selections': 0,
            'elite_selections': 0,
            'diverse_selections': 0,
            'random_selections': 0
        }
    
    def validate_selection(self, population: List[PromptGenome], 
                          selected: List[PromptGenome]) -> Dict[str, Any]:
        """
        Validate selection results.
        
        Args:
            population: Original population
            selected: Selected genomes
            
        Returns:
            Validation statistics
        """
        if not selected:
            return {'error': 'No genomes selected'}
        
        # Check if all selected genomes are from population
        population_ids = {g.genome_id for g in population}
        selected_ids = {g.genome_id for g in selected}
        
        valid_selections = len(selected_ids & population_ids)
        
        # Check fitness distribution
        selected_fitnesses = [g.fitness for g in selected if g.fitness is not None]
        population_fitnesses = [g.fitness for g in population if g.fitness is not None]
        
        stats = {
            'selected_count': len(selected),
            'valid_selections': valid_selections,
            'invalid_selections': len(selected) - valid_selections,
            'all_from_population': valid_selections == len(selected),
            'fitness_stats': {
                'selected_mean': sum(selected_fitnesses) / len(selected_fitnesses) if selected_fitnesses else 0,
                'selected_max': max(selected_fitnesses) if selected_fitnesses else 0,
                'population_mean': sum(population_fitnesses) / len(population_fitnesses) if population_fitnesses else 0,
                'population_max': max(population_fitnesses) if population_fitnesses else 0
            }
        }
        
        return stats
