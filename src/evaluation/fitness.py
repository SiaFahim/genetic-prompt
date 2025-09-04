"""
Fitness calculation system for genetic algorithm.
Implements comprehensive fitness functions with multiple components.
"""

import math
from typing import List, Dict, Any, Optional, Callable
import logging

from src.genetics.genome import PromptGenome
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class FitnessCalculator:
    """Calculates fitness scores for genomes."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize fitness calculator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Configuration
        self.length_penalty_threshold = self.config.get('genome.length_penalty_threshold', 200)
        self.max_length_penalty = self.config.get('genome.max_length_penalty', 0.1)
        
        # Fitness components weights (can be configured)
        self.weights = {
            'accuracy': 1.0,
            'length_penalty': 1.0,
            'diversity_bonus': 0.0,  # Can be enabled for diversity-aware fitness
            'novelty_bonus': 0.0     # Can be enabled for novelty search
        }
    
    def calculate_length_penalty(self, genome: PromptGenome) -> float:
        """
        Calculate length penalty for a genome.
        
        Args:
            genome: Genome to calculate penalty for
            
        Returns:
            Penalty factor (1.0 = no penalty, <1.0 = penalty applied)
        """
        length = len(genome.tokens)
        
        if length <= self.length_penalty_threshold:
            return 1.0
        
        # Linear penalty beyond threshold
        excess_tokens = length - self.length_penalty_threshold
        penalty_factor = min(self.max_length_penalty, (excess_tokens / 300) * self.max_length_penalty)
        
        return 1.0 - penalty_factor
    
    def calculate_basic_fitness(self, accuracy: float, genome: PromptGenome) -> float:
        """
        Calculate basic fitness combining accuracy and length penalty.
        
        Args:
            accuracy: Accuracy score (0-1)
            genome: Genome being evaluated
            
        Returns:
            Fitness score
        """
        length_penalty = self.calculate_length_penalty(genome)
        fitness = accuracy * length_penalty
        
        return fitness
    
    def calculate_diversity_bonus(self, genome: PromptGenome, 
                                population: List[PromptGenome]) -> float:
        """
        Calculate diversity bonus based on uniqueness in population.
        
        Args:
            genome: Target genome
            population: Current population
            
        Returns:
            Diversity bonus (0-1)
        """
        if len(population) <= 1:
            return 0.0
        
        # Calculate average Jaccard distance to other genomes
        total_distance = 0.0
        comparisons = 0
        
        genome_tokens = set(genome.tokens)
        
        for other in population:
            if other.genome_id != genome.genome_id:
                other_tokens = set(other.tokens)
                
                intersection = len(genome_tokens & other_tokens)
                union = len(genome_tokens | other_tokens)
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    jaccard_distance = 1 - jaccard_similarity
                    total_distance += jaccard_distance
                    comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        avg_distance = total_distance / comparisons
        return min(1.0, avg_distance)  # Cap at 1.0
    
    def calculate_novelty_bonus(self, genome: PromptGenome,
                              archive: List[PromptGenome]) -> float:
        """
        Calculate novelty bonus based on distance to archived genomes.
        
        Args:
            genome: Target genome
            archive: Archive of previous genomes
            
        Returns:
            Novelty bonus (0-1)
        """
        if not archive:
            return 1.0  # Maximum novelty if no archive
        
        # Find k-nearest neighbors in archive
        k = min(5, len(archive))
        distances = []
        
        genome_tokens = set(genome.tokens)
        
        for archived in archive:
            archived_tokens = set(archived.tokens)
            
            intersection = len(genome_tokens & archived_tokens)
            union = len(genome_tokens | archived_tokens)
            
            if union > 0:
                jaccard_distance = 1 - (intersection / union)
                distances.append(jaccard_distance)
        
        if not distances:
            return 1.0
        
        # Average distance to k-nearest neighbors
        distances.sort()
        k_nearest_distances = distances[:k]
        avg_distance = sum(k_nearest_distances) / len(k_nearest_distances)
        
        return min(1.0, avg_distance)
    
    def calculate_comprehensive_fitness(self, accuracy: float, genome: PromptGenome,
                                      population: Optional[List[PromptGenome]] = None,
                                      archive: Optional[List[PromptGenome]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive fitness with all components.
        
        Args:
            accuracy: Accuracy score
            genome: Target genome
            population: Current population (for diversity)
            archive: Archive of genomes (for novelty)
            
        Returns:
            Dictionary with fitness components and total
        """
        components = {}
        
        # Basic accuracy
        components['accuracy'] = accuracy
        
        # Length penalty
        length_penalty = self.calculate_length_penalty(genome)
        components['length_penalty'] = length_penalty
        
        # Diversity bonus
        if population and self.weights['diversity_bonus'] > 0:
            diversity_bonus = self.calculate_diversity_bonus(genome, population)
            components['diversity_bonus'] = diversity_bonus
        else:
            components['diversity_bonus'] = 0.0
        
        # Novelty bonus
        if archive and self.weights['novelty_bonus'] > 0:
            novelty_bonus = self.calculate_novelty_bonus(genome, archive)
            components['novelty_bonus'] = novelty_bonus
        else:
            components['novelty_bonus'] = 0.0
        
        # Calculate weighted total
        total_fitness = (
            components['accuracy'] * self.weights['accuracy'] * 
            components['length_penalty'] * self.weights['length_penalty'] +
            components['diversity_bonus'] * self.weights['diversity_bonus'] +
            components['novelty_bonus'] * self.weights['novelty_bonus']
        )
        
        components['total_fitness'] = total_fitness
        
        return components
    
    def calculate_population_fitness(self, population: List[PromptGenome],
                                   accuracies: List[float],
                                   use_diversity: bool = False,
                                   archive: Optional[List[PromptGenome]] = None) -> List[Dict[str, float]]:
        """
        Calculate fitness for entire population.
        
        Args:
            population: List of genomes
            accuracies: List of accuracy scores
            use_diversity: Whether to include diversity bonus
            archive: Optional archive for novelty calculation
            
        Returns:
            List of fitness component dictionaries
        """
        if len(population) != len(accuracies):
            raise ValueError("Population and accuracies must have same length")
        
        fitness_results = []
        
        for i, (genome, accuracy) in enumerate(zip(population, accuracies)):
            # Determine what components to include
            pop_for_diversity = population if use_diversity else None
            
            fitness_components = self.calculate_comprehensive_fitness(
                accuracy, genome, pop_for_diversity, archive
            )
            
            fitness_results.append(fitness_components)
        
        return fitness_results
    
    def update_genome_fitness(self, genome: PromptGenome, 
                            fitness_components: Dict[str, float]) -> None:
        """
        Update genome with calculated fitness.
        
        Args:
            genome: Genome to update
            fitness_components: Fitness components dictionary
        """
        genome.fitness = fitness_components['total_fitness']
        genome.accuracy = fitness_components['accuracy']
        
        # Store detailed fitness information in metadata
        genome.metadata['fitness_components'] = fitness_components
        genome.metadata['length_penalty'] = fitness_components['length_penalty']
    
    def rank_population_by_fitness(self, population: List[PromptGenome]) -> List[PromptGenome]:
        """
        Rank population by fitness (descending order).
        
        Args:
            population: Population to rank
            
        Returns:
            Sorted population (best fitness first)
        """
        # Filter out genomes without fitness
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        # Sort by fitness (descending)
        ranked = sorted(evaluated_genomes, key=lambda g: g.fitness, reverse=True)
        
        return ranked
    
    def get_fitness_statistics(self, population: List[PromptGenome]) -> Dict[str, Any]:
        """
        Get fitness statistics for population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Dictionary with fitness statistics
        """
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            return {'error': 'No evaluated genomes'}
        
        fitnesses = [g.fitness for g in evaluated_genomes]
        accuracies = [g.accuracy for g in evaluated_genomes if g.accuracy is not None]
        lengths = [len(g.tokens) for g in evaluated_genomes]
        
        stats = {
            'population_size': len(population),
            'evaluated_count': len(evaluated_genomes),
            'fitness': {
                'mean': sum(fitnesses) / len(fitnesses),
                'max': max(fitnesses),
                'min': min(fitnesses),
                'std': math.sqrt(sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))
            },
            'length': {
                'mean': sum(lengths) / len(lengths),
                'max': max(lengths),
                'min': min(lengths)
            }
        }
        
        if accuracies:
            stats['accuracy'] = {
                'mean': sum(accuracies) / len(accuracies),
                'max': max(accuracies),
                'min': min(accuracies)
            }
        
        return stats
    
    def set_weights(self, **weights) -> None:
        """
        Update fitness component weights.
        
        Args:
            **weights: Keyword arguments for weight values
        """
        for component, weight in weights.items():
            if component in self.weights:
                self.weights[component] = weight
                logger.info(f"Updated {component} weight to {weight}")
            else:
                logger.warning(f"Unknown fitness component: {component}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current fitness component weights."""
        return self.weights.copy()
