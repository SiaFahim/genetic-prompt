"""
Population management for genetic algorithm prompt evolution.
"""

import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome, create_random_genome
    from src.genetics.crossover import crossover, CrossoverType
    from src.genetics.mutation import mutate, MutationType
    from src.utils.config import config
    from src.embeddings.vocabulary import vocabulary
else:
    from .genome import PromptGenome, create_random_genome
    from .crossover import crossover, CrossoverType
    from .mutation import mutate, MutationType
    from ..utils.config import config
    from ..embeddings.vocabulary import vocabulary


class Population:
    """
    Manages a population of PromptGenomes for genetic algorithm evolution.
    """
    
    def __init__(self, population_size: int = 50, max_genome_length: int = 50):
        """
        Initialize population.
        
        Args:
            population_size: Number of genomes in population
            max_genome_length: Maximum allowed genome length
        """
        self.population_size = population_size
        self.max_genome_length = max_genome_length
        self.genomes: List[PromptGenome] = []
        self.generation = 0
        self.best_genome: Optional[PromptGenome] = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.diversity_history = []
    
    def initialize_random(self, min_length: int = 5, max_length: int = 20):
        """
        Initialize population with random genomes.
        
        Args:
            min_length: Minimum genome length
            max_length: Maximum genome length
        """
        self.genomes = []
        for _ in range(self.population_size):
            genome = create_random_genome(min_length, max_length)
            genome.generation = self.generation
            self.genomes.append(genome)
        
        print(f"Initialized population with {len(self.genomes)} random genomes")
    
    def initialize_from_seeds(self, seed_prompts: List[str]):
        """
        Initialize population from seed prompts.
        
        Args:
            seed_prompts: List of seed prompt texts
        """
        self.genomes = []
        
        # Create genomes from seed prompts
        for i, prompt in enumerate(seed_prompts):
            genome = PromptGenome.from_text(prompt, genome_id=f"seed_{i}")
            genome.generation = self.generation
            self.genomes.append(genome)
        
        # Fill remaining slots with random genomes if needed
        while len(self.genomes) < self.population_size:
            genome = create_random_genome(5, 20)
            genome.generation = self.generation
            self.genomes.append(genome)
        
        # Trim if we have too many
        self.genomes = self.genomes[:self.population_size]
        
        print(f"Initialized population with {len(seed_prompts)} seeds and "
              f"{len(self.genomes) - len(seed_prompts)} random genomes")
    
    def add_genome(self, genome: PromptGenome):
        """Add a genome to the population."""
        if len(self.genomes) < self.population_size:
            self.genomes.append(genome)
        else:
            # Replace worst genome if new one is better
            worst_idx = self.get_worst_genome_index()
            if genome.fitness and self.genomes[worst_idx].fitness:
                if genome.fitness > self.genomes[worst_idx].fitness:
                    self.genomes[worst_idx] = genome
    
    def remove_genome(self, genome_id: str) -> bool:
        """
        Remove a genome by ID.
        
        Args:
            genome_id: ID of genome to remove
            
        Returns:
            True if genome was found and removed
        """
        for i, genome in enumerate(self.genomes):
            if genome.genome_id == genome_id:
                del self.genomes[i]
                return True
        return False
    
    def get_best_genome(self) -> Optional[PromptGenome]:
        """Get the genome with highest fitness."""
        if not self.genomes:
            return None
        
        best_genome = None
        best_fitness = float('-inf')
        
        for genome in self.genomes:
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
        
        return best_genome
    
    def get_worst_genome_index(self) -> int:
        """Get index of genome with lowest fitness."""
        if not self.genomes:
            return -1
        
        worst_idx = 0
        worst_fitness = float('inf')
        
        for i, genome in enumerate(self.genomes):
            fitness = genome.fitness if genome.fitness is not None else float('-inf')
            if fitness < worst_fitness:
                worst_fitness = fitness
                worst_idx = i
        
        return worst_idx
    
    def get_fitness_statistics(self) -> Dict[str, float]:
        """Get fitness statistics for the population."""
        fitnesses = [g.fitness for g in self.genomes if g.fitness is not None]
        
        if not fitnesses:
            return {'count': 0}
        
        return {
            'count': len(fitnesses),
            'mean': statistics.mean(fitnesses),
            'median': statistics.median(fitnesses),
            'std': statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            'min': min(fitnesses),
            'max': max(fitnesses),
            'range': max(fitnesses) - min(fitnesses)
        }
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity based on genome differences.
        
        Returns:
            Average diversity score (0 = identical, 1 = completely different)
        """
        if len(self.genomes) < 2:
            return 0.0
        
        diversity_scores = []
        
        for i in range(len(self.genomes)):
            for j in range(i + 1, len(self.genomes)):
                diversity = self.genomes[i].get_diversity_score(self.genomes[j])
                diversity_scores.append(diversity)
        
        return statistics.mean(diversity_scores) if diversity_scores else 0.0
    
    def tournament_selection(self, tournament_size: int = 3) -> PromptGenome:
        """
        Select a genome using tournament selection.
        
        Args:
            tournament_size: Number of genomes in tournament
            
        Returns:
            Selected genome
        """
        tournament = random.sample(self.genomes, min(tournament_size, len(self.genomes)))
        
        # Select best from tournament
        best_genome = tournament[0]
        best_fitness = best_genome.fitness if best_genome.fitness is not None else float('-inf')
        
        for genome in tournament[1:]:
            fitness = genome.fitness if genome.fitness is not None else float('-inf')
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome
        
        return best_genome
    
    def roulette_selection(self) -> PromptGenome:
        """
        Select a genome using roulette wheel selection.
        
        Returns:
            Selected genome
        """
        # Get fitnesses and handle negative values
        fitnesses = []
        min_fitness = float('inf')
        
        for genome in self.genomes:
            fitness = genome.fitness if genome.fitness is not None else 0.0
            fitnesses.append(fitness)
            min_fitness = min(min_fitness, fitness)
        
        # Shift fitnesses to be non-negative
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        # Calculate selection probabilities
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(self.genomes)
        
        # Spin the wheel
        spin = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, fitness in enumerate(fitnesses):
            cumulative += fitness
            if cumulative >= spin:
                return self.genomes[i]
        
        # Fallback
        return self.genomes[-1]
    
    def evolve_generation(self, crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                         elite_size: int = 2) -> None:
        """
        Evolve the population to the next generation.
        
        Args:
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best genomes to preserve
        """
        # Sort by fitness (descending)
        self.genomes.sort(key=lambda g: g.fitness if g.fitness is not None else float('-inf'), 
                         reverse=True)
        
        # Preserve elite
        new_population = []
        for i in range(min(elite_size, len(self.genomes))):
            elite = self.genomes[i].copy()
            elite.age += 1
            new_population.append(elite)
        
        # Generate offspring to fill population
        while len(new_population) < self.population_size:
            if random.random() < crossover_rate and len(self.genomes) >= 2:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                offspring1, offspring2 = crossover(parent1, parent2, CrossoverType.SINGLE_POINT)
                
                # Add offspring if there's space
                if len(new_population) < self.population_size:
                    new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            else:
                # Clone and mutate
                parent = self.tournament_selection()
                offspring = parent.copy()
                offspring.generation = self.generation + 1
                new_population.append(offspring)
        
        # Apply mutations
        for i in range(elite_size, len(new_population)):
            if random.random() < mutation_rate:
                new_population[i] = mutate(new_population[i], MutationType.SEMANTIC, 
                                         mutation_rate=0.1)
        
        # Update population
        self.genomes = new_population[:self.population_size]
        self.generation += 1
        
        # Update best genome
        current_best = self.get_best_genome()
        if current_best and (self.best_genome is None or 
                           current_best.fitness > self.best_fitness):
            self.best_genome = current_best.copy()
            self.best_fitness = current_best.fitness
        
        # Record statistics
        fitness_stats = self.get_fitness_statistics()
        diversity = self.calculate_diversity()
        
        self.fitness_history.append(fitness_stats)
        self.diversity_history.append(diversity)
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        fitness_stats = self.get_fitness_statistics()
        diversity = self.calculate_diversity()
        
        # Length statistics
        lengths = [g.length() for g in self.genomes]
        length_stats = {
            'mean_length': statistics.mean(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0
        }
        
        # Age statistics
        ages = [g.age for g in self.genomes]
        age_stats = {
            'mean_age': statistics.mean(ages) if ages else 0,
            'max_age': max(ages) if ages else 0
        }
        
        return {
            'generation': self.generation,
            'population_size': len(self.genomes),
            'diversity': diversity,
            'best_fitness': self.best_fitness,
            'fitness_stats': fitness_stats,
            'length_stats': length_stats,
            'age_stats': age_stats
        }
    
    def __len__(self) -> int:
        """Get population size."""
        return len(self.genomes)
    
    def __iter__(self):
        """Iterate over genomes."""
        return iter(self.genomes)


if __name__ == "__main__":
    # Test population management
    print("Testing population management...")
    
    # Load vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print("Vocabulary loaded successfully")
    else:
        print("Vocabulary not found, creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
    
    # Create population
    population = Population(population_size=10)
    
    # Test random initialization
    population.initialize_random(min_length=5, max_length=15)
    print(f"Population size: {len(population)}")
    
    # Set some random fitnesses
    for genome in population:
        genome.set_fitness(random.uniform(0.0, 1.0))
    
    # Test statistics
    stats = population.get_population_statistics()
    print(f"Population statistics: {stats}")
    
    # Test selection
    selected = population.tournament_selection()
    print(f"Tournament selected: {selected}")
    
    # Test evolution
    print("\nEvolving population...")
    for gen in range(3):
        population.evolve_generation()
        stats = population.get_population_statistics()
        print(f"Generation {stats['generation']}: "
              f"Best fitness = {stats['best_fitness']:.3f}, "
              f"Diversity = {stats['diversity']:.3f}")
    
    print("\n✅ Population management tests completed!")
