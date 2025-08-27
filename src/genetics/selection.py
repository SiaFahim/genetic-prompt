"""
Advanced selection strategies for genetic algorithm evolution.
"""

import random
import math
from typing import List, Optional, Tuple
from enum import Enum

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome
    from src.genetics.population import Population
    from src.utils.config import config
else:
    from .genome import PromptGenome
    from .population import Population
    from ..utils.config import config


class SelectionMethod(Enum):
    """Selection method types."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    ELITIST = "elitist"


class SelectionStrategy:
    """Advanced selection strategies for genetic algorithm."""
    
    def __init__(self, method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 tournament_size: int = 3,
                 selection_pressure: float = 1.5,
                 elite_ratio: float = 0.1):
        """
        Initialize selection strategy.
        
        Args:
            method: Selection method to use
            tournament_size: Size of tournament for tournament selection
            selection_pressure: Selection pressure for rank-based selection
            elite_ratio: Ratio of elite individuals to preserve
        """
        self.method = method
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
        self.elite_ratio = elite_ratio
    
    def tournament_selection(self, population: Population, 
                           tournament_size: Optional[int] = None) -> PromptGenome:
        """
        Tournament selection with configurable tournament size.
        
        Args:
            population: Population to select from
            tournament_size: Size of tournament (defaults to instance setting)
            
        Returns:
            Selected genome
        """
        size = tournament_size or self.tournament_size
        size = min(size, len(population))
        
        # Select random individuals for tournament
        tournament = random.sample(population.genomes, size)
        
        # Find best individual in tournament
        best_genome = tournament[0]
        best_fitness = best_genome.fitness if best_genome.fitness is not None else float('-inf')
        
        for genome in tournament[1:]:
            fitness = genome.fitness if genome.fitness is not None else float('-inf')
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome
        
        return best_genome
    
    def roulette_wheel_selection(self, population: Population) -> PromptGenome:
        """
        Roulette wheel selection based on fitness proportions.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected genome
        """
        # Get fitnesses and handle negative values
        fitnesses = []
        min_fitness = float('inf')
        
        for genome in population.genomes:
            fitness = genome.fitness if genome.fitness is not None else 0.0
            fitnesses.append(fitness)
            min_fitness = min(min_fitness, fitness)
        
        # Shift fitnesses to be non-negative
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 0.001 for f in fitnesses]
        
        # Calculate selection probabilities
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population.genomes)
        
        # Spin the wheel
        spin = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, fitness in enumerate(fitnesses):
            cumulative += fitness
            if cumulative >= spin:
                return population.genomes[i]
        
        # Fallback
        return population.genomes[-1]
    
    def rank_based_selection(self, population: Population) -> PromptGenome:
        """
        Rank-based selection with configurable selection pressure.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected genome
        """
        # Sort population by fitness
        sorted_genomes = sorted(population.genomes, 
                              key=lambda g: g.fitness if g.fitness is not None else float('-inf'))
        
        n = len(sorted_genomes)
        if n == 0:
            raise ValueError("Empty population")
        
        # Calculate rank-based probabilities
        probabilities = []
        for rank in range(n):
            # Linear ranking formula
            prob = (2 - self.selection_pressure + 
                   2 * (self.selection_pressure - 1) * rank / (n - 1)) / n
            probabilities.append(prob)
        
        # Select based on probabilities
        spin = random.random()
        cumulative = 0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if cumulative >= spin:
                return sorted_genomes[i]
        
        return sorted_genomes[-1]
    
    def stochastic_universal_sampling(self, population: Population, 
                                    num_selections: int) -> List[PromptGenome]:
        """
        Stochastic Universal Sampling for multiple selections.
        
        Args:
            population: Population to select from
            num_selections: Number of individuals to select
            
        Returns:
            List of selected genomes
        """
        # Get fitnesses
        fitnesses = []
        min_fitness = float('inf')
        
        for genome in population.genomes:
            fitness = genome.fitness if genome.fitness is not None else 0.0
            fitnesses.append(fitness)
            min_fitness = min(min_fitness, fitness)
        
        # Shift fitnesses to be non-negative
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 0.001 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choices(population.genomes, k=num_selections)
        
        # Calculate pointer spacing
        pointer_distance = total_fitness / num_selections
        start_point = random.uniform(0, pointer_distance)
        
        # Select individuals
        selected = []
        cumulative = 0
        genome_index = 0
        
        for i in range(num_selections):
            pointer = start_point + i * pointer_distance
            
            # Find genome at this pointer position
            while cumulative < pointer and genome_index < len(fitnesses):
                cumulative += fitnesses[genome_index]
                genome_index += 1
            
            # Select the genome (adjust for 0-based indexing)
            selected_index = max(0, genome_index - 1)
            selected.append(population.genomes[selected_index])
        
        return selected
    
    def elitist_selection(self, population: Population, 
                         num_elite: Optional[int] = None) -> List[PromptGenome]:
        """
        Elitist selection - select the best individuals.
        
        Args:
            population: Population to select from
            num_elite: Number of elite individuals (defaults to elite_ratio * population_size)
            
        Returns:
            List of elite genomes
        """
        if num_elite is None:
            num_elite = max(1, int(len(population) * self.elite_ratio))
        
        # Sort by fitness (descending)
        sorted_genomes = sorted(population.genomes,
                              key=lambda g: g.fitness if g.fitness is not None else float('-inf'),
                              reverse=True)
        
        return sorted_genomes[:num_elite]
    
    def select(self, population: Population) -> PromptGenome:
        """
        Select a single individual using the configured method.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected genome
        """
        if self.method == SelectionMethod.TOURNAMENT:
            return self.tournament_selection(population)
        elif self.method == SelectionMethod.ROULETTE_WHEEL:
            return self.roulette_wheel_selection(population)
        elif self.method == SelectionMethod.RANK_BASED:
            return self.rank_based_selection(population)
        elif self.method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            # For single selection, use SUS with 1 selection
            return self.stochastic_universal_sampling(population, 1)[0]
        elif self.method == SelectionMethod.ELITIST:
            # For single selection, return the best individual
            return self.elitist_selection(population, 1)[0]
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def select_multiple(self, population: Population, 
                       num_selections: int) -> List[PromptGenome]:
        """
        Select multiple individuals using the configured method.
        
        Args:
            population: Population to select from
            num_selections: Number of individuals to select
            
        Returns:
            List of selected genomes
        """
        if self.method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return self.stochastic_universal_sampling(population, num_selections)
        elif self.method == SelectionMethod.ELITIST:
            return self.elitist_selection(population, num_selections)
        else:
            # For other methods, select individually
            return [self.select(population) for _ in range(num_selections)]
    
    def get_selection_statistics(self, population: Population, 
                               num_trials: int = 1000) -> dict:
        """
        Get statistics about selection behavior.
        
        Args:
            population: Population to analyze
            num_trials: Number of selection trials to run
            
        Returns:
            Dictionary with selection statistics
        """
        selection_counts = {genome.genome_id: 0 for genome in population.genomes}
        
        for _ in range(num_trials):
            selected = self.select(population)
            selection_counts[selected.genome_id] += 1
        
        # Calculate statistics
        fitnesses = [g.fitness if g.fitness is not None else 0.0 for g in population.genomes]
        selection_rates = [selection_counts[g.genome_id] / num_trials for g in population.genomes]
        
        # Correlation between fitness and selection rate
        if len(fitnesses) > 1:
            import statistics
            fitness_mean = statistics.mean(fitnesses)
            selection_mean = statistics.mean(selection_rates)
            
            numerator = sum((f - fitness_mean) * (s - selection_mean) 
                          for f, s in zip(fitnesses, selection_rates))
            fitness_var = sum((f - fitness_mean) ** 2 for f in fitnesses)
            selection_var = sum((s - selection_mean) ** 2 for s in selection_rates)
            
            correlation = (numerator / math.sqrt(fitness_var * selection_var) 
                         if fitness_var > 0 and selection_var > 0 else 0)
        else:
            correlation = 0
        
        return {
            'method': self.method.value,
            'num_trials': num_trials,
            'selection_counts': selection_counts,
            'selection_rates': selection_rates,
            'fitness_selection_correlation': correlation,
            'most_selected': max(selection_counts, key=selection_counts.get),
            'least_selected': min(selection_counts, key=selection_counts.get)
        }


if __name__ == "__main__":
    # Test selection strategies
    print("Testing selection strategies...")
    
    # Load vocabulary for testing
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create test population with varied fitness
    population = Population(population_size=10)
    population.initialize_random(5, 15)
    
    # Assign varied fitness values
    fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for genome, fitness in zip(population.genomes, fitness_values):
        genome.set_fitness(fitness)
    
    print(f"✅ Created test population with fitness range: {min(fitness_values):.1f} - {max(fitness_values):.1f}")
    
    # Test all selection methods
    methods = [
        SelectionMethod.TOURNAMENT,
        SelectionMethod.ROULETTE_WHEEL,
        SelectionMethod.RANK_BASED,
        SelectionMethod.STOCHASTIC_UNIVERSAL,
        SelectionMethod.ELITIST
    ]
    
    for method in methods:
        print(f"\n--- {method.value.upper()} SELECTION ---")
        
        strategy = SelectionStrategy(method=method, tournament_size=3, selection_pressure=1.5)
        
        # Test single selection
        selected = strategy.select(population)
        print(f"✅ Single selection: fitness={selected.fitness:.2f}")
        
        # Test multiple selection
        if method in [SelectionMethod.STOCHASTIC_UNIVERSAL, SelectionMethod.ELITIST]:
            multiple_selected = strategy.select_multiple(population, 3)
            fitnesses = [g.fitness for g in multiple_selected]
            print(f"✅ Multiple selection: fitnesses={[f'{f:.2f}' for f in fitnesses]}")
        
        # Test selection statistics (smaller sample for speed)
        stats = strategy.get_selection_statistics(population, num_trials=100)
        print(f"✅ Selection correlation: {stats['fitness_selection_correlation']:.3f}")
        print(f"   Most selected: {stats['most_selected'][:8]} "
              f"(fitness={next(g.fitness for g in population.genomes if g.genome_id == stats['most_selected']):.2f})")
    
    print("\n🎯 Selection strategies tests completed successfully!")
