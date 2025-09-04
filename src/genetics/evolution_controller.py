"""
Main evolution controller for genetic algorithm.
Coordinates the complete evolutionary process.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
import logging

from src.genetics.genome import PromptGenome
from src.genetics.population import PopulationInitializer
from src.genetics.selection import SelectionOperator
from src.genetics.crossover import CrossoverOperator
from src.genetics.mutation import MutationOperator
from src.genetics.convergence import ConvergenceDetector
from src.evaluation.population_evaluator import PopulationEvaluator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class EvolutionController:
    """Main controller for genetic algorithm evolution."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize evolution controller.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Initialize components
        self.population_initializer = PopulationInitializer(config_path)
        self.selection_operator = SelectionOperator(config_path)
        self.crossover_operator = CrossoverOperator(config_path)
        self.mutation_operator = MutationOperator(config_path)
        self.convergence_detector = ConvergenceDetector(config_path)
        self.population_evaluator = PopulationEvaluator(config_path)
        
        # Configuration
        ga_config = self.config.get('genetic_algorithm', {})
        self.population_size = ga_config.get('population_size', 500)
        self.num_parents = ga_config.get('num_parents', 50)
        self.max_generations = ga_config.get('max_generations', 30)
        
        # State
        self.current_generation = 0
        self.current_population = []
        self.best_genome_history = []
        self.evolution_stats = {
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0,
            'total_mutations': 0,
            'total_crossovers': 0
        }
    
    async def initialize_population(self, seed_prompts: Optional[List[str]] = None) -> List[PromptGenome]:
        """
        Initialize the starting population.
        
        Args:
            seed_prompts: Optional seed prompts
            
        Returns:
            Initial population
        """
        logger.info(f"Initializing population of {self.population_size} genomes")
        
        population = self.population_initializer.initialize_population(
            seed_prompts=seed_prompts,
            target_size=self.population_size
        )
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        await self.population_evaluator.evaluate_population(
            population, 
            generation=0,
            use_progressive=True,
            show_progress=True
        )
        
        self.current_population = population
        self.current_generation = 0
        
        # Update convergence detector
        self.convergence_detector.update(population, 0)
        
        # Track best genome
        best_genome = self._get_best_genome(population)
        if best_genome:
            self.best_genome_history.append({
                'generation': 0,
                'genome': best_genome.copy(),
                'fitness': best_genome.fitness,
                'accuracy': best_genome.accuracy
            })
        
        logger.info(f"Initial population ready. Best fitness: {best_genome.fitness:.4f}")
        
        return population
    
    async def evolve_generation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evolve one generation.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generation statistics
        """
        generation_start_time = time.time()
        self.current_generation += 1
        
        logger.info(f"Starting generation {self.current_generation}")
        
        # Selection
        logger.info("Selecting parents...")
        parents = self.selection_operator.select_parents(self.current_population)

        if len(parents) < 2:
            logger.error("Insufficient parents selected")
            raise ValueError("Need at least 2 parents for reproduction")

        # Log selection statistics
        selection_stats = self.selection_operator.get_selection_statistics()
        logger.info(f"Selected {len(parents)} parents: "
                   f"{selection_stats.get('elite_selections', 0)} elite, "
                   f"{selection_stats.get('diverse_selections', 0)} diverse, "
                   f"{selection_stats.get('random_selections', 0)} random")
        
        # Generate offspring through crossover and mutation
        logger.info("Generating offspring...")
        offspring = []
        
        # Calculate number of offspring needed
        offspring_needed = max(0, self.population_size - len(parents))

        if offspring_needed > 0 and len(parents) >= 2:
            while len(offspring) < offspring_needed:
                # Select two parents randomly
                import random
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                # Ensure different parents if possible
                attempts = 0
                while parent1.genome_id == parent2.genome_id and len(parents) > 1 and attempts < 10:
                    parent2 = random.choice(parents)
                    attempts += 1

                # Crossover
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)

                # Mutation
                is_stagnant = self.convergence_detector.is_stagnant
                child1 = self.mutation_operator.mutate_genome(child1, is_stagnant)
                child2 = self.mutation_operator.mutate_genome(child2, is_stagnant)

                offspring.extend([child1, child2])

                # Update statistics
                self.evolution_stats['total_crossovers'] += 1
        
        # Trim offspring to exact size needed
        offspring = offspring[:offspring_needed]

        # Log genetic operations statistics
        mutation_stats = self.mutation_operator.get_mutation_statistics()
        logger.info(f"Generated {len(offspring)} offspring through {self.evolution_stats['total_crossovers']} crossovers")
        logger.info(f"Applied {mutation_stats.get('total_mutations', 0)} mutations "
                   f"({mutation_stats.get('semantic_rate', 0):.1%} semantic, "
                   f"{mutation_stats.get('random_rate', 0):.1%} random)")
        
        # Evaluate offspring
        logger.info("Evaluating offspring...")
        await self.population_evaluator.evaluate_population(
            offspring,
            generation=self.current_generation,
            use_progressive=True,
            show_progress=True,
            progress_callback=progress_callback
        )
        
        # Survivor selection
        logger.info("Selecting survivors...")
        survivors = self.selection_operator.select_survivors(
            self.current_population,
            offspring,
            self.population_size
        )
        
        self.current_population = survivors
        
        # Update convergence detector
        self.convergence_detector.update(survivors, self.current_generation)
        
        # Track best genome and calculate population diversity
        best_genome = self._get_best_genome(survivors)
        if best_genome:
            self.best_genome_history.append({
                'generation': self.current_generation,
                'genome': best_genome.copy(),
                'fitness': best_genome.fitness,
                'accuracy': best_genome.accuracy
            })

        # Calculate and log population diversity
        evaluated_survivors = [g for g in survivors if g.fitness is not None]
        if len(evaluated_survivors) > 1:
            fitnesses = [g.fitness for g in evaluated_survivors]
            mean_fitness = sum(fitnesses) / len(fitnesses)
            fitness_std = (sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            fitness_range = max(fitnesses) - min(fitnesses)

            logger.info(f"Population diversity: fitness_std={fitness_std:.4f}, "
                       f"fitness_range={fitness_range:.4f}, mean_fitness={mean_fitness:.4f}")
        
        # Calculate generation statistics
        generation_time = time.time() - generation_start_time
        
        generation_stats = {
            'generation': self.current_generation,
            'population_size': len(survivors),
            'parents_selected': len(parents),
            'offspring_generated': len(offspring),
            'best_fitness': best_genome.fitness if best_genome else 0,
            'best_accuracy': best_genome.accuracy if best_genome else 0,
            'generation_time_seconds': generation_time,
            'convergence_status': self.convergence_detector.get_convergence_status(),
            'is_stagnant': self.convergence_detector.is_stagnant
        }
        
        logger.info(f"Generation {self.current_generation} complete. "
                   f"Best fitness: {generation_stats['best_fitness']:.4f}, "
                   f"Best accuracy: {generation_stats['best_accuracy']:.4f}")
        
        return generation_stats
    
    async def run_evolution(self, seed_prompts: Optional[List[str]] = None,
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete evolution process.
        
        Args:
            seed_prompts: Optional seed prompts
            progress_callback: Optional callback for progress updates
            
        Returns:
            Evolution results
        """
        logger.info("Starting genetic algorithm evolution")
        logger.info(f"Configuration: {self.population_size} population, "
                   f"{self.max_generations} max generations")
        
        self.evolution_stats['start_time'] = time.time()
        
        # Initialize population
        await self.initialize_population(seed_prompts)
        
        generation_results = []
        
        # Evolution loop
        while not self.convergence_detector.should_terminate(self.max_generations):
            try:
                generation_stats = await self.evolve_generation(progress_callback)
                generation_results.append(generation_stats)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(generation_stats)
                
            except Exception as e:
                logger.error(f"Error in generation {self.current_generation}: {e}")
                break
        
        self.evolution_stats['end_time'] = time.time()
        
        # Final results
        termination_reason = self.convergence_detector.get_termination_reason(self.max_generations)
        best_overall = self._get_best_overall_genome()
        
        evolution_results = {
            'termination_reason': termination_reason,
            'total_generations': self.current_generation,
            'best_genome': best_overall,
            'generation_results': generation_results,
            'evolution_statistics': self._get_evolution_statistics(),
            'convergence_statistics': self.convergence_detector.get_statistics(),
            'final_population_size': len(self.current_population)
        }
        
        logger.info("Evolution completed!")
        logger.info(f"Reason: {termination_reason}")
        logger.info(f"Best genome fitness: {best_overall['fitness']:.4f}")
        logger.info(f"Best genome accuracy: {best_overall['accuracy']:.4f}")
        
        return evolution_results
    
    def _get_best_genome(self, population: List[PromptGenome]) -> Optional[PromptGenome]:
        """Get best genome from population."""
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            return None
        
        return max(evaluated_genomes, key=lambda g: g.fitness)
    
    def _get_best_overall_genome(self) -> Dict[str, Any]:
        """Get best genome from entire evolution history."""
        if not self.best_genome_history:
            return {}
        
        best_entry = max(self.best_genome_history, key=lambda x: x['fitness'])
        
        return {
            'generation': best_entry['generation'],
            'fitness': best_entry['fitness'],
            'accuracy': best_entry['accuracy'],
            'text': best_entry['genome'].to_text(),
            'genome_id': best_entry['genome'].genome_id,
            'length': len(best_entry['genome'].tokens)
        }
    
    def _get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        total_time = (self.evolution_stats['end_time'] - 
                     self.evolution_stats['start_time']) if self.evolution_stats['end_time'] else 0
        
        stats = {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'generations_completed': self.current_generation,
            'avg_time_per_generation': total_time / self.current_generation if self.current_generation > 0 else 0,
            'selection_stats': self.selection_operator.get_selection_statistics(),
            'mutation_stats': self.mutation_operator.get_mutation_statistics(),
            'evaluation_stats': self.population_evaluator.get_evaluation_statistics(),
            'best_genome_progression': [
                {
                    'generation': entry['generation'],
                    'fitness': entry['fitness'],
                    'accuracy': entry['accuracy']
                }
                for entry in self.best_genome_history
            ]
        }
        
        return stats
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        best_genome = self._get_best_genome(self.current_population)
        
        return {
            'current_generation': self.current_generation,
            'max_generations': self.max_generations,
            'population_size': len(self.current_population),
            'best_current_fitness': best_genome.fitness if best_genome else 0,
            'best_current_accuracy': best_genome.accuracy if best_genome else 0,
            'convergence_status': self.convergence_detector.get_convergence_status(),
            'is_terminated': self.convergence_detector.should_terminate(self.max_generations)
        }
