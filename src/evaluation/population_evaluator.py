"""
Population evaluator for genetic algorithm.
Coordinates evaluation of entire populations with caching and progress tracking.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging

from src.genetics.genome import PromptGenome
from src.evaluation.async_evaluator import AsyncEvaluator
from src.evaluation.fitness import FitnessCalculator
from src.utils.dataset import GSM8KDataset
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class PopulationEvaluator:
    """Coordinates evaluation of genetic algorithm populations."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize population evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.async_evaluator = AsyncEvaluator(config_path)
        self.fitness_calculator = FitnessCalculator(config_path)
        self.dataset = GSM8KDataset(config_path)
        
        # Configuration
        self.results_dir = Path(self.config.get('paths.results_dir', './data/results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Caching
        self.evaluation_cache = {}
        self.cache_file = self.results_dir / 'evaluation_cache.json'
        self._load_evaluation_cache()
        
        # Statistics
        self.evaluation_history = []
    
    def _load_evaluation_cache(self) -> None:
        """Load evaluation cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.evaluation_cache = json.load(f)
                logger.info(f"Loaded {len(self.evaluation_cache)} cached evaluations")
            except Exception as e:
                logger.warning(f"Failed to load evaluation cache: {e}")
                self.evaluation_cache = {}
    
    def _save_evaluation_cache(self) -> None:
        """Save evaluation cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f, indent=2)
            logger.debug(f"Saved {len(self.evaluation_cache)} cached evaluations")
        except Exception as e:
            logger.error(f"Failed to save evaluation cache: {e}")
    
    def _get_cache_key(self, genome_hash: str, problems_hash: str) -> str:
        """Generate cache key for genome-problems pair."""
        return f"{genome_hash}_{problems_hash}"
    
    def _get_problems_hash(self, problems: List[Dict[str, Any]]) -> str:
        """Generate hash for a list of problems."""
        import hashlib
        problems_str = json.dumps([p['id'] for p in problems], sort_keys=True)
        return hashlib.md5(problems_str.encode()).hexdigest()[:16]
    
    async def evaluate_population(self, population: List[PromptGenome],
                                generation: int,
                                use_progressive: bool = True,
                                show_progress: bool = True,
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate entire population and update fitness scores.
        
        Args:
            population: List of genomes to evaluate
            generation: Current generation number
            use_progressive: Whether to use progressive evaluation
            show_progress: Whether to show progress bar
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with evaluation results and statistics
        """
        start_time = time.time()
        
        logger.info(f"Evaluating population of {len(population)} genomes for generation {generation}")
        
        # Get problems for this generation
        if use_progressive:
            problems = self.dataset.get_problems_for_generation(generation)
        else:
            problems = self.dataset.load_evaluation_set('primary')
        
        problems_hash = self._get_problems_hash(problems)
        
        # Check cache and separate cached vs uncached genomes
        cached_results = {}
        uncached_genomes = []
        
        for genome in population:
            genome_hash = genome.get_hash()
            cache_key = self._get_cache_key(genome_hash, problems_hash)
            
            if cache_key in self.evaluation_cache:
                cached_results[genome.genome_id] = self.evaluation_cache[cache_key]
            else:
                uncached_genomes.append(genome)
        
        logger.info(f"Cache status: {len(cached_results)} cached, {len(uncached_genomes)} need evaluation")
        
        # Evaluate uncached genomes
        new_results = {}
        if uncached_genomes:
            evaluation_results = await self.async_evaluator.evaluate_population(
                uncached_genomes, problems, show_progress, progress_callback
            )
            
            # Cache new results
            for genome, result in zip(uncached_genomes, evaluation_results):
                genome_hash = genome.get_hash()
                cache_key = self._get_cache_key(genome_hash, problems_hash)
                
                # Store in cache (without genome-specific data)
                cache_result = {
                    'accuracy': result['accuracy'],
                    'correct_count': result['correct_count'],
                    'total_count': result['total_count'],
                    'evaluation_time_seconds': result['evaluation_time_seconds'],
                    'total_tokens_used': result['total_tokens_used'],
                    'total_cost_usd': result['total_cost_usd']
                }
                
                self.evaluation_cache[cache_key] = cache_result
                new_results[genome.genome_id] = result
        
        # Combine cached and new results
        all_results = {**cached_results, **new_results}
        
        # Calculate fitness for all genomes
        accuracies = []
        for genome in population:
            result = all_results[genome.genome_id]
            accuracies.append(result['accuracy'])
        
        # Calculate comprehensive fitness
        fitness_results = self.fitness_calculator.calculate_population_fitness(
            population, accuracies, use_diversity=False
        )
        
        # Update genome fitness
        for genome, fitness_components in zip(population, fitness_results):
            self.fitness_calculator.update_genome_fitness(genome, fitness_components)
        
        # Calculate population statistics
        evaluation_time = time.time() - start_time
        
        population_stats = self.fitness_calculator.get_fitness_statistics(population)
        
        # Prepare evaluation summary
        evaluation_summary = {
            'generation': generation,
            'population_size': len(population),
            'problems_count': len(problems),
            'cached_count': len(cached_results),
            'evaluated_count': len(uncached_genomes),
            'evaluation_time_seconds': evaluation_time,
            'population_stats': population_stats,
            'best_genome': self._get_best_genome_info(population),
            'timestamp': time.time()
        }
        
        # Add to history
        self.evaluation_history.append(evaluation_summary)
        
        # Save cache periodically
        if len(new_results) > 0:
            self._save_evaluation_cache()
        
        logger.info(f"Population evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Best fitness: {population_stats.get('fitness', {}).get('max', 0):.4f}")
        
        return evaluation_summary
    
    def _get_best_genome_info(self, population: List[PromptGenome]) -> Dict[str, Any]:
        """Get information about the best genome in population."""
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            return {}
        
        best_genome = max(evaluated_genomes, key=lambda g: g.fitness)
        
        return {
            'genome_id': best_genome.genome_id,
            'fitness': best_genome.fitness,
            'accuracy': best_genome.accuracy,
            'length': len(best_genome.tokens),
            'text': best_genome.to_text(),
            'generation_born': best_genome.generation_born
        }
    
    async def evaluate_single_genome(self, genome: PromptGenome,
                                   evaluation_set: str = 'primary') -> Dict[str, Any]:
        """
        Evaluate a single genome on specified evaluation set.
        
        Args:
            genome: Genome to evaluate
            evaluation_set: Name of evaluation set to use
            
        Returns:
            Evaluation result dictionary
        """
        problems = self.dataset.load_evaluation_set(evaluation_set)
        
        result = await self.async_evaluator.evaluate_genome_on_problems(genome, problems)
        
        # Calculate fitness
        fitness_components = self.fitness_calculator.calculate_comprehensive_fitness(
            result['accuracy'], genome
        )
        
        # Update genome
        self.fitness_calculator.update_genome_fitness(genome, fitness_components)
        
        return result
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history.copy()
    
    def get_best_genomes_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of best genomes across generations.
        
        Args:
            count: Number of top genomes to return per generation
            
        Returns:
            List of best genome information
        """
        best_history = []
        
        for eval_summary in self.evaluation_history:
            if 'best_genome' in eval_summary and eval_summary['best_genome']:
                best_info = eval_summary['best_genome'].copy()
                best_info['generation_evaluated'] = eval_summary['generation']
                best_info['evaluation_timestamp'] = eval_summary['timestamp']
                best_history.append(best_info)
        
        # Sort by fitness (descending) and return top count
        best_history.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        return best_history[:count]
    
    def save_evaluation_results(self, filename: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            filename: Optional filename, auto-generated if None
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        results_data = {
            'evaluation_history': self.evaluation_history,
            'best_genomes_history': self.get_best_genomes_history(),
            'evaluation_statistics': self.get_evaluation_statistics(),
            'config_snapshot': self.config.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
        return str(filepath)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        stats = {
            'total_generations_evaluated': len(self.evaluation_history),
            'total_cache_entries': len(self.evaluation_cache),
            'async_evaluator_stats': self.async_evaluator.get_evaluation_statistics()
        }
        
        if self.evaluation_history:
            # Calculate trends
            recent_evals = self.evaluation_history[-5:]  # Last 5 generations
            
            best_fitnesses = [e['population_stats']['fitness']['max'] 
                            for e in recent_evals 
                            if 'fitness' in e.get('population_stats', {})]
            
            if best_fitnesses:
                stats['recent_best_fitness'] = {
                    'current': best_fitnesses[-1],
                    'trend': best_fitnesses[-1] - best_fitnesses[0] if len(best_fitnesses) > 1 else 0,
                    'max_recent': max(best_fitnesses)
                }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Evaluation cache cleared")
