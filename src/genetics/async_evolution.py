"""
Asynchronous Evolution Controller with batch processing capabilities.

This module provides an enhanced evolution controller that integrates with the
async evaluation pipeline for improved performance through batch processing
and concurrent API calls.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .evolution import EvolutionController, EvolutionConfig, GenerationResult
from .population import Population
from ..evaluation.async_pipeline import AsyncEvaluationPipeline, PopulationBatchConfig
from ..evaluation.async_llm_interface import AsyncLLMInterface, BatchConfig
from ..evaluation.fitness import FitnessCalculator
from ..config.hyperparameters import get_hyperparameter_config


@dataclass
class AsyncEvolutionConfig(EvolutionConfig):
    """Extended evolution configuration with async parameters."""
    # Async evaluation settings
    enable_async_evaluation: bool = True
    async_batch_size: int = 20
    max_concurrent_requests: int = 10
    genome_batch_size: int = 10
    max_concurrent_genomes: int = 5
    rate_limit_per_minute: int = 3500
    
    # Performance monitoring
    detailed_performance_logging: bool = True
    benchmark_sync_vs_async: bool = False


class AsyncEvolutionController(EvolutionController):
    """Enhanced evolution controller with async evaluation capabilities."""
    
    def __init__(self, 
                 config: AsyncEvolutionConfig,
                 seed_prompts: Optional[List[str]] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize async evolution controller.
        
        Args:
            config: Async evolution configuration
            seed_prompts: Optional seed prompts for initialization
            progress_callback: Optional callback for progress updates
        """
        # Initialize hyperparameter config
        hyperparams = get_hyperparameter_config()

        # Import config for model settings
        from ..utils.config import config as system_config

        # Create async LLM interface
        batch_config = BatchConfig(
            batch_size=config.async_batch_size,
            max_concurrent_requests=config.max_concurrent_requests,
            rate_limit_per_minute=config.rate_limit_per_minute,
            retry_attempts=hyperparams.async_retry_attempts,
            base_delay=hyperparams.async_base_delay,
            max_delay=hyperparams.async_max_delay,
            timeout=hyperparams.async_timeout
        )

        self.async_llm_interface = AsyncLLMInterface(
            model=system_config.default_model,
            temperature=hyperparams.temperature,
            max_tokens=hyperparams.max_tokens,
            batch_config=batch_config
        )
        
        # Create population batch config
        population_batch_config = PopulationBatchConfig(
            genome_batch_size=config.genome_batch_size,
            problem_batch_size=config.async_batch_size,
            max_concurrent_genomes=config.max_concurrent_genomes,
            enable_progress_bar=True,
            detailed_logging=config.detailed_performance_logging
        )
        
        # Create async evaluation pipeline
        self.async_evaluation_pipeline = AsyncEvaluationPipeline(
            async_llm_interface=self.async_llm_interface,
            fitness_calculator=FitnessCalculator(),
            use_cache=hyperparams.use_cache,
            population_batch_config=population_batch_config,
            max_problems=hyperparams.max_problems
        )
        
        # Store config
        self.async_config = config
        
        # Initialize base class with sync pipeline (fallback)
        super().__init__(
            config=config,
            evaluation_pipeline=None,  # We'll override the evaluation method
            seed_prompts=seed_prompts,
            progress_callback=progress_callback
        )
        
        # Initialize population (CRITICAL FIX)
        self.initialize_population()
        print(f"✅ Population initialized with {len(self.population)} genomes")

        # Performance tracking
        self.async_evaluation_times = []
        self.sync_evaluation_times = []
        self.performance_comparisons = []
    
    async def evolve_generation_async(self) -> GenerationResult:
        """Evolve the population for one generation using async evaluation."""
        generation_start = time.time()
        
        # Evaluate population asynchronously
        eval_start = time.time()
        
        if self.async_config.enable_async_evaluation:
            population_result = await self.async_evaluation_pipeline.evaluate_adaptive_async(
                self.population, self.population.generation
            )
            evaluation_results = population_result.evaluation_results
            eval_time = population_result.total_time
            
            # Track performance metrics
            self.async_evaluation_times.append(eval_time)
            
            if self.async_config.detailed_performance_logging:
                print(f"  ⚡ {eval_time:.1f}s | {population_result.total_api_calls} API | "
                      f"{population_result.total_cache_hits} cache | "
                      f"{population_result.throughput_problems_per_second:.1f} prob/s")
        else:
            # Fallback to sync evaluation
            evaluation_results = self.evaluation_pipeline.evaluate_adaptive(
                self.population, self.population.generation
            )
            eval_time = time.time() - eval_start
            self.sync_evaluation_times.append(eval_time)
        
        self.total_evaluations += len(evaluation_results)
        
        # Update best genome
        current_best = self.population.get_best_genome()
        if current_best and current_best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = current_best.fitness
            self.best_genome_ever = current_best.copy()
        
        # Check convergence
        convergence_status = self.convergence_detector.update(self.population)
        
        # Get population statistics
        pop_stats = self.population.get_population_statistics()
        
        # Evolve to next generation (if not converged)
        if not convergence_status.converged:
            self._evolve_population()
        
        evolution_time = time.time() - generation_start
        
        # Create generation result
        result = GenerationResult(
            generation=self.population.generation,
            best_fitness=pop_stats['best_fitness'],
            mean_fitness=pop_stats['fitness_stats'].get('mean', 0.0),
            diversity=pop_stats['diversity'],
            convergence_status=convergence_status,
            evaluation_time=eval_time,
            evolution_time=evolution_time,
            population_stats=pop_stats
        )
        
        self.generation_results.append(result)
        
        # Progress callback
        if self.progress_callback:
            self.progress_callback(result)
        
        return result
    
    def evolve_generation(self) -> GenerationResult:
        """Override to use async evaluation in sync context."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.evolve_generation_async()
                    )
                    return future.result()
            else:
                # Run in existing loop
                return loop.run_until_complete(self.evolve_generation_async())
        except RuntimeError:
            # No event loop exists, create new one
            return asyncio.run(self.evolve_generation_async())
    
    async def run_evolution_async(self, max_generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete evolution process asynchronously.
        
        Args:
            max_generations: Override max generations from config
            
        Returns:
            Evolution results dictionary
        """
        max_gens = max_generations or self.config.max_generations

        # Validate population before starting evolution
        if len(self.population) == 0:
            error_msg = (
                "❌ CRITICAL ERROR: Population is empty (0 genomes)!\n"
                "   This indicates a population initialization failure.\n"
                "   Troubleshooting:\n"
                "   1. Check that initialize_population() was called in __init__()\n"
                "   2. Verify seed_prompts are provided or random initialization works\n"
                "   3. Check for errors during population creation"
            )
            print(error_msg)
            raise ValueError("Cannot start evolution with empty population")

        print(f"🧬 Starting async evolution with {len(self.population)} genomes for {max_gens} generations")
        print(f"📊 Async config: batch_size={self.async_config.async_batch_size}, "
              f"concurrent_requests={self.async_config.max_concurrent_requests}, "
              f"genome_batch_size={self.async_config.genome_batch_size}")
        
        start_time = time.time()
        
        # Evolution loop
        for generation in range(max_gens):
            result = await self.evolve_generation_async()
            
            # Print progress with more detail
            best_genome = self.population.get_best_genome()
            best_accuracy = best_genome.fitness if hasattr(best_genome, 'fitness') else 0.0

            print(f"Gen {result.generation:2d}: best={result.best_fitness:.3f} | "
                  f"mean={result.mean_fitness:.3f} | div={result.diversity:.3f} | "
                  f"{result.evaluation_time:.1f}s")
            print(f"  📊 Best genome accuracy: {best_accuracy:.3f} | "
                  f"Population size: {len(self.population)} genomes")
            
            # Save checkpoint
            if (self.config.save_checkpoints and 
                result.generation % self.config.checkpoint_interval == 0):
                self._save_checkpoint(result.generation)
            
            # Check convergence
            if result.convergence_status.converged:
                print(f"🎯 Converged after {result.generation} generations: "
                      f"{result.convergence_status.reason.value}")
                break
            
            # Check target fitness
            if (self.config.target_fitness and 
                result.best_fitness >= self.config.target_fitness):
                print(f"🎯 Target fitness {self.config.target_fitness} reached!")
                break
        
        total_time = time.time() - start_time
        
        # Performance summary
        performance_summary = self._generate_performance_summary(total_time)
        
        return {
            'best_genome': self.best_genome_ever,
            'best_fitness': self.best_fitness_ever,
            'total_generations': len(self.generation_results),
            'total_time': total_time,
            'total_evaluations': self.total_evaluations,
            'generation_results': self.generation_results,
            'performance_summary': performance_summary,
            'async_stats': self.async_evaluation_pipeline.get_statistics()
        }
    
    def _generate_performance_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate performance summary comparing async vs sync if available."""
        summary = {
            'total_runtime': total_time,
            'async_evaluation_times': self.async_evaluation_times,
            'average_async_eval_time': (sum(self.async_evaluation_times) / len(self.async_evaluation_times) 
                                      if self.async_evaluation_times else 0.0),
        }
        
        if self.sync_evaluation_times:
            summary.update({
                'sync_evaluation_times': self.sync_evaluation_times,
                'average_sync_eval_time': sum(self.sync_evaluation_times) / len(self.sync_evaluation_times),
                'speedup_factor': (sum(self.sync_evaluation_times) / sum(self.async_evaluation_times) 
                                 if self.async_evaluation_times else 1.0)
            })
        
        return summary
