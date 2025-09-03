"""
Asynchronous Evaluation Pipeline for batch processing of populations.

This module provides an async version of the evaluation pipeline that supports:
- Batch processing of genomes within populations
- Concurrent evaluation of multiple genomes
- Integration with async LLM interface
- Enhanced progress tracking and performance monitoring
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from tqdm.asyncio import tqdm

from ..genetics.genome import PromptGenome
from ..genetics.population import Population
from .async_llm_interface import AsyncLLMInterface, BatchConfig, BatchResult
from .fitness import FitnessCalculator, FitnessComponents
from .cache import evaluation_cache
from .pipeline import EvaluationResult
from ..utils.dataset import gsm8k_dataset


@dataclass
class PopulationBatchConfig:
    """Configuration for population batch processing."""
    genome_batch_size: int = 10  # Number of genomes to process concurrently
    problem_batch_size: int = 20  # Number of problems per batch within each genome
    max_concurrent_genomes: int = 5  # Maximum concurrent genome evaluations
    enable_progress_bar: bool = True
    detailed_logging: bool = False


@dataclass
class PopulationEvaluationResult:
    """Result of evaluating an entire population."""
    evaluation_results: List[EvaluationResult]
    total_time: float
    total_api_calls: int
    total_cache_hits: int
    successful_evaluations: int
    failed_evaluations: int
    average_evaluation_time: float
    throughput_problems_per_second: float


class AsyncEvaluationPipeline:
    """Asynchronous evaluation pipeline with batch processing capabilities."""
    
    def __init__(self, 
                 async_llm_interface: Optional[AsyncLLMInterface] = None,
                 fitness_calculator: Optional[FitnessCalculator] = None,
                 use_cache: bool = True,
                 population_batch_config: Optional[PopulationBatchConfig] = None,
                 max_problems: int = 100):
        """
        Initialize async evaluation pipeline.
        
        Args:
            async_llm_interface: Async LLM interface for evaluation
            fitness_calculator: Fitness calculator
            use_cache: Whether to use caching
            population_batch_config: Population batch processing configuration
            max_problems: Maximum number of problems to evaluate on
        """
        self.async_llm_interface = async_llm_interface or AsyncLLMInterface()
        self.fitness_calculator = fitness_calculator or FitnessCalculator()
        self.use_cache = use_cache
        self.population_batch_config = population_batch_config or PopulationBatchConfig()
        self.max_problems = max_problems
        
        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.total_evaluation_time = 0.0
        self.api_calls_made = 0
        
        # Model info for caching
        self.model_info = {
            'model': self.async_llm_interface.model,
            'temperature': str(self.async_llm_interface.temperature),
            'max_tokens': str(self.async_llm_interface.max_tokens)
        }
    
    async def evaluate_genome_async(self, 
                                   genome: PromptGenome, 
                                   problems: List[Dict[str, Any]],
                                   progress_callback: Optional[Callable] = None) -> EvaluationResult:
        """
        Evaluate a single genome asynchronously.
        
        Args:
            genome: Genome to evaluate
            problems: List of problems to evaluate on
            progress_callback: Optional progress callback
            
        Returns:
            EvaluationResult
        """
        start_time = time.time()
        cache_hit = False
        
        # Limit number of problems
        eval_problems = problems[:self.max_problems]
        
        # Check cache first
        if self.use_cache:
            cache_key = evaluation_cache.get_cache_key(genome, eval_problems, self.model_info)
            cached_entry = evaluation_cache.get_evaluation(cache_key)
            
            if cached_entry:
                cache_hit = True
                self.cache_hits += 1
                
                # Reconstruct fitness components
                fitness_components = FitnessComponents(**cached_entry.fitness_components)
                
                evaluation_time = time.time() - start_time
                
                return EvaluationResult(
                    genome_id=genome.genome_id,
                    prompt_text=genome.to_text(),
                    fitness_components=fitness_components,
                    evaluation_results=cached_entry.evaluation_results,
                    evaluation_time=evaluation_time,
                    cache_hit=cache_hit
                )
        
        # Perform async evaluation
        prompt_text = genome.to_text()
        
        def progress_wrapper(batch_idx, total_batches, batch_info):
            if progress_callback:
                progress_callback(
                    genome.genome_id, 
                    batch_info.get('problems_processed', 0), 
                    len(eval_problems), 
                    batch_info
                )
        
        # Use async batch evaluation
        batch_result = await self.async_llm_interface.batch_evaluate_async(
            prompt_text, eval_problems, progress_wrapper
        )
        
        # Calculate fitness
        fitness_components = self.fitness_calculator.calculate_fitness(
            genome, batch_result.results
        )
        
        # Store in cache
        if self.use_cache:
            evaluation_cache.store_evaluation(
                cache_key, genome, eval_problems, batch_result.results,
                fitness_components, self.model_info
            )
        
        evaluation_time = time.time() - start_time
        self.total_evaluation_time += evaluation_time
        self.total_evaluations += 1
        self.api_calls_made += batch_result.api_calls_made
        
        return EvaluationResult(
            genome_id=genome.genome_id,
            prompt_text=prompt_text,
            fitness_components=fitness_components,
            evaluation_results=batch_result.results,
            evaluation_time=evaluation_time,
            cache_hit=cache_hit
        )
    
    async def evaluate_population_async(self, 
                                       population: Population,
                                       problems: List[Dict[str, Any]],
                                       progress_callback: Optional[Callable] = None) -> PopulationEvaluationResult:
        """
        Evaluate all genomes in a population asynchronously with batch processing.
        
        Args:
            population: Population to evaluate
            problems: List of problems to evaluate on
            progress_callback: Optional progress callback
            
        Returns:
            PopulationEvaluationResult
        """
        start_time = time.time()
        results = []
        total_api_calls = 0
        total_cache_hits = 0
        successful_evaluations = 0
        failed_evaluations = 0
        
        # Split population into batches for concurrent processing
        genome_batches = [
            population.genomes[i:i + self.population_batch_config.genome_batch_size]
            for i in range(0, len(population.genomes), self.population_batch_config.genome_batch_size)
        ]
        
        if self.population_batch_config.detailed_logging:
            print(f"🧬 Evaluating {len(population.genomes)} genomes in {len(genome_batches)} batches")
        
        # Progress tracking
        if self.population_batch_config.enable_progress_bar and progress_callback is None:
            pbar = tqdm(total=len(population.genomes), desc="Evaluating population")
            
            def default_progress(genome_id, current, total, result):
                pbar.set_postfix({
                    'genome': genome_id[:8],
                    'problem': f"{current}/{total}",
                    'correct': result.get('is_correct', False) if isinstance(result, dict) else False
                })
        else:
            default_progress = progress_callback
        
        # Process batches sequentially to manage resource usage
        for batch_idx, genome_batch in enumerate(genome_batches):
            batch_start_time = time.time()
            
            # Create semaphore to limit concurrent genome evaluations
            semaphore = asyncio.Semaphore(self.population_batch_config.max_concurrent_genomes)
            
            async def evaluate_genome_with_semaphore(genome):
                async with semaphore:
                    try:
                        result = await self.evaluate_genome_async(genome, problems, default_progress)
                        return result, None
                    except Exception as e:
                        return None, e
            
            # Process genome batch concurrently
            tasks = [evaluate_genome_with_semaphore(genome) for genome in genome_batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Process results and handle exceptions
            for genome, (result, error) in zip(genome_batch, batch_results):
                if error:
                    print(f"Error evaluating genome {genome.genome_id}: {error}")
                    failed_evaluations += 1
                    # Set a default low fitness for failed evaluations
                    genome.set_fitness(0.0)
                else:
                    results.append(result)
                    successful_evaluations += 1
                    total_api_calls += result.evaluation_results.__len__() if not result.cache_hit else 0
                    total_cache_hits += 1 if result.cache_hit else 0
                    
                    # Update genome fitness
                    genome.set_fitness(result.fitness_components.overall_fitness)
                    
                    if self.population_batch_config.enable_progress_bar and progress_callback is None:
                        pbar.update(1)
                        pbar.set_description(f"Evaluating population (fitness: {result.fitness_components.overall_fitness:.3f})")
            
            batch_time = time.time() - batch_start_time
            if self.population_batch_config.detailed_logging and len(genome_batches) > 1:
                print(f"  Batch {batch_idx + 1}/{len(genome_batches)}: {batch_time:.1f}s")
        
        if self.population_batch_config.enable_progress_bar and progress_callback is None:
            pbar.close()
        
        total_time = time.time() - start_time
        total_problems_processed = successful_evaluations * len(problems[:self.max_problems])
        
        return PopulationEvaluationResult(
            evaluation_results=results,
            total_time=total_time,
            total_api_calls=total_api_calls,
            total_cache_hits=total_cache_hits,
            successful_evaluations=successful_evaluations,
            failed_evaluations=failed_evaluations,
            average_evaluation_time=total_time / len(population.genomes) if len(population.genomes) > 0 else 0.0,
            throughput_problems_per_second=total_problems_processed / total_time if total_time > 0 else 0.0
        )

    async def evaluate_adaptive_async(self, population: Population, generation: int) -> PopulationEvaluationResult:
        """
        Evaluate population with adaptive problem selection asynchronously.

        Args:
            population: Population to evaluate
            generation: Current generation number

        Returns:
            PopulationEvaluationResult
        """
        # Get adaptive problem set
        problems = gsm8k_dataset.get_adaptive_eval_problems(generation)

        if self.population_batch_config.detailed_logging:
            print(f"📊 Gen {generation}: {len(problems)} problems")

        return await self.evaluate_population_async(population, problems)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        llm_stats = self.async_llm_interface.get_statistics()

        return {
            'pipeline_stats': {
                'total_evaluations': self.total_evaluations,
                'cache_hits': self.cache_hits,
                'total_evaluation_time': self.total_evaluation_time,
                'api_calls_made': self.api_calls_made,
                'average_evaluation_time': (self.total_evaluation_time / self.total_evaluations
                                          if self.total_evaluations > 0 else 0.0)
            },
            'llm_interface_stats': llm_stats,
            'batch_config': {
                'genome_batch_size': self.population_batch_config.genome_batch_size,
                'problem_batch_size': self.population_batch_config.problem_batch_size,
                'max_concurrent_genomes': self.population_batch_config.max_concurrent_genomes,
                'max_problems': self.max_problems
            }
        }


# Utility function to run async evaluation in sync context
def run_async_evaluation(pipeline: AsyncEvaluationPipeline,
                        population: Population,
                        problems: List[Dict[str, Any]]) -> PopulationEvaluationResult:
    """
    Run async evaluation in a synchronous context.

    Args:
        pipeline: AsyncEvaluationPipeline instance
        population: Population to evaluate
        problems: List of problems

    Returns:
        PopulationEvaluationResult
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    pipeline.evaluate_population_async(population, problems)
                )
                return future.result()
        else:
            # Run in existing loop
            return loop.run_until_complete(
                pipeline.evaluate_population_async(population, problems)
            )
    except RuntimeError:
        # No event loop exists, create new one
        return asyncio.run(pipeline.evaluate_population_async(population, problems))
