"""
Async evaluation system for genetic algorithm.
Handles concurrent evaluation of genomes with rate limiting and progress tracking.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
import logging

from tqdm.asyncio import tqdm

from src.genetics.genome import PromptGenome
from src.evaluation.llm_interface import LLMInterface
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class AsyncEvaluator:
    """Handles asynchronous evaluation of genomes."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize async evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.llm_interface = LLMInterface(config_path)
        
        # Configuration
        self.concurrency_limit = self.config.get('evaluation.concurrency_limit', 200)
        
        # Statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_time_seconds': 0,
            'avg_time_per_evaluation': 0
        }
    
    async def evaluate_genome_on_problems(self, genome: PromptGenome, 
                                        problems: List[Dict[str, Any]],
                                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate a single genome on multiple problems.
        
        Args:
            genome: Genome to evaluate
            problems: List of problem dictionaries with 'question' and 'final_answer'
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        prompt_text = genome.to_text()
        
        # Prepare evaluation pairs
        eval_pairs = [
            (prompt_text, problem['question'], problem['final_answer'])
            for problem in problems
        ]
        
        # Evaluate all problems
        results = await self.llm_interface.evaluate_batch(
            eval_pairs, 
            max_concurrent=min(self.concurrency_limit, len(eval_pairs))
        )
        
        # Calculate statistics
        correct_count = sum(1 for result in results if result.get('correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # Calculate response statistics
        response_times = [r.get('response_time_seconds', 0) for r in results if 'response_time_seconds' in r]
        total_tokens = sum(r.get('total_tokens', 0) for r in results)
        total_cost = sum(r.get('estimated_cost_usd', 0) for r in results)
        
        evaluation_time = time.time() - start_time
        
        # Update statistics
        self.evaluation_stats['total_evaluations'] += total_count
        self.evaluation_stats['successful_evaluations'] += sum(1 for r in results if 'error' not in r)
        self.evaluation_stats['failed_evaluations'] += sum(1 for r in results if 'error' in r)
        self.evaluation_stats['total_time_seconds'] += evaluation_time
        
        if self.evaluation_stats['total_evaluations'] > 0:
            self.evaluation_stats['avg_time_per_evaluation'] = (
                self.evaluation_stats['total_time_seconds'] / 
                self.evaluation_stats['total_evaluations']
            )
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(genome, accuracy, correct_count, total_count)
        
        evaluation_result = {
            'genome_id': genome.genome_id,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'individual_results': results,
            'evaluation_time_seconds': evaluation_time,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'total_tokens_used': total_tokens,
            'total_cost_usd': total_cost,
            'timestamp': time.time()
        }
        
        logger.debug(f"Evaluated genome {genome.genome_id[:8]}: "
                    f"{correct_count}/{total_count} correct ({accuracy:.3f})")
        
        return evaluation_result
    
    async def evaluate_population(self, population: List[PromptGenome],
                                problems: List[Dict[str, Any]],
                                show_progress: bool = True,
                                progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Evaluate an entire population on problems.
        
        Args:
            population: List of genomes to evaluate
            problems: List of problems to evaluate on
            show_progress: Whether to show progress bar
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating population of {len(population)} genomes on {len(problems)} problems")
        
        start_time = time.time()
        
        # Create semaphore for population-level concurrency
        population_semaphore = asyncio.Semaphore(min(50, len(population)))
        
        async def evaluate_with_semaphore(genome: PromptGenome) -> Dict[str, Any]:
            async with population_semaphore:
                return await self.evaluate_genome_on_problems(genome, problems, progress_callback)
        
        # Create tasks for all genomes
        tasks = [evaluate_with_semaphore(genome) for genome in population]
        
        # Execute with progress bar
        if show_progress:
            results = await tqdm.gather(*tasks, desc="Evaluating population")
        else:
            results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        logger.info(f"Population evaluation completed in {total_time:.2f} seconds")
        
        return results
    
    async def evaluate_population_progressive(self, population: List[PromptGenome],
                                            all_problems: List[Dict[str, Any]],
                                            generation: int,
                                            show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluate population with progressive problem count based on generation.
        
        Args:
            population: List of genomes to evaluate
            all_problems: Complete list of problems
            generation: Current generation number
            show_progress: Whether to show progress bar
            
        Returns:
            List of evaluation results
        """
        # Determine number of problems based on generation
        eval_config = self.config.get('evaluation.progressive_evaluation', {})
        
        if generation <= eval_config.get('early_generations', {}).get('range', [1, 10])[1]:
            problems_count = eval_config.get('early_generations', {}).get('problems_per_genome', 50)
        elif generation <= eval_config.get('middle_generations', {}).get('range', [11, 20])[1]:
            problems_count = eval_config.get('middle_generations', {}).get('problems_per_genome', 100)
        else:
            problems_count = eval_config.get('late_generations', {}).get('problems_per_genome', 150)
        
        # Select problems for this generation
        import random
        random.seed(self.config.get('experiment.random_seed', 42) + generation)
        
        if problems_count >= len(all_problems):
            selected_problems = all_problems
        else:
            selected_problems = random.sample(all_problems, problems_count)
        
        logger.info(f"Generation {generation}: Using {len(selected_problems)} problems for evaluation")
        
        return await self.evaluate_population(population, selected_problems, show_progress)
    
    async def evaluate_single_genome(self, genome: PromptGenome,
                                   problems: List[Dict[str, Any]]) -> float:
        """
        Evaluate a single genome and return just the accuracy.
        
        Args:
            genome: Genome to evaluate
            problems: List of problems
            
        Returns:
            Accuracy score
        """
        result = await self.evaluate_genome_on_problems(genome, problems)
        return result['accuracy']
    
    def update_genome_fitness(self, genome: PromptGenome, 
                            evaluation_result: Dict[str, Any]) -> None:
        """
        Update genome fitness based on evaluation result.
        
        Args:
            genome: Genome to update
            evaluation_result: Result from evaluation
        """
        accuracy = evaluation_result['accuracy']
        
        # Apply length penalty
        length_penalty = genome.get_length_penalty()
        fitness = accuracy * length_penalty
        
        # Update genome
        genome.set_fitness(fitness, accuracy)
        genome.increment_evaluation_count()
        genome.last_evaluated_timestamp = time.time()
        
        # Store evaluation metadata
        genome.metadata['last_evaluation'] = {
            'accuracy': accuracy,
            'fitness': fitness,
            'length_penalty': length_penalty,
            'correct_count': evaluation_result['correct_count'],
            'total_count': evaluation_result['total_count'],
            'evaluation_time': evaluation_result['evaluation_time_seconds'],
            'total_cost': evaluation_result['total_cost_usd']
        }
    
    def update_population_fitness(self, population: List[PromptGenome],
                                evaluation_results: List[Dict[str, Any]]) -> None:
        """
        Update fitness for entire population.
        
        Args:
            population: List of genomes
            evaluation_results: List of evaluation results
        """
        if len(population) != len(evaluation_results):
            raise ValueError("Population and evaluation results must have same length")
        
        for genome, result in zip(population, evaluation_results):
            self.update_genome_fitness(genome, result)
        
        # Log population statistics
        accuracies = [g.accuracy for g in population if g.accuracy is not None]
        fitnesses = [g.fitness for g in population if g.fitness is not None]
        
        if accuracies and fitnesses:
            logger.info(f"Population fitness - Accuracy: {sum(accuracies)/len(accuracies):.3f} "
                       f"(max: {max(accuracies):.3f}), "
                       f"Fitness: {sum(fitnesses)/len(fitnesses):.3f} "
                       f"(max: {max(fitnesses):.3f})")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        stats = self.evaluation_stats.copy()
        
        # Add LLM interface statistics
        llm_stats = self.llm_interface.get_statistics()
        stats.update({
            'llm_' + k: v for k, v in llm_stats.items()
        })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset evaluation statistics."""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_time_seconds': 0,
            'avg_time_per_evaluation': 0
        }
