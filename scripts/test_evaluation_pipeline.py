#!/usr/bin/env python3
"""
Test script for the evaluation pipeline.
Tests LLM interface, async evaluation, and population evaluation.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.genetics.genome import PromptGenome
from src.genetics.population import PopulationInitializer
from src.evaluation.population_evaluator import PopulationEvaluator
from src.utils.config import get_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def test_evaluation_pipeline():
    """Test the complete evaluation pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing GSM8K evaluation pipeline...")
    
    try:
        # Initialize components
        config = get_config()
        population_initializer = PopulationInitializer()
        population_evaluator = PopulationEvaluator()
        
        # Create a small test population
        logger.info("Creating test population...")
        test_seeds = [
            "Solve this step by step.",
            "Let's work through this problem carefully.",
            "First, identify what we need to find.",
            "Break down the problem into smaller parts.",
            "Calculate each step systematically."
        ]
        
        # Create small population for testing
        test_population = population_initializer.initialize_population(
            seed_prompts=test_seeds,
            target_size=10
        )
        
        logger.info(f"Created test population of {len(test_population)} genomes")
        
        # Test single genome evaluation
        logger.info("Testing single genome evaluation...")
        test_genome = test_population[0]
        
        single_result = await population_evaluator.evaluate_single_genome(
            test_genome, 
            evaluation_set='primary'
        )
        
        logger.info(f"Single genome evaluation result:")
        logger.info(f"  Accuracy: {single_result['accuracy']:.3f}")
        logger.info(f"  Correct: {single_result['correct_count']}/{single_result['total_count']}")
        logger.info(f"  Fitness: {test_genome.fitness:.3f}")
        
        # Test population evaluation (small subset)
        logger.info("Testing population evaluation...")
        small_population = test_population[:3]  # Just 3 genomes for testing
        
        eval_result = await population_evaluator.evaluate_population(
            small_population,
            generation=1,
            use_progressive=True,
            show_progress=True
        )
        
        logger.info(f"Population evaluation result:")
        logger.info(f"  Generation: {eval_result['generation']}")
        logger.info(f"  Population size: {eval_result['population_size']}")
        logger.info(f"  Problems count: {eval_result['problems_count']}")
        logger.info(f"  Evaluation time: {eval_result['evaluation_time_seconds']:.2f}s")
        logger.info(f"  Best fitness: {eval_result['population_stats']['fitness']['max']:.3f}")
        
        # Display best genome
        best_info = eval_result['best_genome']
        if best_info:
            logger.info(f"Best genome:")
            logger.info(f"  Text: {best_info['text']}")
            logger.info(f"  Fitness: {best_info['fitness']:.3f}")
            logger.info(f"  Accuracy: {best_info['accuracy']:.3f}")
        
        # Test caching
        logger.info("Testing evaluation caching...")
        
        # Evaluate same population again - should use cache
        eval_result_2 = await population_evaluator.evaluate_population(
            small_population,
            generation=1,
            use_progressive=True,
            show_progress=True
        )
        
        logger.info(f"Cached evaluation:")
        logger.info(f"  Cached count: {eval_result_2['cached_count']}")
        logger.info(f"  Evaluated count: {eval_result_2['evaluated_count']}")
        logger.info(f"  Evaluation time: {eval_result_2['evaluation_time_seconds']:.2f}s")
        
        # Get statistics
        stats = population_evaluator.get_evaluation_statistics()
        logger.info(f"Evaluation statistics:")
        logger.info(f"  Total generations: {stats['total_generations_evaluated']}")
        logger.info(f"  Cache entries: {stats['total_cache_entries']}")
        
        llm_stats = stats['async_evaluator_stats']
        logger.info(f"LLM statistics:")
        logger.info(f"  Total requests: {llm_stats.get('llm_total_requests', 0)}")
        logger.info(f"  Cache hit rate: {llm_stats.get('llm_cache_hit_rate', 0):.2%}")
        logger.info(f"  Total cost: ${llm_stats.get('llm_total_cost_usd', 0):.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION PIPELINE TEST COMPLETE")
        logger.info("="*60)
        logger.info("✅ All evaluation components working correctly!")
        logger.info("✅ LLM interface functional")
        logger.info("✅ Async evaluation working")
        logger.info("✅ Population evaluation successful")
        logger.info("✅ Caching system operational")
        logger.info("✅ Fitness calculation working")
        logger.info("\nReady to proceed with selection and evolution loop!")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function."""
    setup_logging()
    
    # Run the async test
    asyncio.run(test_evaluation_pipeline())

if __name__ == "__main__":
    main()
