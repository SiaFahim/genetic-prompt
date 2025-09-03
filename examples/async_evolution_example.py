#!/usr/bin/env python3
"""
Complete example demonstrating the async batch evaluation system.

This example shows how to use the new asynchronous evaluation pipeline
with batch processing for significantly improved performance in genetic
algorithm evolution of prompts.
"""

import asyncio
import time
import sys
from typing import List
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.genetics.population import Population
from src.genetics.genome import PromptGenome
from src.evaluation.async_pipeline import AsyncEvaluationPipeline, PopulationBatchConfig
from src.evaluation.async_llm_interface import AsyncLLMInterface, BatchConfig
from src.utils.dataset import gsm8k_dataset


async def demonstrate_async_evaluation():
    """Demonstrate the async evaluation system with a small example."""
    print("🚀 Async Batch Evaluation Demonstration")
    print("="*50)
    
    # Configuration for optimal performance
    batch_config = BatchConfig(
        batch_size=20,                    # Process 20 problems per batch
        max_concurrent_requests=10,       # 10 concurrent API calls per batch
        rate_limit_per_minute=3500,      # Stay under OpenAI rate limits
        retry_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        timeout=30
    )
    
    population_batch_config = PopulationBatchConfig(
        genome_batch_size=10,            # Process 10 genomes concurrently
        problem_batch_size=20,           # 20 problems per batch within each genome
        max_concurrent_genomes=5,        # 5 genomes evaluated simultaneously
        enable_progress_bar=True,
        detailed_logging=True
    )
    
    # Create async LLM interface
    async_llm = AsyncLLMInterface(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=150,
        batch_config=batch_config
    )
    
    # Create async evaluation pipeline
    async_pipeline = AsyncEvaluationPipeline(
        async_llm_interface=async_llm,
        population_batch_config=population_batch_config,
        max_problems=50  # Evaluate on 50 problems for demo
    )
    
    # Create a test population
    population = Population(20)  # 20 genomes
    
    seed_prompts = [
        "Solve this step by step:",
        "Let's think through this problem carefully:",
        "To find the answer, I need to:",
        "Breaking this down into steps:",
        "First, let me understand what's being asked:",
        "I'll solve this systematically:",
        "Let's work through this problem:",
        "To calculate this, I'll:",
        "Step-by-step solution:",
        "Let me analyze this problem:",
        "I'll approach this methodically:",
        "Let's break this into smaller parts:",
        "To solve this, I'll start by:",
        "Working through this step by step:",
        "Let me think about this logically:",
        "I need to calculate:",
        "Let's solve this problem:",
        "To find the solution:",
        "I'll work through this carefully:",
        "Let me solve this systematically:"
    ]
    
    for i, prompt in enumerate(seed_prompts):
        genome = PromptGenome.from_text(prompt)
        population.add_genome(genome)
    
    # Get test problems
    problems = gsm8k_dataset.get_primary_eval_set()[:50]
    
    print(f"📊 Test Setup:")
    print(f"   Population size: {len(population)} genomes")
    print(f"   Problems: {len(problems)} GSM8K problems")
    print(f"   Total evaluations: {len(population) * len(problems)} = {len(population) * len(problems)}")
    print(f"   Batch configuration: {batch_config.batch_size} problems/batch, {batch_config.max_concurrent_requests} concurrent")
    print(f"   Genome batching: {population_batch_config.genome_batch_size} genomes/batch, {population_batch_config.max_concurrent_genomes} concurrent")
    
    # Run async evaluation
    print(f"\n🚀 Starting async evaluation...")
    start_time = time.time()
    
    result = await async_pipeline.evaluate_population_async(population, problems)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n✅ Async Evaluation Complete!")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Successful evaluations: {result.successful_evaluations}")
    print(f"   Failed evaluations: {result.failed_evaluations}")
    print(f"   API calls made: {result.total_api_calls}")
    print(f"   Cache hits: {result.total_cache_hits}")
    print(f"   Average evaluation time per genome: {result.average_evaluation_time:.2f}s")
    print(f"   Throughput: {result.throughput_problems_per_second:.1f} problems/second")
    
    # Show top performing genomes
    population.genomes.sort(key=lambda g: g.fitness or 0, reverse=True)
    print(f"\n🏆 Top 3 Performing Prompts:")
    for i, genome in enumerate(population.genomes[:3]):
        print(f"   {i+1}. Fitness: {genome.fitness:.3f} - '{genome.to_text()}'")
    
    return result


async def demonstrate_full_evolution():
    """Demonstrate a complete evolution run with async evaluation."""
    print("\n" + "="*70)
    print("🧬 COMPLETE ASYNC EVOLUTION DEMONSTRATION")
    print("="*70)
    
    # Create async evolution configuration
    config = AsyncEvolutionConfig(
        population_size=30,
        max_generations=10,  # Short demo
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=5,
        target_fitness=0.7,  # Stop early if we reach 70% accuracy
        
        # Async-specific settings
        enable_async_evaluation=True,
        async_batch_size=20,
        max_concurrent_requests=10,
        genome_batch_size=10,
        max_concurrent_genomes=5,
        rate_limit_per_minute=3500,
        detailed_performance_logging=True
    )
    
    # Seed prompts for initial population
    seed_prompts = [
        "Solve this step by step:",
        "Let's think through this problem:",
        "To find the answer:",
        "Breaking this down:",
        "I need to calculate:",
        "Let me work through this:",
        "Step-by-step solution:",
        "To solve this problem:",
        "Let me analyze this:",
        "I'll approach this systematically:"
    ]
    
    # Create and run async evolution controller
    controller = AsyncEvolutionController(
        config=config,
        seed_prompts=seed_prompts
    )
    
    print(f"🚀 Starting evolution with {config.population_size} genomes for {config.max_generations} generations")
    print(f"📊 Async settings: batch_size={config.async_batch_size}, concurrent_requests={config.max_concurrent_requests}")
    
    # Run evolution
    start_time = time.time()
    results = await controller.run_evolution_async()
    total_time = time.time() - start_time
    
    # Print final results
    print(f"\n🎯 Evolution Complete!")
    print(f"   Total runtime: {total_time:.2f} seconds")
    print(f"   Generations completed: {results['total_generations']}")
    print(f"   Best fitness achieved: {results['best_fitness']:.3f}")
    print(f"   Total evaluations: {results['total_evaluations']}")
    
    if results['best_genome']:
        print(f"   Best prompt: '{results['best_genome'].to_text()}'")
    
    # Performance summary
    perf_summary = results['performance_summary']
    print(f"\n📈 Performance Summary:")
    print(f"   Average evaluation time: {perf_summary['average_async_eval_time']:.2f}s per generation")
    
    if 'speedup_factor' in perf_summary:
        print(f"   Speedup vs sync: {perf_summary['speedup_factor']:.2f}x")
    
    # Async pipeline statistics
    async_stats = results['async_stats']
    pipeline_stats = async_stats['pipeline_stats']
    print(f"   Total API calls: {pipeline_stats['api_calls_made']}")
    print(f"   Cache hit rate: {pipeline_stats['cache_hits'] / max(pipeline_stats['total_evaluations'], 1) * 100:.1f}%")
    
    return results


def compare_configurations():
    """Compare different batch configurations to find optimal settings."""
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION OPTIMIZATION GUIDE")
    print("="*70)
    
    configurations = [
        {
            'name': 'Conservative',
            'description': 'Safe settings for rate limit compliance',
            'async_batch_size': 10,
            'max_concurrent_requests': 5,
            'genome_batch_size': 5,
            'max_concurrent_genomes': 3,
            'expected_speedup': '2-3x'
        },
        {
            'name': 'Balanced',
            'description': 'Recommended settings for most use cases',
            'async_batch_size': 20,
            'max_concurrent_requests': 10,
            'genome_batch_size': 10,
            'max_concurrent_genomes': 5,
            'expected_speedup': '3-5x'
        },
        {
            'name': 'Aggressive',
            'description': 'Maximum performance (monitor rate limits)',
            'async_batch_size': 30,
            'max_concurrent_requests': 15,
            'genome_batch_size': 15,
            'max_concurrent_genomes': 8,
            'expected_speedup': '5-8x'
        }
    ]
    
    print("📊 Recommended Configurations:")
    for config in configurations:
        print(f"\n🔧 {config['name']} Configuration:")
        print(f"   Description: {config['description']}")
        print(f"   Async batch size: {config['async_batch_size']}")
        print(f"   Max concurrent requests: {config['max_concurrent_requests']}")
        print(f"   Genome batch size: {config['genome_batch_size']}")
        print(f"   Max concurrent genomes: {config['max_concurrent_genomes']}")
        print(f"   Expected speedup: {config['expected_speedup']}")
    
    print(f"\n💡 Optimization Tips:")
    print(f"   • Start with 'Balanced' configuration")
    print(f"   • Monitor API rate limits and adjust accordingly")
    print(f"   • Increase batch sizes for larger populations")
    print(f"   • Use 'Conservative' if you hit rate limits")
    print(f"   • Try 'Aggressive' for maximum speed (with monitoring)")


async def main():
    """Main demonstration function."""
    print("🧬 Genetic Algorithm Async Batch Evaluation System")
    print("="*60)
    print("This demonstration shows the new async batch processing system")
    print("that provides significant performance improvements over sequential evaluation.")
    
    try:
        # Demonstrate async evaluation
        await demonstrate_async_evaluation()
        
        # Demonstrate full evolution
        await demonstrate_full_evolution()
        
        # Show configuration options
        compare_configurations()
        
        print(f"\n✅ All demonstrations completed successfully!")
        print(f"🚀 The async batch evaluation system is ready for production use.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print(f"💡 Make sure you have:")
        print(f"   • OpenAI API key set in .env file")
        print(f"   • All required dependencies installed")
        print(f"   • Sufficient API rate limits")


if __name__ == "__main__":
    asyncio.run(main())
