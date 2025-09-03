#!/usr/bin/env python3
"""
Performance testing script for async batch evaluation system.

This script compares the performance of the new async batch processing system
against the original sequential evaluation approach, providing detailed metrics
and recommendations for optimal configuration.
"""

import asyncio
import time
import json
import sys
from typing import Dict, Any, List
from dataclasses import asdict
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.genetics.evolution import EvolutionController, EvolutionConfig
from src.genetics.population import Population
from src.genetics.genome import PromptGenome
from src.evaluation.async_pipeline import AsyncEvaluationPipeline, PopulationBatchConfig
from src.evaluation.async_llm_interface import AsyncLLMInterface, BatchConfig
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.llm_interface import LLMInterface
from src.utils.dataset import gsm8k_dataset
from src.config.hyperparameters import get_hyperparameter_config


class PerformanceBenchmark:
    """Comprehensive performance benchmark for async vs sync evaluation."""
    
    def __init__(self, test_config: Dict[str, Any]):
        """
        Initialize benchmark with test configuration.
        
        Args:
            test_config: Dictionary containing test parameters
        """
        self.test_config = test_config
        self.results = {
            'test_config': test_config,
            'sync_results': {},
            'async_results': {},
            'performance_comparison': {},
            'recommendations': {}
        }
    
    def create_test_population(self, size: int) -> Population:
        """Create a test population with diverse prompts."""
        population = Population(size)
        
        # Sample seed prompts for testing
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
            "Let me analyze this problem:"
        ]
        
        # Create genomes with varied prompts
        for i in range(size):
            prompt_text = seed_prompts[i % len(seed_prompts)]
            if i >= len(seed_prompts):
                prompt_text += f" (variant {i // len(seed_prompts)})"
            
            genome = PromptGenome.from_text(prompt_text)
            population.add_genome(genome)
        
        return population
    
    async def benchmark_async_evaluation(self, population: Population, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark async evaluation performance."""
        print(f"🚀 Testing async evaluation with {len(population)} genomes, {len(problems)} problems")
        
        # Create async pipeline with test configuration
        batch_config = BatchConfig(
            batch_size=self.test_config['async_batch_size'],
            max_concurrent_requests=self.test_config['max_concurrent_requests'],
            rate_limit_per_minute=self.test_config['rate_limit_per_minute']
        )
        
        async_llm = AsyncLLMInterface(batch_config=batch_config)
        
        population_batch_config = PopulationBatchConfig(
            genome_batch_size=self.test_config['genome_batch_size'],
            problem_batch_size=self.test_config['async_batch_size'],
            max_concurrent_genomes=self.test_config['max_concurrent_genomes'],
            detailed_logging=True
        )
        
        async_pipeline = AsyncEvaluationPipeline(
            async_llm_interface=async_llm,
            population_batch_config=population_batch_config,
            max_problems=len(problems)
        )
        
        # Run benchmark
        start_time = time.time()
        result = await async_pipeline.evaluate_population_async(population, problems)
        total_time = time.time() - start_time
        
        # Collect detailed metrics
        async_stats = async_pipeline.get_statistics()
        
        return {
            'total_time': total_time,
            'successful_evaluations': result.successful_evaluations,
            'failed_evaluations': result.failed_evaluations,
            'total_api_calls': result.total_api_calls,
            'total_cache_hits': result.total_cache_hits,
            'average_evaluation_time': result.average_evaluation_time,
            'throughput_problems_per_second': result.throughput_problems_per_second,
            'pipeline_stats': async_stats,
            'problems_processed': len(population) * len(problems),
            'genomes_per_second': len(population) / total_time if total_time > 0 else 0
        }
    
    def benchmark_sync_evaluation(self, population: Population, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark sync evaluation performance."""
        print(f"🐌 Testing sync evaluation with {len(population)} genomes, {len(problems)} problems")
        
        # Create sync pipeline
        sync_llm = LLMInterface()
        sync_pipeline = EvaluationPipeline(
            llm_interface=sync_llm,
            max_problems=len(problems)
        )
        
        # Run benchmark
        start_time = time.time()
        results = sync_pipeline.evaluate_population(population, problems)
        total_time = time.time() - start_time
        
        # Collect metrics
        sync_stats = sync_llm.get_statistics()
        
        return {
            'total_time': total_time,
            'successful_evaluations': len(results),
            'failed_evaluations': len(population) - len(results),
            'total_api_calls': sync_stats['total_requests'],
            'total_cache_hits': sync_stats['cache_hits'],
            'average_evaluation_time': total_time / len(population) if len(population) > 0 else 0,
            'throughput_problems_per_second': (len(population) * len(problems)) / total_time if total_time > 0 else 0,
            'pipeline_stats': sync_stats,
            'problems_processed': len(population) * len(problems),
            'genomes_per_second': len(population) / total_time if total_time > 0 else 0
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("🧪 Starting comprehensive async vs sync performance benchmark")
        print(f"Test configuration: {json.dumps(self.test_config, indent=2)}")
        
        # Create test data
        population = self.create_test_population(self.test_config['population_size'])
        problems = gsm8k_dataset.get_primary_eval_set()[:self.test_config['num_problems']]
        
        print(f"Created test population: {len(population)} genomes")
        print(f"Using problems: {len(problems)} GSM8K problems")
        
        # Benchmark async evaluation
        print("\n" + "="*50)
        print("ASYNC EVALUATION BENCHMARK")
        print("="*50)
        
        async_results = await self.benchmark_async_evaluation(population, problems)
        self.results['async_results'] = async_results
        
        # Benchmark sync evaluation (create fresh population to ensure fair comparison)
        print("\n" + "="*50)
        print("SYNC EVALUATION BENCHMARK")
        print("="*50)
        
        fresh_population = self.create_test_population(self.test_config['population_size'])
        sync_results = self.benchmark_sync_evaluation(fresh_population, problems)
        self.results['sync_results'] = sync_results
        
        # Calculate performance comparison
        self.results['performance_comparison'] = self._calculate_performance_comparison()
        
        # Generate recommendations
        self.results['recommendations'] = self._generate_recommendations()
        
        return self.results
    
    def _calculate_performance_comparison(self) -> Dict[str, Any]:
        """Calculate detailed performance comparison metrics."""
        async_res = self.results['async_results']
        sync_res = self.results['sync_results']
        
        speedup_factor = sync_res['total_time'] / async_res['total_time'] if async_res['total_time'] > 0 else 0
        throughput_improvement = (async_res['throughput_problems_per_second'] / 
                                sync_res['throughput_problems_per_second'] 
                                if sync_res['throughput_problems_per_second'] > 0 else 0)
        
        api_efficiency_async = (async_res['problems_processed'] / 
                              max(async_res['total_api_calls'], 1))
        api_efficiency_sync = (sync_res['problems_processed'] / 
                             max(sync_res['total_api_calls'], 1))
        
        return {
            'speedup_factor': speedup_factor,
            'time_saved_seconds': sync_res['total_time'] - async_res['total_time'],
            'time_saved_percentage': ((sync_res['total_time'] - async_res['total_time']) / 
                                    sync_res['total_time'] * 100 if sync_res['total_time'] > 0 else 0),
            'throughput_improvement_factor': throughput_improvement,
            'async_throughput': async_res['throughput_problems_per_second'],
            'sync_throughput': sync_res['throughput_problems_per_second'],
            'api_efficiency_improvement': api_efficiency_async / api_efficiency_sync if api_efficiency_sync > 0 else 0,
            'cache_hit_rate_async': (async_res['total_cache_hits'] / 
                                   max(async_res['total_cache_hits'] + async_res['total_api_calls'], 1)),
            'cache_hit_rate_sync': (sync_res['total_cache_hits'] / 
                                  max(sync_res['total_cache_hits'] + sync_res['total_api_calls'], 1))
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on benchmark results."""
        comparison = self.results['performance_comparison']
        async_res = self.results['async_results']
        
        recommendations = {
            'overall_recommendation': '',
            'optimal_batch_sizes': {},
            'concurrency_recommendations': {},
            'cost_analysis': {},
            'scaling_projections': {}
        }
        
        # Overall recommendation
        if comparison['speedup_factor'] > 2.0:
            recommendations['overall_recommendation'] = "Excellent performance improvement! Async evaluation is highly recommended."
        elif comparison['speedup_factor'] > 1.5:
            recommendations['overall_recommendation'] = "Good performance improvement. Async evaluation is recommended."
        elif comparison['speedup_factor'] > 1.1:
            recommendations['overall_recommendation'] = "Moderate improvement. Consider async for larger populations."
        else:
            recommendations['overall_recommendation'] = "Limited improvement. Review configuration or consider sync evaluation."
        
        # Batch size recommendations
        current_batch_size = self.test_config['async_batch_size']
        if async_res['throughput_problems_per_second'] < 10:
            recommendations['optimal_batch_sizes']['problem_batch_size'] = min(current_batch_size * 2, 50)
        else:
            recommendations['optimal_batch_sizes']['problem_batch_size'] = current_batch_size
        
        # Scaling projections
        recommendations['scaling_projections'] = {
            'population_100': f"{100 * async_res['average_evaluation_time']:.1f}s estimated",
            'population_500': f"{500 * async_res['average_evaluation_time']:.1f}s estimated",
            'population_1000': f"{1000 * async_res['average_evaluation_time']:.1f}s estimated"
        }
        
        return recommendations
    
    def print_results_summary(self):
        """Print a comprehensive results summary."""
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        comparison = self.results['performance_comparison']
        
        print(f"🚀 Speedup Factor: {comparison['speedup_factor']:.2f}x")
        print(f"⏱️  Time Saved: {comparison['time_saved_seconds']:.1f}s ({comparison['time_saved_percentage']:.1f}%)")
        print(f"📈 Throughput Improvement: {comparison['throughput_improvement_factor']:.2f}x")
        print(f"🎯 Async Throughput: {comparison['async_throughput']:.1f} problems/second")
        print(f"🐌 Sync Throughput: {comparison['sync_throughput']:.1f} problems/second")
        
        print(f"\n💡 {self.results['recommendations']['overall_recommendation']}")
        
        print(f"\n📊 Scaling Projections:")
        for scale, time in self.results['recommendations']['scaling_projections'].items():
            print(f"   {scale}: {time}")


async def main():
    """Main benchmark execution."""
    # Test configurations to benchmark
    test_configs = [
        {
            'name': 'Small Population Test',
            'population_size': 10,
            'num_problems': 20,
            'async_batch_size': 10,
            'max_concurrent_requests': 5,
            'genome_batch_size': 5,
            'max_concurrent_genomes': 3,
            'rate_limit_per_minute': 3500
        },
        {
            'name': 'Medium Population Test',
            'population_size': 30,
            'num_problems': 50,
            'async_batch_size': 20,
            'max_concurrent_requests': 10,
            'genome_batch_size': 10,
            'max_concurrent_genomes': 5,
            'rate_limit_per_minute': 3500
        }
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧪 Running benchmark: {config['name']}")
        benchmark = PerformanceBenchmark(config)
        results = await benchmark.run_comprehensive_benchmark()
        benchmark.print_results_summary()
        all_results.append(results)
        
        # Save individual results
        with open(f"benchmark_results_{config['name'].lower().replace(' ', '_')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Save combined results
    with open("comprehensive_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Benchmark complete! Results saved to JSON files.")


if __name__ == "__main__":
    asyncio.run(main())
