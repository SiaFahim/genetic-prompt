#!/usr/bin/env python3
"""
Test script to verify concurrent batch processing performance improvements.
"""

import sys
import os
import asyncio
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.utils.config import config
from src.embeddings.vocabulary import vocabulary

async def test_concurrent_batch_performance():
    """Test that concurrent batch processing achieves expected performance."""
    print("🧪 Testing Concurrent Batch Performance")
    print("=" * 50)
    
    # Initialize vocabulary first
    print("Initializing vocabulary...")
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print(f"✅ Vocabulary loaded: {len(vocabulary.token_to_id)} tokens")
    else:
        print("📚 Creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
        print(f"✅ Basic vocabulary created: {len(vocabulary.token_to_id)} tokens")
    
    # Create optimized config for performance testing
    config_obj = AsyncEvolutionConfig(
        population_size=10,  # Small population for quick test
        max_generations=2,   # Just 2 generations
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=2,
        
        # Async settings optimized for performance
        enable_async_evaluation=True,
        async_batch_size=10,  # Larger batches
        max_concurrent_requests=5,  # More concurrency
        genome_batch_size=5,
        max_concurrent_genomes=3,
        rate_limit_per_minute=500,  # Higher rate limit
        detailed_performance_logging=True
    )
    
    print(f"Configuration:")
    print(f"  Population: {config_obj.population_size} genomes")
    print(f"  Generations: {config_obj.max_generations}")
    print(f"  Async batch size: {config_obj.async_batch_size}")
    print(f"  Max concurrent requests: {config_obj.max_concurrent_requests}")
    print(f"  Genome batch size: {config_obj.genome_batch_size}")
    print(f"  Max concurrent genomes: {config_obj.max_concurrent_genomes}")
    
    # Test seed prompts
    seed_prompts = [
        "Solve step by step:",
        "Calculate carefully:",
        "Find the answer:",
        "Work through this problem:",
        "Let's solve this:"
    ]
    
    try:
        print("\nCreating AsyncEvolutionController...")
        controller = AsyncEvolutionController(
            config=config_obj,
            seed_prompts=seed_prompts
        )
        
        print(f"✅ Controller created with {len(controller.population)} genomes")
        
        # Run a short evolution to test performance
        print("\nRunning evolution test...")
        start_time = time.time()
        
        results = await controller.run_evolution_async(max_generations=config_obj.max_generations)
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Generations: {results.get('total_generations', 0)}")
        print(f"   Best fitness: {results.get('best_fitness', 0):.3f}")
        
        # Analyze async stats
        async_stats = results.get('async_stats', {})
        if async_stats:
            pipeline_stats = async_stats.get('pipeline_stats', {})
            llm_stats = async_stats.get('llm_interface_stats', {})
            
            print(f"\n⚡ Async Performance:")
            print(f"   Total evaluations: {pipeline_stats.get('total_evaluations', 0)}")
            print(f"   Cache hits: {pipeline_stats.get('cache_hits', 0)}")
            print(f"   API calls: {pipeline_stats.get('api_calls_made', 0)}")
            print(f"   Throughput: {llm_stats.get('throughput_requests_per_second', 0):.2f} req/s")
            
            # Rate limiting stats
            rate_limiting = llm_stats.get('rate_limiting', {})
            if rate_limiting:
                print(f"   Rate limit hits: {rate_limiting.get('rate_limit_hits', 0)}")
                print(f"   Avg wait time: {rate_limiting.get('average_wait_time', 0):.2f}s")
        
        # Performance validation
        success = True
        
        # Check that evolution completed
        if results.get('total_generations', 0) < config_obj.max_generations:
            print(f"⚠️  Evolution completed only {results.get('total_generations', 0)} generations")
            success = False
        else:
            print(f"✅ Evolution completed all {config_obj.max_generations} generations")
        
        # Check that population was properly initialized and evolved
        if results.get('best_fitness', 0) > 0:
            print(f"✅ Evolution produced fitness > 0: {results.get('best_fitness', 0):.3f}")
        else:
            print(f"⚠️  Evolution produced low fitness: {results.get('best_fitness', 0):.3f}")
        
        # Check concurrent processing worked
        if async_stats and pipeline_stats.get('total_evaluations', 0) > 0:
            print(f"✅ Async evaluation pipeline processed {pipeline_stats.get('total_evaluations', 0)} evaluations")
        else:
            print(f"⚠️  No async evaluations recorded")
            success = False
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_concurrent_batch_performance())
        if success:
            print("\n🎉 Concurrent batch processing is working!")
            sys.exit(0)
        else:
            print("\n💥 Concurrent batch processing needs improvement!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
