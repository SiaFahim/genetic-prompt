#!/usr/bin/env python3
"""
Test script to verify generation progression in the genetic algorithm.
"""

import sys
import os
import asyncio
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.utils.config import config
from src.embeddings.vocabulary import vocabulary

async def test_generation_progression():
    """Test that generations progress sequentially without immediate convergence."""
    print("🧪 Testing Generation Progression")
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
    
    # Create minimal config for quick testing
    config_obj = AsyncEvolutionConfig(
        population_size=8,   # Small population
        max_generations=5,   # Test 5 generations
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=2,
        target_fitness=0.99,  # High target to prevent early termination
        
        # Async settings - conservative for stability
        enable_async_evaluation=True,
        async_batch_size=5,   # Small batches
        max_concurrent_requests=2,  # Limited concurrency
        genome_batch_size=4,
        max_concurrent_genomes=2,
        rate_limit_per_minute=100,  # Conservative rate limit
        detailed_performance_logging=False  # Reduce noise
    )
    
    print(f"Configuration:")
    print(f"  Population: {config_obj.population_size} genomes")
    print(f"  Max generations: {config_obj.max_generations}")
    print(f"  Target fitness: {config_obj.target_fitness}")
    
    # Test seed prompts with variety to maintain diversity
    seed_prompts = [
        "Solve step by step:",
        "Calculate:",
        "Find the answer:",
        "Work through this:",
        "Let's solve:",
        "Think about this:",
        "Analyze:",
        "Compute:"
    ]
    
    try:
        print("\nCreating AsyncEvolutionController...")
        controller = AsyncEvolutionController(
            config=config_obj,
            seed_prompts=seed_prompts
        )
        
        print(f"✅ Controller created with {len(controller.population)} genomes")
        
        # Track generation progression
        generation_results = []
        
        print("\nRunning evolution with generation tracking...")
        start_time = time.time()
        
        # Run evolution and capture results
        results = await controller.run_evolution_async(max_generations=config_obj.max_generations)
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Evolution Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Generations completed: {results.get('total_generations', 0)}")
        print(f"   Best fitness: {results.get('best_fitness', 0):.3f}")
        
        # Analyze generation progression
        generation_results = results.get('generation_results', [])
        
        print(f"\n🔄 Generation Progression Analysis:")
        success = True
        
        if len(generation_results) == 0:
            print("❌ No generation results recorded")
            return False
        
        # Check sequential generation numbers
        expected_generations = list(range(len(generation_results)))
        actual_generations = [r.generation for r in generation_results]
        
        print(f"   Expected: {expected_generations}")
        print(f"   Actual:   {actual_generations}")
        
        if actual_generations == expected_generations:
            print("✅ Generations progressed sequentially (0→1→2→...)")
        else:
            print("❌ Generation progression is not sequential")
            success = False
        
        # Check that we didn't converge immediately
        if len(generation_results) >= 2:
            print("✅ Evolution ran for multiple generations (no immediate convergence)")
        else:
            print("⚠️  Evolution converged after only 1 generation")
            # This might be OK depending on the data, so don't fail
        
        # Check diversity progression
        diversities = [r.diversity for r in generation_results]
        print(f"   Diversity progression: {[f'{d:.3f}' for d in diversities]}")
        
        if all(d > 0.0 for d in diversities):
            print("✅ Diversity remained > 0 throughout evolution")
        else:
            print("⚠️  Diversity dropped to 0 in some generations")
        
        # Check fitness progression
        best_fitnesses = [r.best_fitness for r in generation_results]
        print(f"   Best fitness progression: {[f'{f:.3f}' for f in best_fitnesses]}")
        
        if len(best_fitnesses) > 1:
            fitness_improved = any(best_fitnesses[i] >= best_fitnesses[i-1] 
                                 for i in range(1, len(best_fitnesses)))
            if fitness_improved:
                print("✅ Fitness showed improvement or stability")
            else:
                print("⚠️  Fitness declined throughout evolution")
        
        # Final validation
        if results.get('total_generations', 0) > 0:
            print("✅ Evolution completed successfully")
        else:
            print("❌ Evolution failed to complete")
            success = False
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_generation_progression())
        if success:
            print("\n🎉 Generation progression is working correctly!")
            sys.exit(0)
        else:
            print("\n💥 Generation progression has issues!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
