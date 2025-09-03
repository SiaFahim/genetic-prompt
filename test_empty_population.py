#!/usr/bin/env python3
"""
Test script to verify empty population handling in AsyncEvolutionController.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.genetics.population import Population
from src.utils.config import config
from src.embeddings.vocabulary import vocabulary

async def test_empty_population_handling():
    """Test that empty population scenarios are caught with meaningful errors."""
    print("🧪 Testing Empty Population Handling")
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
    
    # Create minimal config for testing
    config_obj = AsyncEvolutionConfig(
        population_size=10,
        max_generations=5,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=2,
        
        # Async settings
        enable_async_evaluation=True,
        async_batch_size=5,
        max_concurrent_requests=2,
        genome_batch_size=3,
        max_concurrent_genomes=2,
        rate_limit_per_minute=100,
        detailed_performance_logging=False
    )
    
    print("Testing normal population initialization...")
    try:
        # Test 1: Normal initialization should work
        controller = AsyncEvolutionController(
            config=config_obj,
            seed_prompts=["Test prompt 1", "Test prompt 2"]
        )
        
        population_size = len(controller.population)
        if population_size > 0:
            print(f"✅ Normal initialization works: {population_size} genomes")
        else:
            print(f"❌ Normal initialization failed: {population_size} genomes")
            return False
            
    except Exception as e:
        print(f"❌ Normal initialization failed with error: {e}")
        return False
    
    print("\nTesting empty population detection...")
    try:
        # Test 2: Create controller and manually empty the population
        controller = AsyncEvolutionController(
            config=config_obj,
            seed_prompts=["Test prompt"]
        )
        
        # Manually empty the population to simulate the bug
        controller.population = Population(0)  # Empty population
        print(f"Population manually set to: {len(controller.population)} genomes")
        
        # Try to run evolution - should fail with meaningful error
        try:
            await controller.run_evolution_async(max_generations=1)
            print("❌ Evolution ran with empty population - validation failed!")
            return False
        except ValueError as e:
            error_msg = str(e)
            if "empty population" in error_msg.lower():
                print(f"✅ Empty population caught with error: {error_msg}")
                return True
            else:
                print(f"❌ Wrong error type: {error_msg}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_empty_population_handling())
        if success:
            print("\n🎉 Empty population handling is working!")
            sys.exit(0)
        else:
            print("\n💥 Empty population handling needs improvement!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
