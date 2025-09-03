#!/usr/bin/env python3
"""
Test script to verify population initialization fix in AsyncEvolutionController.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.utils.config import config as app_config
from src.embeddings.vocabulary import vocabulary

def test_population_initialization():
    """Test that AsyncEvolutionController properly initializes population."""
    print("🧪 Testing Population Initialization Fix")
    print("=" * 50)

    # Initialize vocabulary first
    print("Initializing vocabulary...")
    vocab_file = app_config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print(f"✅ Vocabulary loaded: {len(vocabulary.token_to_id)} tokens")
    else:
        print("📚 Creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
        print(f"✅ Basic vocabulary created: {len(vocabulary.token_to_id)} tokens")
    
    # Create minimal config for testing
    config = AsyncEvolutionConfig(
        population_size=50,
        max_generations=5,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=5,
        
        # Async settings
        enable_async_evaluation=True,
        async_batch_size=10,
        max_concurrent_requests=3,
        genome_batch_size=5,
        max_concurrent_genomes=2,
        rate_limit_per_minute=100,
        detailed_performance_logging=False
    )
    
    # Test seed prompts
    seed_prompts = [
        "Solve step by step:",
        "Calculate carefully:",
        "Find the answer:"
    ]
    
    try:
        print("Creating AsyncEvolutionController...")
        controller = AsyncEvolutionController(
            config=config,
            seed_prompts=seed_prompts
        )
        
        # Check population size
        population_size = len(controller.population)
        print(f"Population size after initialization: {population_size}")
        
        # Verify population is not empty
        if population_size == 0:
            print("❌ FAILED: Population is empty (0 genomes)")
            return False
        elif population_size == config.population_size:
            print(f"✅ SUCCESS: Population correctly initialized with {population_size} genomes")
            return True
        else:
            print(f"⚠️  WARNING: Population size ({population_size}) doesn't match config ({config.population_size})")
            return population_size > 0
            
    except Exception as e:
        print(f"❌ ERROR: Failed to create AsyncEvolutionController: {e}")
        return False

if __name__ == "__main__":
    success = test_population_initialization()
    if success:
        print("\n🎉 Population initialization fix is working!")
        sys.exit(0)
    else:
        print("\n💥 Population initialization fix failed!")
        sys.exit(1)
