#!/usr/bin/env python3
"""
Test script to verify the SeedPrompt object to text conversion fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
from src.seeds.seed_manager import SeedManager
from src.utils.config import config
from src.embeddings.vocabulary import vocabulary

def test_seed_prompt_conversion():
    """Test that SeedPrompt objects are properly converted to text strings."""
    print("🧪 Testing SeedPrompt Object to Text Conversion")
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
    
    # Load base seeds (these are SeedPrompt objects)
    print("\nLoading base seed collection...")
    seed_manager = SeedManager()
    base_seeds = seed_manager.get_base_seeds()
    print(f"✅ Loaded {len(base_seeds)} seed prompts")
    
    # Verify they are SeedPrompt objects
    if base_seeds:
        first_seed = base_seeds[0]
        print(f"✅ First seed type: {type(first_seed).__name__}")
        print(f"✅ First seed has text attribute: {hasattr(first_seed, 'text')}")
        print(f"✅ First seed text: \"{first_seed.text}\"")
    
    # Test the conversion (this is what the notebook now does)
    print("\nTesting SeedPrompt to text conversion...")
    seed_texts = [seed.text for seed in base_seeds[:5]] if base_seeds else None
    
    if seed_texts:
        print(f"✅ Converted {len(seed_texts)} SeedPrompt objects to text strings")
        print("✅ Sample converted texts:")
        for i, text in enumerate(seed_texts[:3], 1):
            print(f"   {i}. \"{text}\" (type: {type(text).__name__})")
    else:
        print("❌ No seed texts converted")
        return False
    
    # Test AsyncEvolutionController with converted texts
    print("\nTesting AsyncEvolutionController with converted seed texts...")
    
    # Create minimal config
    config_obj = AsyncEvolutionConfig(
        population_size=5,
        max_generations=2,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elite_size=1,
        
        # Async settings
        enable_async_evaluation=True,
        async_batch_size=5,
        max_concurrent_requests=2,
        genome_batch_size=3,
        max_concurrent_genomes=2,
        rate_limit_per_minute=50,
        detailed_performance_logging=False
    )
    
    try:
        # This should now work without AttributeError
        controller = AsyncEvolutionController(
            config=config_obj,
            seed_prompts=seed_texts  # Using converted text strings
        )
        
        print(f"✅ AsyncEvolutionController created successfully!")
        print(f"✅ Population size: {len(controller.population)} genomes")
        
        # Verify the genomes were created from the seed texts
        if len(controller.population) > 0:
            first_genome = controller.population.genomes[0]
            print(f"✅ First genome type: {type(first_genome).__name__}")
            print(f"✅ First genome text: \"{first_genome.to_text()}\"")
        
        return True
        
    except AttributeError as e:
        if "'SeedPrompt' object has no attribute 'lower'" in str(e):
            print(f"❌ Still getting SeedPrompt AttributeError: {e}")
            print("❌ The fix didn't work - SeedPrompt objects are still being passed")
            return False
        else:
            print(f"❌ Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_seed_prompt_conversion()
        if success:
            print("\n🎉 SeedPrompt to text conversion fix is working!")
            print("✅ AsyncEvolutionController can now handle SeedPrompt objects correctly")
            sys.exit(0)
        else:
            print("\n💥 SeedPrompt conversion fix failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
