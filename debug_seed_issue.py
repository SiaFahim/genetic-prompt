#!/usr/bin/env python3
"""
Debug script to identify the exact SeedPrompt issue and provide the correct fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_seed_issue():
    """Debug the SeedPrompt conversion issue step by step."""
    print("🔍 DEBUGGING SEEDPROMPT ISSUE")
    print("=" * 60)
    
    try:
        # Step 1: Import required modules
        print("Step 1: Importing modules...")
        from src.seeds.seed_manager import SeedManager
        from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
        from src.utils.config import config
        from src.embeddings.vocabulary import vocabulary
        print("✅ All imports successful")
        
        # Step 2: Initialize vocabulary
        print("\nStep 2: Initializing vocabulary...")
        vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
        if vocab_file.exists():
            vocabulary.load_vocabulary(vocab_file)
            print(f"✅ Vocabulary loaded: {len(vocabulary.token_to_id)} tokens")
        else:
            print("📚 Creating basic vocabulary...")
            vocabulary._create_basic_vocabulary()
            print(f"✅ Basic vocabulary created: {len(vocabulary.token_to_id)} tokens")
        
        # Step 3: Load base seeds
        print("\nStep 3: Loading base seeds...")
        seed_manager = SeedManager()
        base_seeds = seed_manager.get_base_seeds()
        print(f"✅ Loaded {len(base_seeds)} base seeds")
        
        if base_seeds:
            first_seed = base_seeds[0]
            print(f"✅ First seed type: {type(first_seed)}")
            print(f"✅ First seed attributes: {dir(first_seed)}")
            if hasattr(first_seed, 'text'):
                print(f"✅ First seed text: '{first_seed.text}'")
            else:
                print("❌ First seed has no 'text' attribute!")
                return False
        
        # Step 4: Test the WRONG way (what user is doing)
        print("\nStep 4: Testing WRONG approach (passing SeedPrompt objects)...")
        try:
            wrong_seeds = base_seeds[:3]  # This is what's causing the error
            print(f"❌ wrong_seeds type: {type(wrong_seeds[0])}")
            print("❌ This would cause: AttributeError: 'SeedPrompt' object has no attribute 'lower'")
        except Exception as e:
            print(f"❌ Error with wrong approach: {e}")
        
        # Step 5: Test the RIGHT way (convert to text)
        print("\nStep 5: Testing CORRECT approach (converting to text)...")
        try:
            seed_texts = [seed.text for seed in base_seeds[:3]]
            print(f"✅ seed_texts type: {type(seed_texts[0])}")
            print(f"✅ seed_texts content: {seed_texts}")
            
            # Validate they are strings
            for i, text in enumerate(seed_texts):
                if not isinstance(text, str):
                    print(f"❌ seed_texts[{i}] is not a string: {type(text)}")
                    return False
                if hasattr(text, 'text'):
                    print(f"❌ seed_texts[{i}] is still a SeedPrompt object!")
                    return False
            
            print("✅ All seed_texts are proper strings")
            
        except Exception as e:
            print(f"❌ Error with correct approach: {e}")
            return False
        
        # Step 6: Test AsyncEvolutionController with correct seeds
        print("\nStep 6: Testing AsyncEvolutionController with correct seed texts...")
        
        config_obj = AsyncEvolutionConfig(
            population_size=3,
            max_generations=2,
            crossover_rate=0.8,
            mutation_rate=0.3,
            elite_size=1,
            enable_async_evaluation=True,
            async_batch_size=5,
            max_concurrent_requests=2,
            genome_batch_size=3,
            max_concurrent_genomes=2,
            rate_limit_per_minute=50,
            detailed_performance_logging=False
        )
        
        try:
            controller = AsyncEvolutionController(
                config=config_obj,
                seed_prompts=seed_texts  # Using converted strings
            )
            print(f"✅ AsyncEvolutionController created successfully!")
            print(f"✅ Population size: {len(controller.population)} genomes")
            return True
            
        except Exception as e:
            print(f"❌ AsyncEvolutionController failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_correct_code():
    """Print the exact correct code to use."""
    print("\n" + "="*60)
    print("🔧 CORRECT CODE TO USE IN JUPYTER:")
    print("="*60)
    print("""
# STEP 1: Load base seeds (these are SeedPrompt objects)
seed_manager = SeedManager()
base_seeds = seed_manager.get_base_seeds()

# STEP 2: Convert SeedPrompt objects to text strings
if base_seeds:
    seed_texts = [seed.text for seed in base_seeds[:10]]
    print(f"🔧 Converted {len(seed_texts)} SeedPrompt objects to text strings")
else:
    seed_texts = None
    print("⚠️  No base_seeds available")

# STEP 3: Create AsyncEvolutionController with TEXT STRINGS
async_controller = AsyncEvolutionController(
    config=async_config,
    seed_prompts=seed_texts  # ✅ Use text strings, NOT SeedPrompt objects
)

print("✅ AsyncEvolutionController initialized successfully!")
""")
    print("="*60)

if __name__ == "__main__":
    success = debug_seed_issue()
    print_correct_code()
    
    if success:
        print("\n🎉 Debug successful - the fix works!")
        print("📋 Copy the CORRECT CODE above into your Jupyter cell")
    else:
        print("\n💥 Debug failed - there's a deeper issue")
    
    sys.exit(0 if success else 1)
