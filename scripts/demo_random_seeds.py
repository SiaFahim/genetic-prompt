#!/usr/bin/env python3
"""
Demo script to showcase the random seed prompt initialization system.
Shows the difference between semantic and random seed initialization.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.genetics.random_seeds import RandomSeedGenerator
from src.genetics.population import PopulationInitializer
from src.genetics.mutation import MutationOperator
from src.utils.config import get_config

def setup_logging():
    """Set up clean logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def demo_random_seed_generation():
    """Demonstrate random seed generation."""
    logger = logging.getLogger(__name__)
    
    print("🎲 RANDOM SEED GENERATION DEMO")
    print("=" * 50)
    
    # Load current config
    config = get_config()
    
    # Show configuration
    use_random = config.get('genetic_algorithm.use_random_seed_prompts', False)
    vocab_size = config.get('genetic_algorithm.random_seed_vocabulary_size', 1000)
    length_range = config.get('genetic_algorithm.random_seed_length_range', [5, 20])
    
    print(f"📋 Configuration:")
    print(f"   use_random_seed_prompts: {use_random}")
    print(f"   vocabulary_size: {vocab_size}")
    print(f"   length_range: {length_range}")
    
    if use_random:
        print(f"\n🎯 Generating random seed prompts...")
        
        # Generate random seeds
        generator = RandomSeedGenerator()
        random_seeds = generator.generate_random_seed_prompts(num_seeds=10)
        
        print(f"\n✅ Generated {len(random_seeds)} random seed prompts:")
        for i, seed in enumerate(random_seeds[:5], 1):
            tokens = seed.split()
            print(f"   {i}. [{len(tokens)} tokens] {seed}")
        
        if len(random_seeds) > 5:
            print(f"   ... and {len(random_seeds) - 5} more")
        
        # Show token statistics
        all_tokens = []
        for seed in random_seeds:
            all_tokens.extend(seed.split())
        
        unique_tokens = set(all_tokens)
        avg_length = sum(len(seed.split()) for seed in random_seeds) / len(random_seeds)
        
        print(f"\n📊 Statistics:")
        print(f"   Total tokens: {len(all_tokens)}")
        print(f"   Unique tokens: {len(unique_tokens)}")
        print(f"   Average length: {avg_length:.1f} tokens")
        print(f"   Token diversity: {len(unique_tokens)/len(all_tokens):.1%}")
        
    else:
        print(f"\n📝 Random seeds are DISABLED - would use semantic seed prompts")
        
        # Show what semantic seeds would look like
        semantic_seeds = [
            "Let's solve this step by step.",
            "First, I need to identify what the problem is asking.",
            "I'll work through this systematically.",
            "Let me break down the given information.",
            "To find the answer, I should organize the data."
        ]
        
        print(f"\n📝 Example semantic seed prompts:")
        for i, seed in enumerate(semantic_seeds, 1):
            tokens = seed.split()
            print(f"   {i}. [{len(tokens)} tokens] {seed}")

def demo_semantic_mutation_control():
    """Demonstrate semantic mutation control."""
    print(f"\n🧠 SEMANTIC MUTATION CONTROL DEMO")
    print("=" * 50)
    
    # Load current config
    config = get_config()
    
    # Initialize mutation operator
    mutation_op = MutationOperator()
    
    # Get mutation statistics
    stats = mutation_op.get_mutation_statistics()
    
    print(f"📋 Mutation Configuration:")
    print(f"   use_random_seed_prompts: {stats['use_random_seed_prompts']}")
    print(f"   force_disable_semantic_mutation: {stats['force_disable_semantic_mutation']}")
    print(f"   semantic_mutation_disabled: {stats['semantic_mutation_disabled']}")
    
    if stats['semantic_mutation_disabled']:
        print(f"\n🚫 Semantic mutation is DISABLED")
        if stats['use_random_seed_prompts']:
            print(f"   Reason: Random seed prompts enabled (automatic)")
        if stats['force_disable_semantic_mutation']:
            print(f"   Reason: Manual override (force_disable_semantic_mutation=true)")
        print(f"   Effect: All mutations will be purely random token replacements")
    else:
        semantic_prob = config.get('mutation.semantic_neighbor_prob', 0.9)
        print(f"\n✅ Semantic mutation is ENABLED")
        print(f"   Semantic probability: {semantic_prob}")
        print(f"   Effect: {semantic_prob:.0%} of mutations use semantic neighborhoods")

def demo_population_initialization():
    """Demonstrate population initialization with current config."""
    print(f"\n👥 POPULATION INITIALIZATION DEMO")
    print("=" * 50)
    
    # Initialize population
    pop_init = PopulationInitializer()
    
    print(f"📋 Population Configuration:")
    print(f"   use_random_seed_prompts: {pop_init.use_random_seed_prompts}")
    print(f"   population_size: {pop_init.population_size}")
    
    # Load seed prompts
    print(f"\n🌱 Loading seed prompts...")
    seed_prompts = pop_init.load_seed_prompts()
    
    print(f"✅ Loaded {len(seed_prompts)} seed prompts")
    
    if pop_init.use_random_seed_prompts:
        print(f"🎲 Using RANDOM seed prompts:")
    else:
        print(f"📝 Using SEMANTIC seed prompts:")
    
    # Show first few seeds
    for i, seed in enumerate(seed_prompts[:3], 1):
        tokens = seed.split()
        print(f"   {i}. [{len(tokens)} tokens] {seed}")
    
    if len(seed_prompts) > 3:
        print(f"   ... and {len(seed_prompts) - 3} more")
    
    # Initialize a small population for demo
    print(f"\n🧬 Initializing sample population (5 genomes)...")
    population = pop_init.initialize_population(target_size=5)
    
    print(f"✅ Created {len(population)} genomes:")
    for i, genome in enumerate(population, 1):
        print(f"   {i}. [{len(genome.tokens)} tokens] {' '.join(genome.tokens[:8])}{'...' if len(genome.tokens) > 8 else ''}")

def main():
    """Main demo function."""
    setup_logging()
    
    print("🔬 RANDOM SEED PROMPT SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the configurable random seed initialization system")
    print("and how it affects genetic algorithm behavior.\n")
    
    try:
        # Demo 1: Random seed generation
        demo_random_seed_generation()
        
        # Demo 2: Semantic mutation control
        demo_semantic_mutation_control()
        
        # Demo 3: Population initialization
        demo_population_initialization()
        
        print(f"\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Load current config for summary
        config = get_config()
        use_random = config.get('genetic_algorithm.use_random_seed_prompts', False)
        
        if use_random:
            print("🎲 CURRENT MODE: Random Seed Initialization")
            print("   • Seed prompts: Generated from random vocabulary tokens")
            print("   • Semantic mutation: Automatically disabled")
            print("   • Evolution: Pure randomness → convergence testing")
        else:
            print("📝 CURRENT MODE: Semantic Seed Initialization")
            print("   • Seed prompts: Meaningful problem-solving text")
            print("   • Semantic mutation: Enabled with neighborhood search")
            print("   • Evolution: Semantic-aware prompt optimization")
        
        print(f"\n💡 To switch modes, edit configs/experiment_config.json:")
        print(f"   Set 'use_random_seed_prompts' to {not use_random}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
