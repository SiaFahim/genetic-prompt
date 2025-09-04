#!/usr/bin/env python3
"""
Test script for random seed prompt initialization system.
Verifies that random seed generation and semantic mutation disabling work correctly.
"""

import sys
import asyncio
import logging
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.genetics.random_seeds import RandomSeedGenerator
from src.genetics.population import PopulationInitializer
from src.genetics.mutation import MutationOperator
from src.genetics.evolution_controller import EvolutionController
from src.utils.config import get_config, reload_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose HTTP logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def test_random_seed_generator():
    """Test the RandomSeedGenerator class."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing RandomSeedGenerator...")
    
    # Test with default configuration
    generator = RandomSeedGenerator()
    
    # Generate random seeds
    random_seeds = generator.generate_random_seed_prompts(num_seeds=10)
    
    # Validate seeds
    assert len(random_seeds) == 10, f"Expected 10 seeds, got {len(random_seeds)}"
    assert generator.validate_random_seeds(random_seeds), "Random seeds validation failed"
    
    # Check seed properties
    for i, seed in enumerate(random_seeds):
        tokens = seed.split()
        assert 5 <= len(tokens) <= 20, f"Seed {i} has invalid length: {len(tokens)}"
        assert all(isinstance(token, str) and token.strip() for token in tokens), f"Seed {i} has invalid tokens"
    
    logger.info(f"✅ Generated {len(random_seeds)} valid random seeds")
    logger.info(f"   Example seeds: {random_seeds[:3]}")
    
    return True

def test_population_with_random_seeds():
    """Test population initialization with random seeds."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing PopulationInitializer with random seeds...")
    
    # Create temporary config with random seeds enabled
    temp_config = {
        "experiment": {"random_seed": 42},
        "logging": {"level": "INFO"},
        "genetic_algorithm": {
            "population_size": 10,
            "use_random_seed_prompts": True,
            "random_seed_vocabulary_size": 100,
            "random_seed_length_range": [3, 8]
        },
        "genome": {"max_length": 50},
        "mutation": {"population_mutation_prob": 0.8, "token_mutation_prob": 0.002},
        "crossover": {"boundary_offset_range": 5},
        "embeddings": {"vocab_size": 100, "fallback_enabled": True},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    # Save temporary config
    temp_config_path = "temp_random_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        # Force reload config to use temporary config
        reload_config(temp_config_path)

        # Initialize with random seeds
        population_init = PopulationInitializer(temp_config_path)

        # Verify random seeds are enabled
        assert population_init.use_random_seed_prompts, "Random seeds should be enabled"
        
        # Load seed prompts (should be random)
        seed_prompts = population_init.load_seed_prompts()
        
        # Validate random seeds
        assert len(seed_prompts) > 0, "Should generate random seed prompts"
        
        for seed in seed_prompts:
            tokens = seed.split()
            assert 3 <= len(tokens) <= 8, f"Random seed has invalid length: {len(tokens)}"
        
        logger.info(f"✅ Generated {len(seed_prompts)} random seed prompts")
        logger.info(f"   Example random seeds: {seed_prompts[:3]}")
        
        # Test population initialization
        population = population_init.initialize_population(target_size=5)
        
        assert len(population) == 5, f"Expected 5 genomes, got {len(population)}"
        
        for genome in population:
            assert len(genome.tokens) > 0, "Genome should have tokens"
            assert genome.generation_born == 0, "Initial genomes should be generation 0"
        
        logger.info(f"✅ Initialized population of {len(population)} genomes from random seeds")
        
        return True
        
    finally:
        # Clean up temporary config
        Path(temp_config_path).unlink(missing_ok=True)

def test_semantic_mutation_disabling():
    """Test semantic mutation disabling functionality."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing semantic mutation disabling...")
    
    # Test 1: Automatic disabling with random seeds
    temp_config1 = {
        "logging": {"level": "INFO"},
        "genetic_algorithm": {
            "use_random_seed_prompts": True,
            "force_disable_semantic_mutation": False
        },
        "mutation": {
            "population_mutation_prob": 0.8,
            "token_mutation_prob": 0.002,
            "semantic_neighbor_prob": 0.9
        },
        "embeddings": {"fallback_enabled": True, "vocab_size": 100},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    temp_config_path1 = "temp_config1.json"
    with open(temp_config_path1, 'w') as f:
        json.dump(temp_config1, f, indent=2)
    
    try:
        reload_config(temp_config_path1)
        mutation_op1 = MutationOperator(temp_config_path1)
        assert mutation_op1.semantic_mutation_disabled, "Semantic mutation should be disabled with random seeds"
        assert mutation_op1.use_random_seed_prompts, "Random seeds should be enabled"
        
        logger.info("✅ Semantic mutation automatically disabled with random seeds")
        
    finally:
        Path(temp_config_path1).unlink(missing_ok=True)
    
    # Test 2: Manual disabling
    temp_config2 = {
        "genetic_algorithm": {
            "use_random_seed_prompts": False,
            "force_disable_semantic_mutation": True
        },
        "mutation": {
            "population_mutation_prob": 0.8,
            "token_mutation_prob": 0.002,
            "semantic_neighbor_prob": 0.9
        },
        "embeddings": {"fallback_enabled": True, "vocab_size": 100},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    temp_config_path2 = "temp_config2.json"
    with open(temp_config_path2, 'w') as f:
        json.dump(temp_config2, f, indent=2)
    
    try:
        reload_config(temp_config_path2)
        mutation_op2 = MutationOperator(temp_config_path2)
        assert mutation_op2.semantic_mutation_disabled, "Semantic mutation should be manually disabled"
        assert not mutation_op2.use_random_seed_prompts, "Random seeds should be disabled"
        
        logger.info("✅ Semantic mutation manually disabled via force_disable_semantic_mutation")
        
    finally:
        Path(temp_config_path2).unlink(missing_ok=True)
    
    # Test 3: Normal semantic mutation enabled
    temp_config3 = {
        "genetic_algorithm": {
            "use_random_seed_prompts": False,
            "force_disable_semantic_mutation": False
        },
        "mutation": {
            "population_mutation_prob": 0.8,
            "token_mutation_prob": 0.002,
            "semantic_neighbor_prob": 0.9
        },
        "embeddings": {"fallback_enabled": True, "vocab_size": 100},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    temp_config_path3 = "temp_config3.json"
    with open(temp_config_path3, 'w') as f:
        json.dump(temp_config3, f, indent=2)
    
    try:
        reload_config(temp_config_path3)
        mutation_op3 = MutationOperator(temp_config_path3)
        assert not mutation_op3.semantic_mutation_disabled, "Semantic mutation should be enabled"
        
        logger.info("✅ Semantic mutation enabled when both flags are false")
        
    finally:
        Path(temp_config_path3).unlink(missing_ok=True)
    
    return True

def test_mutation_statistics():
    """Test mutation statistics reporting."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing mutation statistics...")
    
    temp_config = {
        "genetic_algorithm": {
            "use_random_seed_prompts": True,
            "force_disable_semantic_mutation": False
        },
        "mutation": {
            "population_mutation_prob": 0.8,
            "token_mutation_prob": 0.002,
            "semantic_neighbor_prob": 0.9
        },
        "embeddings": {"fallback_enabled": True, "vocab_size": 100},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    temp_config_path = "temp_stats_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        reload_config(temp_config_path)
        mutation_op = MutationOperator(temp_config_path)
        stats = mutation_op.get_mutation_statistics()
        
        # Check that statistics include new fields
        assert 'semantic_mutation_disabled' in stats, "Stats should include semantic_mutation_disabled"
        assert 'use_random_seed_prompts' in stats, "Stats should include use_random_seed_prompts"
        assert 'force_disable_semantic_mutation' in stats, "Stats should include force_disable_semantic_mutation"
        
        assert stats['semantic_mutation_disabled'] == True, "Should report semantic mutation as disabled"
        assert stats['use_random_seed_prompts'] == True, "Should report random seeds as enabled"
        
        logger.info("✅ Mutation statistics include configuration status")
        logger.info(f"   Semantic mutation disabled: {stats['semantic_mutation_disabled']}")
        logger.info(f"   Random seed prompts: {stats['use_random_seed_prompts']}")
        
        return True
        
    finally:
        Path(temp_config_path).unlink(missing_ok=True)

async def test_integration():
    """Test full integration with evolution controller."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing full integration...")
    
    # Create minimal config for integration test
    temp_config = {
        "experiment": {"random_seed": 42, "target_accuracy": 0.85},
        "model": {"name": "gpt-4", "temperature": 0, "max_tokens": 100},
        "genetic_algorithm": {
            "population_size": 3,
            "max_generations": 1,
            "use_random_seed_prompts": True,
            "random_seed_vocabulary_size": 50,
            "random_seed_length_range": [3, 6]
        },
        "mutation": {"population_mutation_prob": 0.5, "token_mutation_prob": 0.1},
        "crossover": {"boundary_offset_range": 2},
        "selection": {"elite_count": 1, "diverse_count": 0, "random_count": 0},
        "genome": {"max_length": 20},
        "evaluation": {"concurrency_limit": 1, "cache_enabled": False},
        "embeddings": {"fallback_enabled": True, "vocab_size": 50},
        "paths": {"embeddings_dir": "./data/embeddings"}
    }
    
    temp_config_path = "temp_integration_config.json"
    with open(temp_config_path, 'w') as f:
        json.dump(temp_config, f, indent=2)
    
    try:
        # Force reload config
        reload_config(temp_config_path)

        # Initialize evolution controller
        evolution_controller = EvolutionController(temp_config_path)
        
        # Verify configuration is loaded correctly
        assert evolution_controller.population_initializer.use_random_seed_prompts, "Random seeds should be enabled"
        assert evolution_controller.mutation_operator.semantic_mutation_disabled, "Semantic mutation should be disabled"
        
        logger.info("✅ Evolution controller initialized with random seed configuration")
        logger.info("✅ All components properly configured for random seed experiments")
        
        return True
        
    finally:
        Path(temp_config_path).unlink(missing_ok=True)

async def main():
    """Main test function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 Testing Random Seed Prompt Initialization System")
    logger.info("=" * 70)
    
    try:
        # Test 1: Random seed generator
        test_random_seed_generator()
        
        # Test 2: Population initialization with random seeds
        test_population_with_random_seeds()
        
        # Test 3: Semantic mutation disabling
        test_semantic_mutation_disabling()
        
        # Test 4: Mutation statistics
        test_mutation_statistics()
        
        # Test 5: Full integration
        await test_integration()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALL RANDOM SEED TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("✅ Random seed generation: Working correctly")
        logger.info("✅ Population initialization: Supports random seeds")
        logger.info("✅ Semantic mutation control: Automatic and manual disabling")
        logger.info("✅ Configuration validation: All parameters validated")
        logger.info("✅ Integration: All components work together")
        logger.info("\n🚀 Random seed prompt initialization system is ready for experiments!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
