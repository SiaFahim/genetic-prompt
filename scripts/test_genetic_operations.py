#!/usr/bin/env python3
"""
Comprehensive test suite for genetic algorithm operations.
"""

import sys
import random
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.embeddings.vocabulary import vocabulary
from src.embeddings.neighborhoods import semantic_neighborhoods
from src.genetics.genome import PromptGenome, create_random_genome
from src.genetics.crossover import crossover, CrossoverType
from src.genetics.mutation import mutate, MutationType
from src.genetics.population import Population


def test_genome_operations():
    """Test basic genome operations."""
    print("🧬 Testing Genome Operations")
    print("-" * 40)
    
    # Test genome creation
    text_genome = PromptGenome.from_text("Solve this step by step.")
    token_genome = PromptGenome.from_tokens([1, 2, 3, 4, 5])
    random_genome = create_random_genome(5, 10)
    
    print(f"✅ Text genome: {text_genome}")
    print(f"✅ Token genome: {token_genome}")
    print(f"✅ Random genome: {random_genome}")
    
    # Test copying and equality
    copied_genome = text_genome.copy()
    print(f"✅ Copied genome: {copied_genome}")
    print(f"✅ Are equal: {text_genome == copied_genome}")
    print(f"✅ Different IDs: {text_genome.genome_id != copied_genome.genome_id}")
    
    # Test diversity calculation
    diversity = text_genome.get_diversity_score(token_genome)
    print(f"✅ Diversity score: {diversity:.3f}")
    
    # Test fitness setting
    text_genome.set_fitness(0.85)
    print(f"✅ Fitness set: {text_genome.fitness}")
    
    # Test validation
    print(f"✅ Genome valid: {text_genome.validate()}")
    
    return True


def test_crossover_operations():
    """Test crossover operations."""
    print("\n🔀 Testing Crossover Operations")
    print("-" * 40)
    
    parent1 = PromptGenome.from_text("Let's solve this problem step by step.")
    parent2 = PromptGenome.from_text("Think carefully about the solution.")
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    crossover_types = [
        CrossoverType.SINGLE_POINT,
        CrossoverType.TWO_POINT,
        CrossoverType.UNIFORM,
        CrossoverType.SEMANTIC_BLEND
    ]
    
    for crossover_type in crossover_types:
        offspring1, offspring2 = crossover(parent1, parent2, crossover_type)
        print(f"✅ {crossover_type.value}: {offspring1.to_text()[:50]}...")
        print(f"   {' ' * len(crossover_type.value)}: {offspring2.to_text()[:50]}...")
        
        # Verify genealogy
        assert offspring1.parent_ids == [parent1.genome_id, parent2.genome_id]
        assert offspring2.parent_ids == [parent1.genome_id, parent2.genome_id]
        assert offspring1.generation == 1
        assert offspring2.generation == 1
    
    return True


def test_mutation_operations():
    """Test mutation operations."""
    print("\n🧬 Testing Mutation Operations")
    print("-" * 40)
    
    original = PromptGenome.from_text("Let's solve this problem step by step carefully.")
    print(f"Original: {original}")
    
    mutation_types = [
        (MutationType.SEMANTIC, {'mutation_rate': 0.3}),
        (MutationType.RANDOM, {'mutation_rate': 0.3}),
        (MutationType.INSERTION, {'insertion_rate': 0.2}),
        (MutationType.DELETION, {'deletion_rate': 0.2}),
        (MutationType.SWAP, {'swap_rate': 0.3}),
        (MutationType.DUPLICATION, {'duplication_rate': 0.5})
    ]
    
    for mutation_type, kwargs in mutation_types:
        mutated = mutate(original, mutation_type, **kwargs)
        print(f"✅ {mutation_type.value}: {mutated.to_text()[:60]}...")
        
        # Verify mutation history
        if mutated.mutation_history:
            print(f"   Mutations made: {len(mutated.mutation_history[-1]['mutations'])}")
    
    return True


def test_population_management():
    """Test population management."""
    print("\n👥 Testing Population Management")
    print("-" * 40)
    
    # Test random initialization
    population = Population(population_size=20)
    population.initialize_random(5, 15)
    print(f"✅ Random population: {len(population)} genomes")
    
    # Test seed initialization
    seed_prompts = [
        "Let's solve this step by step.",
        "Think about the problem carefully.",
        "Calculate the answer systematically.",
        "Work through this methodically."
    ]
    
    seed_population = Population(population_size=10)
    seed_population.initialize_from_seeds(seed_prompts)
    print(f"✅ Seed population: {len(seed_population)} genomes")
    
    # Set random fitnesses
    for genome in population:
        genome.set_fitness(random.uniform(0.0, 1.0))
    
    # Test statistics
    stats = population.get_population_statistics()
    print(f"✅ Population stats: diversity={stats['diversity']:.3f}, "
          f"mean_fitness={stats['fitness_stats']['mean']:.3f}")
    
    # Test selection
    selected = population.tournament_selection(tournament_size=5)
    print(f"✅ Tournament selection: fitness={selected.fitness:.3f}")
    
    # Test evolution
    initial_diversity = population.calculate_diversity()
    population.evolve_generation(crossover_rate=0.8, mutation_rate=0.3)
    final_diversity = population.calculate_diversity()
    
    print(f"✅ Evolution: gen={population.generation}, "
          f"diversity: {initial_diversity:.3f} -> {final_diversity:.3f}")
    
    return True


def test_integration():
    """Test integration of all components."""
    print("\n🔧 Testing Integration")
    print("-" * 40)
    
    # Create a small population
    population = Population(population_size=8)
    
    # Initialize with diverse prompts
    seed_prompts = [
        "Solve this problem step by step.",
        "Let's think about this carefully.",
        "Calculate the answer systematically.",
        "Work through this methodically."
    ]
    population.initialize_from_seeds(seed_prompts)
    
    # Simulate evolution over several generations
    print("Simulating evolution...")
    
    for generation in range(5):
        # Assign random fitnesses (simulating evaluation)
        for genome in population:
            # Simulate fitness based on length and some randomness
            base_fitness = 0.5 + 0.3 * (genome.length() / 20.0)
            noise = random.uniform(-0.2, 0.2)
            fitness = max(0.0, min(1.0, base_fitness + noise))
            genome.set_fitness(fitness)
        
        # Evolve
        population.evolve_generation(
            crossover_rate=0.7,
            mutation_rate=0.2,
            elite_size=2
        )
        
        # Get statistics
        stats = population.get_population_statistics()
        best_genome = population.get_best_genome()
        
        fitness_mean = stats['fitness_stats'].get('mean', 0.0) if stats['fitness_stats']['count'] > 0 else 0.0
        print(f"  Gen {generation + 1}: "
              f"best={stats['best_fitness']:.3f}, "
              f"mean={fitness_mean:.3f}, "
              f"diversity={stats['diversity']:.3f}")
        
        if best_genome:
            print(f"    Best: {best_genome.to_text()[:50]}...")
    
    print("✅ Integration test completed successfully!")
    return True


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n⚠️  Testing Edge Cases")
    print("-" * 40)
    
    # Test empty genomes
    empty_genome = PromptGenome.from_tokens([])
    print(f"✅ Empty genome: {empty_genome}")
    
    # Test crossover with empty genome
    normal_genome = PromptGenome.from_text("Hello world")
    offspring1, offspring2 = crossover(empty_genome, normal_genome)
    print(f"✅ Crossover with empty: {offspring1}, {offspring2}")
    
    # Test mutation on empty genome
    mutated_empty = mutate(empty_genome, MutationType.INSERTION)
    print(f"✅ Mutated empty: {mutated_empty}")
    
    # Test very short genomes
    short_genome = PromptGenome.from_tokens([1])
    mutated_short = mutate(short_genome, MutationType.SEMANTIC)
    print(f"✅ Mutated short: {mutated_short}")
    
    # Test population with no fitness values
    pop = Population(5)
    pop.initialize_random(3, 8)
    best = pop.get_best_genome()
    print(f"✅ Best with no fitness: {best}")
    
    print("✅ Edge cases handled correctly!")
    return True


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🧪 Comprehensive Genetic Operations Test Suite")
    print("=" * 60)
    
    # Load required components
    print("Loading components...")
    
    # Load vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print("✅ Vocabulary loaded")
    else:
        print("⚠️  Vocabulary not found, creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
    
    # Load neighborhoods
    neighborhoods_file = config.get_data_dir() / "embeddings" / "semantic_neighborhoods.pkl"
    if neighborhoods_file.exists():
        semantic_neighborhoods.load_neighborhoods(neighborhoods_file)
        semantic_neighborhoods.vocabulary = vocabulary
        print("✅ Semantic neighborhoods loaded")
    else:
        print("⚠️  Neighborhoods not found, mutations will be random")
    
    # Run tests
    tests = [
        test_genome_operations,
        test_crossover_operations,
        test_mutation_operations,
        test_population_management,
        test_integration,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All genetic operations tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        sys.exit(1)
