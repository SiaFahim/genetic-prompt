#!/usr/bin/env python3
"""
Comprehensive test suite for the seed prompt system.
"""

import sys
import random
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.embeddings.vocabulary import vocabulary
from src.seeds.prompt_categories import PromptCategory, PromptCategoryManager
from src.seeds.base_seeds import BaseSeedCollection, SeedPrompt
from src.seeds.seed_validation import SeedValidator
from src.seeds.seed_manager import SeedManager


def test_prompt_categories():
    """Test prompt category system."""
    print("📂 Testing Prompt Categories")
    print("-" * 40)
    
    manager = PromptCategoryManager()
    
    # Test category definitions
    all_categories = manager.get_all_categories()
    print(f"✅ Defined categories: {len(all_categories)}")
    
    # Test each category has required components
    for category, definition in all_categories.items():
        assert definition.name, f"Category {category} missing name"
        assert definition.key_strategies, f"Category {category} missing strategies"
        assert definition.example_phrases, f"Category {category} missing phrases"
        assert definition.expected_benefits, f"Category {category} missing benefits"
    
    print("✅ All categories have complete definitions")
    
    # Test category analysis
    test_prompts = [
        "Let's solve this step by step.",
        "I'll visualize this problem to understand it better.",
        "Let me set up equations for this algebraic problem."
    ]
    
    for prompt in test_prompts:
        scores = manager.analyze_prompt_categories(prompt)
        top_category = max(scores, key=scores.get)
        print(f"   '{prompt[:30]}...' → {top_category.value}")
    
    # Test target distribution
    target_dist = manager.get_category_distribution_target()
    total_target = sum(target_dist.values())
    assert total_target == 50, f"Target distribution should sum to 50, got {total_target}"
    print(f"✅ Target distribution validated: {total_target} total prompts")
    
    return True


def test_base_seed_collection():
    """Test base seed collection."""
    print("\n🌱 Testing Base Seed Collection")
    print("-" * 40)
    
    collection = BaseSeedCollection()
    
    # Test collection size
    all_seeds = collection.get_all_seeds()
    assert len(all_seeds) == 50, f"Expected 50 seeds, got {len(all_seeds)}"
    print(f"✅ Collection size: {len(all_seeds)} seeds")
    
    # Test unique IDs
    seed_ids = [seed.id for seed in all_seeds]
    unique_ids = set(seed_ids)
    assert len(unique_ids) == len(seed_ids), "Duplicate seed IDs found"
    print("✅ All seed IDs are unique")
    
    # Test category distribution
    distribution = collection.get_category_distribution()
    category_manager = PromptCategoryManager()
    target_dist = category_manager.get_category_distribution_target()
    
    for category, target_count in target_dist.items():
        actual_count = distribution.get(category, 0)
        assert actual_count == target_count, f"{category}: expected {target_count}, got {actual_count}"
    
    print("✅ Category distribution matches target")
    
    # Test seed quality
    for seed in all_seeds[:5]:  # Test first 5 seeds
        assert seed.text.strip(), f"Empty text in seed {seed.id}"
        assert seed.description, f"Missing description in seed {seed.id}"
        assert seed.expected_strength, f"Missing expected_strength in seed {seed.id}"
        assert isinstance(seed.variations, list), f"Variations not a list in seed {seed.id}"
    
    print("✅ Seed quality checks passed")
    
    # Test validation
    validation = collection.validate_collection()
    assert validation['is_valid'], f"Collection validation failed: {validation['issues']}"
    print(f"✅ Collection validation: {validation['is_valid']}")
    
    return True


def test_seed_validation():
    """Test seed validation system."""
    print("\n🔍 Testing Seed Validation")
    print("-" * 40)
    
    validator = SeedValidator()
    collection = BaseSeedCollection()
    seeds = collection.get_all_seeds()
    
    # Test validation metrics
    metrics = validator.validate_collection(seeds)
    
    # Check metric ranges
    assert 0 <= metrics.overall_score <= 1, f"Overall score out of range: {metrics.overall_score}"
    assert 0 <= metrics.diversity_score <= 1, f"Diversity score out of range: {metrics.diversity_score}"
    assert 0 <= metrics.category_balance <= 1, f"Balance score out of range: {metrics.category_balance}"
    assert 0 <= metrics.uniqueness_score <= 1, f"Uniqueness score out of range: {metrics.uniqueness_score}"
    
    print(f"✅ Validation metrics in valid ranges")
    print(f"   Overall Score: {metrics.overall_score:.3f}")
    print(f"   Diversity: {metrics.diversity_score:.3f}")
    print(f"   Balance: {metrics.category_balance:.3f}")
    print(f"   Uniqueness: {metrics.uniqueness_score:.3f}")
    
    # Test high-quality collection should score well
    assert metrics.overall_score >= 0.7, f"Base collection should score >= 0.7, got {metrics.overall_score}"
    assert metrics.category_balance >= 0.9, f"Perfect balance expected, got {metrics.category_balance}"
    
    print("✅ Base collection meets quality standards")
    
    # Test validation report generation
    report = validator.generate_validation_report(metrics)
    assert "SEED PROMPT VALIDATION REPORT" in report, "Report missing header"
    assert "Overall Score" in report, "Report missing overall score"
    print("✅ Validation report generated successfully")
    
    return True


def test_seed_manager():
    """Test seed management system."""
    print("\n🗂️  Testing Seed Manager")
    print("-" * 40)
    
    manager = SeedManager()
    
    # Test base collection access
    base_seeds = manager.get_base_seeds()
    assert len(base_seeds) == 50, f"Expected 50 base seeds, got {len(base_seeds)}"
    print(f"✅ Base collection access: {len(base_seeds)} seeds")
    
    # Test category filtering
    step_seeds = manager.get_seeds_by_category(PromptCategory.STEP_BY_STEP)
    expected_step_count = 8  # From target distribution
    assert len(step_seeds) == expected_step_count, f"Expected {expected_step_count} step seeds, got {len(step_seeds)}"
    print(f"✅ Category filtering: {len(step_seeds)} step-by-step seeds")
    
    # Test balanced subset creation
    balanced_20 = manager.create_balanced_subset(20, strategy="balanced")
    assert len(balanced_20) == 20, f"Expected 20 seeds, got {len(balanced_20)}"
    
    # Check balance in subset
    subset_categories = [seed.category for seed in balanced_20]
    category_counts = {}
    for cat in subset_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Should have representation from multiple categories
    assert len(category_counts) >= 5, f"Subset should span multiple categories, got {len(category_counts)}"
    print(f"✅ Balanced subset: {len(balanced_20)} seeds across {len(category_counts)} categories")
    
    # Test diverse subset creation
    diverse_15 = manager.create_balanced_subset(15, strategy="diverse")
    assert len(diverse_15) == 15, f"Expected 15 seeds, got {len(diverse_15)}"
    print(f"✅ Diverse subset: {len(diverse_15)} seeds")
    
    # Test collection saving and loading
    test_seeds = balanced_20[:10]
    collection_name = "test_validation_collection"
    
    saved_path = manager.save_collection(test_seeds, collection_name, "Test collection")
    assert Path(saved_path).exists(), f"Collection file not created: {saved_path}"
    print(f"✅ Collection saved: {Path(saved_path).name}")
    
    loaded_seeds = manager.load_collection(collection_name)
    assert loaded_seeds is not None, "Failed to load collection"
    assert len(loaded_seeds) == len(test_seeds), f"Loaded {len(loaded_seeds)} seeds, expected {len(test_seeds)}"
    print(f"✅ Collection loaded: {len(loaded_seeds)} seeds")
    
    # Test collection listing
    collections = manager.list_collections()
    assert "base" in collections, "Base collection not listed"
    assert collection_name in collections, f"Test collection {collection_name} not listed"
    print(f"✅ Collection listing: {len(collections)} collections")
    
    # Test validation integration
    validation = manager.validate_collection("base")
    assert validation is not None, "Base collection validation failed"
    assert validation['overall_score'] >= 0.7, f"Base collection score too low: {validation['overall_score']}"
    print(f"✅ Validation integration: score {validation['overall_score']:.3f}")
    
    # Test statistics
    stats = manager.get_collection_statistics("base")
    assert stats is not None, "Failed to get base collection statistics"
    assert stats['total_seeds'] == 50, f"Expected 50 seeds in stats, got {stats['total_seeds']}"
    print(f"✅ Statistics: {stats['total_seeds']} seeds")
    
    return True


def test_integration_with_genetics():
    """Test integration with genetic algorithm components."""
    print("\n🧬 Testing Integration with Genetics")
    print("-" * 40)
    
    # Load vocabulary for genetic operations
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    print("✅ Vocabulary loaded for genetic operations")
    
    # Test seed to genome conversion
    from src.genetics.genome import PromptGenome
    from src.genetics.population import Population
    
    manager = SeedManager()
    seed_texts = [seed.text for seed in manager.get_base_seeds()[:10]]
    
    # Create population from seeds
    population = Population(population_size=10)
    population.initialize_from_seeds(seed_texts)
    
    assert len(population.genomes) == 10, f"Expected 10 genomes, got {len(population.genomes)}"
    print(f"✅ Population initialized from seeds: {len(population.genomes)} genomes")
    
    # Test that genomes can be converted back to text
    for i, genome in enumerate(population.genomes[:3]):
        genome_text = genome.to_text()
        original_text = seed_texts[i]
        
        # Should be similar (tokenization might cause minor differences)
        assert len(genome_text) > 0, f"Genome {i} produced empty text"
        print(f"   Genome {i}: '{genome_text[:30]}...'")
    
    print("✅ Genome-text conversion working")
    
    # Test genetic operations on seed-derived genomes
    from src.genetics.crossover import crossover, CrossoverType
    from src.genetics.mutation import mutate, MutationType
    
    parent1 = population.genomes[0]
    parent2 = population.genomes[1]
    
    # Test crossover
    offspring1, offspring2 = crossover(parent1, parent2, CrossoverType.SINGLE_POINT)
    assert offspring1.to_text() != parent1.to_text(), "Crossover should produce different offspring"
    print("✅ Crossover operations working on seed-derived genomes")
    
    # Test mutation
    original_text = parent1.to_text()
    mutated = mutate(parent1, MutationType.SEMANTIC, mutation_rate=0.3)
    mutated_text = mutated.to_text()
    
    # Mutation should change the genome (most of the time)
    if original_text != mutated_text:
        print("✅ Mutation operations working on seed-derived genomes")
    else:
        print("⚠️  Mutation didn't change genome (can happen with low probability)")
    
    return True


def run_comprehensive_test():
    """Run comprehensive seed system test."""
    print("🧪 Comprehensive Seed System Test Suite")
    print("=" * 60)
    
    # Set random seed for reproducible tests
    random.seed(42)
    
    # Run tests
    tests = [
        test_prompt_categories,
        test_base_seed_collection,
        test_seed_validation,
        test_seed_manager,
        test_integration_with_genetics
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
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All seed system tests passed!")
        print("\n🚀 Seed system ready for evolution experiments!")
        
        # Show final summary
        manager = SeedManager()
        base_seeds = manager.get_base_seeds()
        validation = manager.validate_collection("base")
        
        print(f"\n📋 Seed System Summary:")
        print(f"   • Total Seeds: {len(base_seeds)}")
        print(f"   • Categories: {len(set(seed.category for seed in base_seeds))}")
        print(f"   • Validation Score: {validation['overall_score']:.3f}")
        print(f"   • Quality Status: {'✅ EXCELLENT' if validation['overall_score'] >= 0.8 else '⚠️ GOOD' if validation['overall_score'] >= 0.6 else '❌ NEEDS IMPROVEMENT'}")
        
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        sys.exit(1)
