#!/usr/bin/env python3
"""
Comprehensive test suite for the complete evolution system.
"""

import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.utils.dataset import gsm8k_dataset
from src.embeddings.vocabulary import vocabulary
from src.genetics.genome import PromptGenome
from src.genetics.population import Population
from src.genetics.selection import SelectionStrategy, SelectionMethod
from src.genetics.crossover import CrossoverType
from src.genetics.mutation import MutationType
from src.genetics.convergence import ConvergenceDetector
from src.genetics.evolution import EvolutionController, EvolutionConfig
from src.genetics.checkpoint import CheckpointManager
from src.evaluation.pipeline import EvaluationPipeline


def test_selection_integration():
    """Test selection strategies integration."""
    print("🎯 Testing Selection Integration")
    print("-" * 40)
    
    # Create test population with varied fitness
    population = Population(10)
    population.initialize_random(5, 15)
    
    # Set fitness values
    for i, genome in enumerate(population.genomes):
        genome.set_fitness(0.1 + i * 0.1)
    
    # Test different selection methods
    methods = [SelectionMethod.TOURNAMENT, SelectionMethod.ROULETTE_WHEEL, SelectionMethod.RANK_BASED]
    
    for method in methods:
        strategy = SelectionStrategy(method=method)
        selected = strategy.select(population)
        print(f"✅ {method.value}: selected fitness={selected.fitness:.2f}")
    
    return True


def test_convergence_integration():
    """Test convergence detection integration."""
    print("\n📈 Testing Convergence Integration")
    print("-" * 40)

    import random

    # Create population
    population = Population(5)
    population.initialize_random(5, 10)
    
    # Create convergence detector
    detector = ConvergenceDetector(
        fitness_plateau_generations=3,
        max_generations=10,
        target_fitness=0.9
    )
    
    # Simulate evolution with plateau
    for generation in range(8):
        # Set plateau fitness after generation 3
        if generation < 3:
            base_fitness = 0.5 + generation * 0.1
        else:
            base_fitness = 0.8  # Plateau
        
        for genome in population.genomes:
            genome.set_fitness(base_fitness + random.uniform(-0.05, 0.05))
        
        status = detector.update(population)
        
        if status.converged:
            print(f"✅ Converged at generation {generation + 1}: {status.reason.value}")
            break
    
    return True


def test_checkpoint_integration():
    """Test checkpoint system integration."""
    print("\n💾 Testing Checkpoint Integration")
    print("-" * 40)
    
    # Create checkpoint manager
    manager = CheckpointManager("integration_test")
    
    # Create test population
    population = Population(3)
    population.initialize_random(5, 10)
    population.generation = 5
    
    # Set fitness
    for i, genome in enumerate(population.genomes):
        genome.set_fitness(0.6 + i * 0.1)
    
    # Create evolution config
    config = EvolutionConfig(population_size=3, max_generations=10)
    
    # Save checkpoint
    best_genome = population.get_best_genome()
    checkpoint_filename = manager.save_checkpoint(
        population, [], best_genome, config
    )
    print(f"✅ Checkpoint saved: {checkpoint_filename}")
    
    # Load checkpoint
    loaded_data = manager.load_checkpoint()
    restored_population = manager.restore_population(loaded_data)
    restored_best = manager.restore_best_genome(loaded_data)
    
    print(f"✅ Checkpoint loaded: {len(restored_population)} genomes")
    print(f"✅ Best genome restored: fitness={restored_best.fitness}")
    
    return True


def test_mock_evolution():
    """Test complete evolution system with mock evaluation."""
    print("\n🧬 Testing Mock Evolution System")
    print("-" * 40)
    
    # Create mock evaluation pipeline
    class MockEvaluationPipeline:
        def __init__(self):
            self.total_evaluations = 0
        
        def evaluate_adaptive(self, population, generation):
            import random
            results = []
            
            for genome in population:
                # Mock fitness based on prompt characteristics
                prompt_text = genome.to_text().lower()
                base_fitness = 0.3
                
                # Reward certain keywords
                if "step" in prompt_text:
                    base_fitness += 0.2
                if "solve" in prompt_text:
                    base_fitness += 0.15
                if "calculate" in prompt_text:
                    base_fitness += 0.1
                if "think" in prompt_text:
                    base_fitness += 0.1
                
                # Add some randomness
                fitness = base_fitness + random.uniform(-0.1, 0.1)
                fitness = max(0.0, min(1.0, fitness))
                
                genome.set_fitness(fitness)
                
                # Mock evaluation result
                results.append({
                    'genome_id': genome.genome_id,
                    'fitness': fitness
                })
                
                self.total_evaluations += 1
            
            return results
        
        def get_evaluation_statistics(self):
            return {'total_evaluations': self.total_evaluations}
    
    # Create evolution configuration
    evolution_config = EvolutionConfig(
        population_size=8,
        max_generations=15,
        crossover_rate=0.7,
        mutation_rate=0.3,
        elite_size=2,
        target_fitness=0.8,
        convergence_patience=5,
        save_checkpoints=False  # Disable for test
    )
    
    # Create seed prompts
    seed_prompts = [
        "Let's solve this step by step.",
        "Think about this problem carefully.",
        "Calculate the answer systematically.",
        "Work through this methodically."
    ]
    
    # Create evolution controller
    mock_pipeline = MockEvaluationPipeline()
    
    def progress_callback(result):
        print(f"  Gen {result.generation}: "
              f"best={result.best_fitness:.3f}, "
              f"mean={result.mean_fitness:.3f}, "
              f"diversity={result.diversity:.3f}")
    
    controller = EvolutionController(
        config=evolution_config,
        evaluation_pipeline=mock_pipeline,
        seed_prompts=seed_prompts,
        progress_callback=progress_callback
    )
    
    # Run evolution
    print("Running mock evolution...")
    results = controller.run_evolution()
    
    print(f"✅ Evolution completed in {results['total_generations']} generations")
    print(f"✅ Best fitness: {results['best_fitness']:.3f}")
    print(f"✅ Best prompt: {results['best_genome'][:50]}..." if results['best_genome'] else "None")
    print(f"✅ Convergence reason: {results['convergence_reason']}")
    
    return True


def test_system_components():
    """Test individual system components."""
    print("\n🔧 Testing System Components")
    print("-" * 40)
    
    # Test genome operations
    genome1 = PromptGenome.from_text("Solve this problem step by step.")
    genome2 = PromptGenome.from_text("Think about the solution carefully.")
    
    # Test crossover
    from src.genetics.crossover import crossover
    offspring1, offspring2 = crossover(genome1, genome2, CrossoverType.SINGLE_POINT)
    print(f"✅ Crossover: {offspring1.to_text()[:30]}...")
    
    # Test mutation
    from src.genetics.mutation import mutate
    mutated = mutate(genome1, MutationType.SEMANTIC, mutation_rate=0.3)
    print(f"✅ Mutation: {mutated.to_text()[:30]}...")
    
    # Test population
    population = Population(5)
    population.initialize_random(5, 15)
    print(f"✅ Population: {len(population)} genomes")
    
    return True


def run_comprehensive_test():
    """Run comprehensive evolution system test."""
    print("🧪 Comprehensive Evolution System Test Suite")
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
    
    # Add random import for tests
    import random
    random.seed(42)  # For reproducible tests
    
    # Run tests
    tests = [
        test_system_components,
        test_selection_integration,
        test_convergence_integration,
        test_checkpoint_integration,
        test_mock_evolution
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
        print("🎉 All evolution system tests passed!")
        print("\n🚀 System ready for full GSM8K evolution experiments!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        sys.exit(1)
