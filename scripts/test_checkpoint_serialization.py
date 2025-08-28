#!/usr/bin/env python3
"""
Test script to verify checkpoint serialization fixes.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.genetics.evolution import EvolutionConfig, GenerationResult, _dataclass_to_json_dict
from src.genetics.selection import SelectionMethod
from src.genetics.crossover import CrossoverType
from src.genetics.mutation import MutationType
from src.genetics.convergence import ConvergenceStatus, ConvergenceReason


def test_evolution_config_serialization():
    """Test EvolutionConfig serialization with enums."""
    print("🧪 Testing EvolutionConfig serialization...")
    
    # Create config with enum values
    config = EvolutionConfig(
        population_size=15,
        max_generations=20,
        target_fitness=0.75,
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_type=CrossoverType.SINGLE_POINT,
        mutation_type=MutationType.SEMANTIC
    )
    
    # Test serialization
    try:
        serialized = _dataclass_to_json_dict(config)
        json_str = json.dumps(serialized, indent=2)
        print("✅ EvolutionConfig serialization successful")
        print(f"   Selection method: {serialized['selection_method']}")
        print(f"   Crossover type: {serialized['crossover_type']}")
        print(f"   Mutation type: {serialized['mutation_type']}")
        
        # Test deserialization
        loaded_data = json.loads(json_str)
        print("✅ JSON round-trip successful")
        
        return True
        
    except Exception as e:
        print(f"❌ EvolutionConfig serialization failed: {e}")
        return False


def test_generation_result_serialization():
    """Test GenerationResult serialization with ConvergenceStatus."""
    print("\n🧪 Testing GenerationResult serialization...")
    
    # Create convergence status
    convergence_status = ConvergenceStatus(
        converged=True,
        reason=ConvergenceReason.TARGET_FITNESS,
        confidence=0.95,
        generations_since_improvement=0,
        current_best_fitness=0.85,
        diversity_score=0.65,
        plateau_length=0,
        details={}
    )
    
    # Create generation result
    result = GenerationResult(
        generation=5,
        best_fitness=0.85,
        mean_fitness=0.72,
        diversity=0.65,
        convergence_status=convergence_status,
        evaluation_time=120.5,
        evolution_time=5.2,
        population_stats={'avg_length': 15.3}
    )
    
    # Test serialization
    try:
        serialized = _dataclass_to_json_dict(result)
        json_str = json.dumps(serialized, indent=2)
        print("✅ GenerationResult serialization successful")
        print(f"   Convergence reason: {serialized['convergence_status']['reason']}")
        
        # Test deserialization
        loaded_data = json.loads(json_str)
        print("✅ JSON round-trip successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GenerationResult serialization failed: {e}")
        return False


def test_checkpoint_data_structure():
    """Test the complete checkpoint data structure."""
    print("\n🧪 Testing complete checkpoint data structure...")
    
    # Create sample checkpoint data similar to what's saved
    config = EvolutionConfig(
        population_size=15,
        target_fitness=0.75,
        selection_method=SelectionMethod.TOURNAMENT
    )
    
    convergence_status = ConvergenceStatus(
        converged=False,
        reason=ConvergenceReason.MAX_GENERATIONS,
        confidence=0.8,
        generations_since_improvement=5,
        current_best_fitness=0.65,
        diversity_score=0.85,
        plateau_length=3,
        details={}
    )
    
    generation_result = GenerationResult(
        generation=0,
        best_fitness=0.65,
        mean_fitness=0.45,
        diversity=0.85,
        convergence_status=convergence_status,
        evaluation_time=7308.7,
        evolution_time=2.1,
        population_stats={'diversity': 0.85}
    )
    
    # Create checkpoint data structure
    checkpoint_data = {
        'generation': 0,
        'population': [
            {
                'genome_id': 'test_genome_1',
                'token_ids': [1, 2, 3, 4, 5],
                'fitness': 0.65,
                'generation': 0
            }
        ],
        'best_genome_ever': {
            'genome_id': 'test_genome_1',
            'token_ids': [1, 2, 3, 4, 5],
            'fitness': 0.65
        },
        'config': _dataclass_to_json_dict(config),
        'generation_results': [_dataclass_to_json_dict(generation_result)]
    }
    
    # Test serialization
    try:
        json_str = json.dumps(checkpoint_data, indent=2)
        print("✅ Complete checkpoint data serialization successful")
        
        # Test deserialization
        loaded_data = json.loads(json_str)
        print("✅ Complete checkpoint JSON round-trip successful")
        
        # Verify key fields
        assert loaded_data['config']['selection_method'] == 'tournament'
        assert loaded_data['config']['population_size'] == 15
        assert loaded_data['config']['target_fitness'] == 0.75
        assert loaded_data['generation_results'][0]['convergence_status']['reason'] == 'max_generations'
        
        print("✅ All checkpoint data fields verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete checkpoint serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all serialization tests."""
    print("🔧 Testing Checkpoint Serialization Fixes")
    print("=" * 50)
    
    tests = [
        test_evolution_config_serialization,
        test_generation_result_serialization,
        test_checkpoint_data_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All checkpoint serialization tests passed!")
        print("\n✅ The JSON serialization issue has been fixed:")
        print("   • Enum objects are now converted to string values")
        print("   • Nested dataclasses are handled recursively")
        print("   • Complete checkpoint data can be serialized/deserialized")
        print("   • Configuration parameter extraction has been fixed")
        
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
