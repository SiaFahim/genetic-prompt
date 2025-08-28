#!/usr/bin/env python3
"""
Test script to verify that the logging system can handle enum serialization correctly.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.evolution_logging import EvolutionLogger, _convert_to_json_serializable
from src.genetics.convergence import ConvergenceReason, ConvergenceStatus
from src.genetics.evolution import GenerationResult


def test_enum_serialization():
    """Test that enums can be serialized correctly."""
    print("🧪 Testing Enum Serialization")
    print("-" * 40)
    
    # Test ConvergenceReason enum
    reason = ConvergenceReason.STAGNATION
    serialized = _convert_to_json_serializable(reason)
    print(f"ConvergenceReason.STAGNATION → {serialized}")
    assert serialized == "stagnation", f"Expected 'stagnation', got {serialized}"
    
    # Test that it can be JSON dumped
    json_str = json.dumps(serialized)
    print(f"JSON string: {json_str}")
    
    print("✅ Enum serialization works")
    return True


def test_convergence_status_serialization():
    """Test that ConvergenceStatus dataclass can be serialized."""
    print("\n🧪 Testing ConvergenceStatus Serialization")
    print("-" * 40)
    
    # Create convergence status with enum
    status = ConvergenceStatus(
        converged=True,
        reason=ConvergenceReason.STAGNATION,
        confidence=0.95,
        generations_since_improvement=10,
        current_best_fitness=0.75,
        diversity_score=0.45,
        plateau_length=8,
        details={'threshold': 15}
    )
    
    # Serialize it
    serialized = _convert_to_json_serializable(status)
    print(f"Serialized ConvergenceStatus:")
    print(json.dumps(serialized, indent=2))
    
    # Verify the enum was converted
    assert serialized['reason'] == "stagnation", f"Expected 'stagnation', got {serialized['reason']}"
    assert serialized['converged'] == True
    assert serialized['confidence'] == 0.95
    
    print("✅ ConvergenceStatus serialization works")
    return True


def test_final_results_serialization():
    """Test serialization of final results that would cause the original error."""
    print("\n🧪 Testing Final Results Serialization")
    print("-" * 40)
    
    # Create final results similar to what caused the error
    final_results = {
        'best_fitness': 0.583,
        'best_genome': 'cousin this yet true then what follows do they <UNK> the solution',
        'total_generations': 14,
        'total_time': 9756.4,
        'total_evaluations': 210,
        'convergence_reason': ConvergenceReason.STAGNATION,  # This was the problem!
        'generation_results': [
            {
                'generation': 1,
                'best_fitness': 0.41,
                'convergence_status': ConvergenceStatus(
                    converged=False,
                    reason=ConvergenceReason.NOT_CONVERGED,
                    confidence=0.0,
                    generations_since_improvement=0,
                    current_best_fitness=0.41,
                    diversity_score=0.9,
                    plateau_length=0,
                    details={}
                )
            }
        ]
    }
    
    # This should work now with our serialization function
    serialized = _convert_to_json_serializable(final_results)
    
    # Try to JSON dump it
    json_str = json.dumps(serialized, indent=2)
    print(f"Successfully serialized final results:")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    
    # Verify key conversions
    assert serialized['convergence_reason'] == "stagnation"
    assert serialized['generation_results'][0]['convergence_status']['reason'] == "not_converged"
    
    print("✅ Final results serialization works")
    return True


def test_logging_system_integration():
    """Test that the logging system can handle problematic data."""
    print("\n🧪 Testing Logging System Integration")
    print("-" * 40)
    
    # Create a test logger
    logger = EvolutionLogger("test_serialization_experiment")
    
    # Test configuration with enums (this could happen in experiment configs)
    test_config = {
        'name': 'Test Experiment',
        'population_size': 15,
        'convergence_reason': ConvergenceReason.TARGET_FITNESS,  # Enum in config
        'nested_data': {
            'status': ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.DIVERSITY_LOSS,
                confidence=0.8,
                generations_since_improvement=5,
                current_best_fitness=0.9,
                diversity_score=0.1,
                plateau_length=0,
                details={}
            )
        }
    }
    
    # This should not crash now
    try:
        logger.log_experiment_start(test_config)
        print("✅ log_experiment_start handled enums correctly")
    except Exception as e:
        print(f"❌ log_experiment_start failed: {e}")
        return False
    
    # Test final results logging
    final_results = {
        'best_fitness': 0.85,
        'convergence_reason': ConvergenceReason.STAGNATION,
        'final_status': ConvergenceStatus(
            converged=True,
            reason=ConvergenceReason.STAGNATION,
            confidence=1.0,
            generations_since_improvement=15,
            current_best_fitness=0.85,
            diversity_score=0.3,
            plateau_length=15,
            details={}
        )
    }
    
    try:
        logger.log_experiment_end(final_results)
        print("✅ log_experiment_end handled enums correctly")
    except Exception as e:
        print(f"❌ log_experiment_end failed: {e}")
        return False
    
    print("✅ Logging system integration works")
    return True


def main():
    """Run all serialization tests."""
    print("🔍 JSON Serialization Fix Verification")
    print("=" * 60)
    
    tests = [
        test_enum_serialization,
        test_convergence_status_serialization,
        test_final_results_serialization,
        test_logging_system_integration
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
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL SERIALIZATION TESTS PASSED!")
        print("\n✅ Fix Summary:")
        print("   • Added _convert_to_json_serializable() function")
        print("   • Fixed log_experiment_start() to handle enums")
        print("   • Fixed log_experiment_end() to handle enums")
        print("   • Fixed all JSON dump operations in logging")
        print("   • ConvergenceReason enums now serialize to their string values")
        print("   • ConvergenceStatus dataclasses serialize correctly")
        
        print("\n🚀 The experiment should now complete without JSON serialization errors!")
        return True
    else:
        print("❌ Some serialization tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
