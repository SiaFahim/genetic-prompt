#!/usr/bin/env python3
"""
Integration test for the complete async batch evaluation system.

This script tests the entire pipeline from configuration to execution,
ensuring all components work together correctly.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test imports to ensure all modules are properly integrated
try:
    from src.genetics.async_evolution import AsyncEvolutionController, AsyncEvolutionConfig
    from src.evaluation.async_pipeline import AsyncEvaluationPipeline, PopulationBatchConfig
    from src.evaluation.async_llm_interface import AsyncLLMInterface, BatchConfig
    from src.genetics.population import Population
    from src.genetics.genome import PromptGenome
    from src.utils.dataset import gsm8k_dataset
    from src.config.hyperparameters import get_hyperparameter_config
    from src.embeddings.vocabulary import vocabulary

    print("✅ All imports successful")

    # Initialize vocabulary for genome operations
    print("🔧 Initializing vocabulary...")
    if not vocabulary.vocab_built:
        vocabulary.build_vocabulary_from_dataset()
    print("✅ Vocabulary initialized")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class IntegrationTester:
    """Comprehensive integration tester for the async system."""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
    
    async def test_async_llm_interface(self) -> bool:
        """Test the async LLM interface."""
        print("\n🧪 Testing AsyncLLMInterface...")
        
        try:
            # Create minimal batch config for testing
            batch_config = BatchConfig(
                batch_size=5,
                max_concurrent_requests=3,
                rate_limit_per_minute=100,  # Very conservative for testing
                retry_attempts=2,
                timeout=30
            )
            
            # Create interface
            async_llm = AsyncLLMInterface(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=50,  # Small for testing
                batch_config=batch_config
            )
            
            # Test single evaluation
            test_prompt = "Solve step by step:"
            test_question = "What is 2 + 2?"
            
            result = await async_llm._evaluate_single_problem_async(
                test_prompt, test_question, 4.0
            )
            
            # Verify result structure
            required_keys = ['question', 'ground_truth', 'predicted_answer', 'is_correct', 'response']
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing key in result: {key}")
            
            print(f"   ✅ Single evaluation test passed")
            print(f"   📊 Result: {result['is_correct']} (predicted: {result['predicted_answer']})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ AsyncLLMInterface test failed: {e}")
            self.errors.append(f"AsyncLLMInterface: {e}")
            return False
    
    async def test_async_pipeline(self) -> bool:
        """Test the async evaluation pipeline."""
        print("\n🧪 Testing AsyncEvaluationPipeline...")
        
        try:
            # Create test configuration
            batch_config = BatchConfig(
                batch_size=3,
                max_concurrent_requests=2,
                rate_limit_per_minute=50,
                timeout=30
            )
            
            population_batch_config = PopulationBatchConfig(
                genome_batch_size=3,
                problem_batch_size=3,
                max_concurrent_genomes=2,
                enable_progress_bar=False,  # Disable for testing
                detailed_logging=False
            )
            
            # Create pipeline
            async_llm = AsyncLLMInterface(batch_config=batch_config)
            pipeline = AsyncEvaluationPipeline(
                async_llm_interface=async_llm,
                population_batch_config=population_batch_config,
                max_problems=5  # Small number for testing
            )
            
            # Create test population
            population = Population(3)
            test_prompts = [
                "Solve this step by step:",
                "Let me calculate:",
                "To find the answer:"
            ]
            
            for prompt in test_prompts:
                genome = PromptGenome.from_text(prompt)
                population.add_genome(genome)
            
            # Get test problems
            problems = gsm8k_dataset.get_primary_eval_set()[:5]
            
            # Run evaluation
            result = await pipeline.evaluate_population_async(population, problems)
            
            # Verify results
            if result.successful_evaluations != len(population):
                print(f"   ⚠️  Warning: {result.failed_evaluations} failed evaluations")
            
            if result.successful_evaluations == 0:
                raise ValueError("No successful evaluations")
            
            print(f"   ✅ Pipeline test passed")
            print(f"   📊 Successful: {result.successful_evaluations}, Failed: {result.failed_evaluations}")
            print(f"   ⏱️  Total time: {result.total_time:.2f}s")
            print(f"   🚀 Throughput: {result.throughput_problems_per_second:.1f} problems/s")
            
            return True
            
        except Exception as e:
            print(f"   ❌ AsyncEvaluationPipeline test failed: {e}")
            self.errors.append(f"AsyncEvaluationPipeline: {e}")
            return False
    
    async def test_async_evolution_controller(self) -> bool:
        """Test the async evolution controller."""
        print("\n🧪 Testing AsyncEvolutionController...")
        
        try:
            # Create minimal config for testing
            config = AsyncEvolutionConfig(
                population_size=5,
                max_generations=2,  # Very short for testing
                crossover_rate=0.8,
                mutation_rate=0.3,
                elite_size=2,
                
                # Async settings
                enable_async_evaluation=True,
                async_batch_size=3,
                max_concurrent_requests=2,
                genome_batch_size=3,
                max_concurrent_genomes=2,
                rate_limit_per_minute=50,
                detailed_performance_logging=False
            )
            
            # Create controller
            controller = AsyncEvolutionController(
                config=config,
                seed_prompts=["Solve step by step:", "Calculate:", "Find the answer:"]
            )
            
            # Test single generation
            result = await controller.evolve_generation_async()
            
            # Verify result
            if not hasattr(result, 'generation'):
                raise ValueError("Invalid generation result")
            
            if result.best_fitness is None:
                raise ValueError("No fitness calculated")
            
            print(f"   ✅ Evolution controller test passed")
            print(f"   📊 Generation: {result.generation}")
            print(f"   🏆 Best fitness: {result.best_fitness:.3f}")
            print(f"   ⏱️  Evaluation time: {result.evaluation_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"   ❌ AsyncEvolutionController test failed: {e}")
            self.errors.append(f"AsyncEvolutionController: {e}")
            return False
    
    def test_configuration_system(self) -> bool:
        """Test the configuration system integration."""
        print("\n🧪 Testing Configuration System...")
        
        try:
            # Test hyperparameter config
            hyperparams = get_hyperparameter_config()
            
            # Check for new async parameters
            async_params = [
                'enable_async_evaluation',
                'async_batch_size',
                'max_concurrent_requests',
                'genome_batch_size',
                'max_concurrent_genomes',
                'rate_limit_per_minute'
            ]
            
            for param in async_params:
                if not hasattr(hyperparams, param):
                    raise ValueError(f"Missing async parameter: {param}")
            
            # Test parameter specs
            specs = hyperparams.get_parameter_specs()
            for param in async_params:
                if param not in specs:
                    raise ValueError(f"Missing parameter spec: {param}")
            
            print(f"   ✅ Configuration system test passed")
            print(f"   📊 Async parameters: {len(async_params)} configured")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Configuration system test failed: {e}")
            self.errors.append(f"Configuration: {e}")
            return False
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🧬 Starting Async Batch Evaluation Integration Tests")
        print("="*60)
        
        tests = [
            ("Configuration System", self.test_configuration_system()),
            ("AsyncLLMInterface", self.test_async_llm_interface()),
            ("AsyncEvaluationPipeline", self.test_async_pipeline()),
            ("AsyncEvolutionController", self.test_async_evolution_controller())
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_coro in tests:
            try:
                if asyncio.iscoroutine(test_coro):
                    result = await test_coro
                else:
                    result = test_coro
                
                results[test_name] = result
                if result:
                    passed += 1
                    
            except Exception as e:
                print(f"   ❌ {test_name} test crashed: {e}")
                traceback.print_exc()
                results[test_name] = False
                self.errors.append(f"{test_name}: {e}")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"🧪 INTEGRATION TEST RESULTS")
        print(f"="*60)
        print(f"✅ Passed: {passed}/{total} tests")
        print(f"❌ Failed: {total - passed}/{total} tests")
        
        if self.errors:
            print(f"\n🚨 Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")
        
        if passed == total:
            print(f"\n🎉 All integration tests passed!")
            print(f"🚀 The async batch evaluation system is ready for use.")
        else:
            print(f"\n⚠️  Some tests failed. Please review the errors above.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'test_results': results,
            'errors': self.errors,
            'success': passed == total
        }


async def main():
    """Main test execution."""
    print("🧬 Genetic Algorithm Async Batch Evaluation")
    print("Integration Test Suite")
    print("="*50)
    
    tester = IntegrationTester()
    results = await tester.run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
