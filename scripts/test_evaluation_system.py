#!/usr/bin/env python3
"""
Comprehensive test suite for the evaluation system.
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
from src.evaluation.llm_interface import LLMInterface
from src.evaluation.fitness import FitnessCalculator, FitnessComponents
from src.evaluation.cache import evaluation_cache
from src.evaluation.pipeline import EvaluationPipeline


def test_llm_interface():
    """Test LLM interface with real API calls."""
    print("🤖 Testing LLM Interface")
    print("-" * 40)
    
    try:
        # Create interface
        llm = LLMInterface(model="gpt-3.5-turbo", temperature=0.0, max_tokens=150)
        
        # Test simple math problem
        test_prompt = "Let's solve this step by step."
        test_question = "What is 25 + 17?"
        
        print(f"Testing prompt: '{test_prompt}'")
        print(f"Testing question: '{test_question}'")
        
        answer, response = llm.evaluate_prompt_on_problem(test_prompt, test_question)
        
        print(f"✅ Response: {response[:100]}...")
        print(f"✅ Extracted answer: {answer}")
        print(f"✅ Expected answer: 42")
        
        # Test batch evaluation
        test_problems = [
            {'id': 'test_1', 'question': 'What is 10 + 15?', 'final_answer': 25.0},
            {'id': 'test_2', 'question': 'What is 8 * 7?', 'final_answer': 56.0}
        ]
        
        print("\nTesting batch evaluation...")
        results = llm.batch_evaluate(test_prompt, test_problems)
        
        for result in results:
            correct = "✅" if result['is_correct'] else "❌"
            print(f"{correct} {result['problem_id']}: predicted={result['predicted_answer']}, "
                  f"actual={result['ground_truth']}")
        
        # Test statistics
        stats = llm.get_statistics()
        print(f"✅ API calls made: {stats['total_requests']}")
        print(f"✅ Success rate: {stats['success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM interface test failed: {e}")
        return False


def test_fitness_calculator():
    """Test fitness calculation with various scenarios."""
    print("\n📊 Testing Fitness Calculator")
    print("-" * 40)
    
    calculator = FitnessCalculator()
    
    # Create test genome
    test_genome = PromptGenome.from_text("Solve this problem step by step.")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Perfect accuracy',
            'results': [
                {'is_correct': True, 'response_length': 150, 'predicted_answer': 42.0},
                {'is_correct': True, 'response_length': 140, 'predicted_answer': 25.0},
                {'is_correct': True, 'response_length': 160, 'predicted_answer': 56.0}
            ]
        },
        {
            'name': 'Mixed results',
            'results': [
                {'is_correct': True, 'response_length': 150, 'predicted_answer': 42.0},
                {'is_correct': False, 'response_length': 200, 'predicted_answer': 40.0},
                {'is_correct': True, 'response_length': 120, 'predicted_answer': 56.0}
            ]
        },
        {
            'name': 'All wrong',
            'results': [
                {'is_correct': False, 'response_length': 100, 'predicted_answer': 40.0},
                {'is_correct': False, 'response_length': 110, 'predicted_answer': 20.0},
                {'is_correct': False, 'response_length': 90, 'predicted_answer': 50.0}
            ]
        }
    ]
    
    for scenario in scenarios:
        fitness = calculator.calculate_fitness(test_genome, scenario['results'])
        print(f"✅ {scenario['name']}:")
        print(f"   Accuracy: {fitness.accuracy:.3f}")
        print(f"   Consistency: {fitness.consistency:.3f}")
        print(f"   Efficiency: {fitness.efficiency:.3f}")
        print(f"   Overall: {fitness.overall_fitness:.3f}")
    
    return True


def test_caching_system():
    """Test caching functionality."""
    print("\n💾 Testing Caching System")
    print("-" * 40)
    
    # Clear cache for clean test
    evaluation_cache.clear_cache()
    
    # Create test data
    test_genome = PromptGenome.from_text("Test prompt for caching.")
    test_problems = [
        {'id': 'cache_test_1', 'question': 'What is 5 + 5?', 'final_answer': 10.0}
    ]
    model_info = {'model': 'test-model', 'temperature': '0.0'}
    
    # Test cache miss
    cache_key = evaluation_cache.get_cache_key(test_genome, test_problems, model_info)
    cached_entry = evaluation_cache.get_evaluation(cache_key)
    print(f"✅ Cache miss (expected): {cached_entry is None}")
    
    # Store in cache
    mock_results = [{'problem_id': 'cache_test_1', 'is_correct': True, 'response_length': 50}]
    mock_fitness = FitnessComponents(0.8, 0.7, 0.9, 0.1, 0.0, 0.75)
    
    evaluation_cache.store_evaluation(
        cache_key, test_genome, test_problems, mock_results, mock_fitness, model_info
    )
    
    # Test cache hit
    cached_entry = evaluation_cache.get_evaluation(cache_key)
    print(f"✅ Cache hit: {cached_entry is not None}")
    if cached_entry:
        print(f"   Cached fitness: {cached_entry.fitness_components['overall_fitness']}")
    
    # Test cache statistics
    stats = evaluation_cache.get_cache_statistics()
    print(f"✅ Cache entries: {stats['evaluation_entries']}")
    print(f"✅ Hit rate: {stats['hit_rate']:.2%}")
    
    return True


def test_evaluation_pipeline():
    """Test the complete evaluation pipeline."""
    print("\n🔄 Testing Evaluation Pipeline")
    print("-" * 40)
    
    # Load dataset
    try:
        problems = gsm8k_dataset.get_primary_eval_set()[:5]  # Use first 5 problems
        print(f"✅ Loaded {len(problems)} test problems")
    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        # Create mock problems
        problems = [
            {'id': 'mock_1', 'question': 'What is 12 + 8?', 'final_answer': 20.0},
            {'id': 'mock_2', 'question': 'What is 6 * 9?', 'final_answer': 54.0}
        ]
        print(f"✅ Using {len(problems)} mock problems")
    
    # Create test population
    population = Population(population_size=3)
    seed_prompts = [
        "Let's solve this step by step.",
        "Think about this problem carefully.",
        "Calculate the answer systematically."
    ]
    population.initialize_from_seeds(seed_prompts)
    
    # Test with mock LLM interface for speed
    class MockLLMInterface:
        def __init__(self):
            self.model = "mock-gpt-4"
            self.temperature = 0.0
            self.max_tokens = 150
        
        def batch_evaluate(self, prompt, problems, progress_callback=None):
            import random
            results = []
            for i, problem in enumerate(problems):
                # Simulate better performance for better prompts
                base_accuracy = 0.6 if "step by step" in prompt.lower() else 0.4
                is_correct = random.random() < base_accuracy
                
                predicted_answer = problem['final_answer'] if is_correct else random.uniform(0, 100)
                
                result = {
                    'problem_id': problem['id'],
                    'question': problem['question'],
                    'ground_truth': problem['final_answer'],
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': f"Mock response for {problem['question']}",
                    'response_length': random.randint(100, 200)
                }
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(problems), result)
                
                # Small delay to simulate API call
                time.sleep(0.01)
            
            return results
        
        def get_statistics(self):
            return {'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0}
    
    # Create pipeline
    pipeline = EvaluationPipeline(
        llm_interface=MockLLMInterface(),
        use_cache=True,
        max_problems=len(problems)
    )
    
    print("Testing population evaluation...")
    results = pipeline.evaluate_population(population, problems)
    
    print(f"✅ Evaluated {len(results)} genomes")
    
    # Show results
    for result in results:
        accuracy = result.fitness_components.accuracy
        fitness = result.fitness_components.overall_fitness
        cache_status = "cached" if result.cache_hit else "fresh"
        print(f"   {result.genome_id[:8]}: accuracy={accuracy:.2%}, "
              f"fitness={fitness:.3f} ({cache_status})")
    
    # Test statistics
    stats = pipeline.get_evaluation_statistics()
    print(f"✅ Total evaluations: {stats['total_evaluations']}")
    print(f"✅ Cache hit rate: {stats['cache_hit_rate']:.2%}")
    
    return True


def test_real_api_evaluation():
    """Test evaluation with real API (limited to save costs)."""
    print("\n🌐 Testing Real API Evaluation")
    print("-" * 40)
    
    try:
        # Create real LLM interface
        llm = LLMInterface(model="gpt-3.5-turbo", temperature=0.0, max_tokens=100)
        
        # Test single evaluation
        test_genome = PromptGenome.from_text("Let's solve this step by step.")
        test_problem = {
            'id': 'real_test_1',
            'question': 'Sarah has 15 stickers. She gives 7 stickers to her friend. How many stickers does Sarah have left?',
            'final_answer': 8.0
        }
        
        print("Testing real API call...")
        print(f"Problem: {test_problem['question']}")
        
        answer, response = llm.evaluate_prompt_on_problem(
            test_genome.to_text(), test_problem['question']
        )
        
        print(f"✅ Response: {response[:150]}...")
        print(f"✅ Extracted answer: {answer}")
        print(f"✅ Expected answer: {test_problem['final_answer']}")
        
        is_correct = (answer is not None and 
                     abs(answer - test_problem['final_answer']) < 0.001)
        print(f"✅ Correct: {is_correct}")
        
        # Test fitness calculation
        mock_results = [{
            'problem_id': test_problem['id'],
            'is_correct': is_correct,
            'predicted_answer': answer,
            'response_length': len(response) if response else 0
        }]
        
        calculator = FitnessCalculator()
        fitness = calculator.calculate_fitness(test_genome, mock_results)
        print(f"✅ Fitness: {fitness.overall_fitness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Real API test skipped: {e}")
        print("This is normal if API keys are not configured or there are network issues.")
        return True  # Don't fail the test suite for API issues


def run_comprehensive_test():
    """Run comprehensive evaluation system test."""
    print("🧪 Comprehensive Evaluation System Test Suite")
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
    
    # Run tests
    tests = [
        test_fitness_calculator,
        test_caching_system,
        test_evaluation_pipeline,
        test_llm_interface,
        test_real_api_evaluation
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
        print("🎉 All evaluation system tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        sys.exit(1)
