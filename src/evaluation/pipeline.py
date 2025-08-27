"""
Evaluation pipeline for genetic algorithm prompt evolution.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome
    from src.genetics.population import Population
    from src.evaluation.llm_interface import LLMInterface
    from src.evaluation.fitness import FitnessCalculator, FitnessComponents
    from src.evaluation.cache import evaluation_cache
    from src.utils.config import config
    from src.utils.dataset import gsm8k_dataset
else:
    from ..genetics.genome import PromptGenome
    from ..genetics.population import Population
    from .llm_interface import LLMInterface
    from .fitness import FitnessCalculator, FitnessComponents
    from .cache import evaluation_cache
    from ..utils.config import config
    from ..utils.dataset import gsm8k_dataset


@dataclass
class EvaluationResult:
    """Result of evaluating a genome."""
    genome_id: str
    prompt_text: str
    fitness_components: FitnessComponents
    evaluation_results: List[Dict[str, Any]]
    evaluation_time: float
    cache_hit: bool


class EvaluationPipeline:
    """Pipeline for evaluating prompt genomes on GSM8K problems."""
    
    def __init__(self, 
                 llm_interface: Optional[LLMInterface] = None,
                 fitness_calculator: Optional[FitnessCalculator] = None,
                 use_cache: bool = True,
                 batch_size: int = 10,
                 max_problems: int = 100):
        """
        Initialize evaluation pipeline.
        
        Args:
            llm_interface: LLM interface for evaluation
            fitness_calculator: Fitness calculator
            use_cache: Whether to use caching
            batch_size: Batch size for processing
            max_problems: Maximum number of problems to evaluate on
        """
        self.llm_interface = llm_interface or LLMInterface()
        self.fitness_calculator = fitness_calculator or FitnessCalculator()
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.max_problems = max_problems
        
        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.total_evaluation_time = 0.0
        self.api_calls_made = 0
        
        # Model info for caching
        self.model_info = {
            'model': self.llm_interface.model,
            'temperature': str(self.llm_interface.temperature),
            'max_tokens': str(self.llm_interface.max_tokens)
        }
    
    def evaluate_genome(self, genome: PromptGenome, 
                       problems: List[Dict[str, Any]],
                       progress_callback: Optional[Callable] = None) -> EvaluationResult:
        """
        Evaluate a single genome on a set of problems.
        
        Args:
            genome: Genome to evaluate
            problems: List of problems to evaluate on
            progress_callback: Optional progress callback
            
        Returns:
            EvaluationResult
        """
        start_time = time.time()
        cache_hit = False
        
        # Limit number of problems
        eval_problems = problems[:self.max_problems]
        
        # Check cache first
        if self.use_cache:
            cache_key = evaluation_cache.get_cache_key(genome, eval_problems, self.model_info)
            cached_entry = evaluation_cache.get_evaluation(cache_key)
            
            if cached_entry:
                cache_hit = True
                self.cache_hits += 1
                
                # Reconstruct fitness components
                fitness_components = FitnessComponents(**cached_entry.fitness_components)
                
                evaluation_time = time.time() - start_time
                
                return EvaluationResult(
                    genome_id=genome.genome_id,
                    prompt_text=genome.to_text(),
                    fitness_components=fitness_components,
                    evaluation_results=cached_entry.evaluation_results,
                    evaluation_time=evaluation_time,
                    cache_hit=cache_hit
                )
        
        # Perform evaluation
        prompt_text = genome.to_text()
        
        def progress_wrapper(current, total, result):
            if progress_callback:
                progress_callback(genome.genome_id, current, total, result)
        
        evaluation_results = self.llm_interface.batch_evaluate(
            prompt_text, eval_problems, progress_wrapper
        )
        
        # Calculate fitness
        fitness_components = self.fitness_calculator.calculate_fitness(
            genome, evaluation_results
        )
        
        # Store in cache
        if self.use_cache:
            evaluation_cache.store_evaluation(
                cache_key, genome, eval_problems, evaluation_results,
                fitness_components, self.model_info
            )
        
        # Update statistics
        self.total_evaluations += 1
        self.api_calls_made += len(eval_problems)
        evaluation_time = time.time() - start_time
        self.total_evaluation_time += evaluation_time
        
        return EvaluationResult(
            genome_id=genome.genome_id,
            prompt_text=prompt_text,
            fitness_components=fitness_components,
            evaluation_results=evaluation_results,
            evaluation_time=evaluation_time,
            cache_hit=cache_hit
        )
    
    def evaluate_population(self, population: Population,
                           problems: List[Dict[str, Any]],
                           progress_callback: Optional[Callable] = None) -> List[EvaluationResult]:
        """
        Evaluate all genomes in a population.
        
        Args:
            population: Population to evaluate
            problems: List of problems to evaluate on
            progress_callback: Optional progress callback
            
        Returns:
            List of EvaluationResults
        """
        results = []
        
        # Progress tracking
        if progress_callback is None:
            pbar = tqdm(total=len(population), desc="Evaluating population")
            
            def default_progress(genome_id, current, total, result):
                pbar.set_postfix({
                    'genome': genome_id[:8],
                    'problem': f"{current}/{total}",
                    'correct': result.get('is_correct', False)
                })
        else:
            default_progress = progress_callback
        
        # Evaluate each genome
        for i, genome in enumerate(population):
            try:
                result = self.evaluate_genome(genome, problems, default_progress)
                results.append(result)
                
                # Update genome fitness
                genome.set_fitness(result.fitness_components.overall_fitness)
                
                if progress_callback is None:
                    pbar.update(1)
                    pbar.set_description(f"Evaluating population (fitness: {result.fitness_components.overall_fitness:.3f})")
                
            except Exception as e:
                print(f"Error evaluating genome {genome.genome_id}: {e}")
                # Set low fitness for failed evaluations
                genome.set_fitness(0.0)
                
                # Create dummy result
                dummy_result = EvaluationResult(
                    genome_id=genome.genome_id,
                    prompt_text=genome.to_text(),
                    fitness_components=FitnessComponents(0, 0, 0, 0, 0, 0),
                    evaluation_results=[],
                    evaluation_time=0.0,
                    cache_hit=False
                )
                results.append(dummy_result)
        
        if progress_callback is None:
            pbar.close()
        
        return results
    
    def evaluate_adaptive(self, population: Population, generation: int) -> List[EvaluationResult]:
        """
        Evaluate population with adaptive problem selection.
        
        Args:
            population: Population to evaluate
            generation: Current generation number
            
        Returns:
            List of EvaluationResults
        """
        # Get adaptive problem set
        problems = gsm8k_dataset.get_adaptive_eval_problems(generation)
        
        print(f"Generation {generation}: Evaluating on {len(problems)} problems")
        
        return self.evaluate_population(population, problems)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation pipeline statistics."""
        avg_eval_time = (self.total_evaluation_time / self.total_evaluations 
                        if self.total_evaluations > 0 else 0.0)
        
        cache_hit_rate = (self.cache_hits / self.total_evaluations 
                         if self.total_evaluations > 0 else 0.0)
        
        # Get LLM interface stats
        llm_stats = self.llm_interface.get_statistics()
        
        # Get cache stats
        cache_stats = evaluation_cache.get_cache_statistics()
        
        return {
            'total_evaluations': self.total_evaluations,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'total_evaluation_time': self.total_evaluation_time,
            'avg_evaluation_time': avg_eval_time,
            'api_calls_made': self.api_calls_made,
            'llm_stats': llm_stats,
            'cache_stats': cache_stats,
            'batch_size': self.batch_size,
            'max_problems': self.max_problems
        }
    
    def save_evaluation_results(self, results: List[EvaluationResult], 
                               filepath: Path):
        """Save evaluation results to file."""
        import json
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'genome_id': result.genome_id,
                'prompt_text': result.prompt_text,
                'fitness_components': {
                    'accuracy': result.fitness_components.accuracy,
                    'consistency': result.fitness_components.consistency,
                    'efficiency': result.fitness_components.efficiency,
                    'diversity_bonus': result.fitness_components.diversity_bonus,
                    'length_penalty': result.fitness_components.length_penalty,
                    'overall_fitness': result.fitness_components.overall_fitness
                },
                'evaluation_time': result.evaluation_time,
                'cache_hit': result.cache_hit,
                'num_problems': len(result.evaluation_results),
                'correct_answers': sum(1 for r in result.evaluation_results if r.get('is_correct', False))
            })
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")


if __name__ == "__main__":
    # Test evaluation pipeline
    print("Testing evaluation pipeline...")
    
    # Load required components
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create test population
    population = Population(population_size=3)
    seed_prompts = [
        "Let's solve this step by step.",
        "Think about this problem carefully.",
        "Calculate the answer systematically."
    ]
    population.initialize_from_seeds(seed_prompts)
    
    # Create test problems
    test_problems = [
        {
            'id': 'test_1',
            'question': 'What is 15 + 27?',
            'final_answer': 42.0
        },
        {
            'id': 'test_2', 
            'question': 'If John has 5 apples and gives away 2, how many does he have left?',
            'final_answer': 3.0
        }
    ]
    
    # Create pipeline (without actual LLM calls)
    print("✅ Creating evaluation pipeline...")
    
    # Mock LLM interface for testing
    class MockLLMInterface:
        def __init__(self):
            self.model = "mock-model"
            self.temperature = 0.0
            self.max_tokens = 150
        
        def batch_evaluate(self, prompt, problems, progress_callback=None):
            import random
            results = []
            for i, problem in enumerate(problems):
                # Mock evaluation result
                is_correct = random.choice([True, False])
                predicted_answer = problem['final_answer'] if is_correct else random.uniform(0, 100)
                
                result = {
                    'problem_id': problem['id'],
                    'question': problem['question'],
                    'ground_truth': problem['final_answer'],
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': f"Mock response for {problem['question']}",
                    'response_length': random.randint(50, 200)
                }
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(problems), result)
            
            return results
        
        def get_statistics(self):
            return {'mock': True}
    
    # Create pipeline with mock interface
    pipeline = EvaluationPipeline(
        llm_interface=MockLLMInterface(),
        use_cache=False,  # Disable cache for testing
        max_problems=2
    )
    
    print("✅ Testing single genome evaluation...")
    test_genome = population.genomes[0]
    result = pipeline.evaluate_genome(test_genome, test_problems)
    
    print(f"✅ Genome evaluated: {result.genome_id[:8]}")
    print(f"   Fitness: {result.fitness_components.overall_fitness:.3f}")
    print(f"   Accuracy: {result.fitness_components.accuracy:.3f}")
    print(f"   Cache hit: {result.cache_hit}")
    
    print("✅ Testing population evaluation...")
    results = pipeline.evaluate_population(population, test_problems)
    
    print(f"✅ Population evaluated: {len(results)} genomes")
    for result in results:
        print(f"   {result.genome_id[:8]}: fitness={result.fitness_components.overall_fitness:.3f}")
    
    # Test statistics
    stats = pipeline.get_evaluation_statistics()
    print(f"✅ Pipeline statistics: {stats['total_evaluations']} evaluations")
    
    print("\n🎯 Evaluation pipeline tests completed successfully!")
