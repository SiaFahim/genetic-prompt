"""
Fitness calculation for genetic algorithm prompt evolution.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome
    from src.utils.config import config
else:
    from ..genetics.genome import PromptGenome
    from ..utils.config import config


@dataclass
class FitnessComponents:
    """Components that make up the overall fitness score."""
    accuracy: float  # Primary metric: fraction of correct answers
    consistency: float  # How consistent the prompt is across problems
    efficiency: float  # Response length efficiency
    diversity_bonus: float  # Bonus for diverse problem-solving approaches
    length_penalty: float  # Penalty for overly long prompts
    overall_fitness: float  # Combined fitness score


class FitnessCalculator:
    """Calculates fitness scores for prompt genomes."""
    
    def __init__(self, 
                 accuracy_weight: float = 0.7,
                 consistency_weight: float = 0.15,
                 efficiency_weight: float = 0.1,
                 diversity_weight: float = 0.05,
                 length_penalty_threshold: int = 30,
                 length_penalty_factor: float = 0.01):
        """
        Initialize fitness calculator.
        
        Args:
            accuracy_weight: Weight for accuracy component
            consistency_weight: Weight for consistency component
            efficiency_weight: Weight for efficiency component
            diversity_weight: Weight for diversity bonus
            length_penalty_threshold: Prompt length threshold for penalty
            length_penalty_factor: Factor for length penalty
        """
        self.accuracy_weight = accuracy_weight
        self.consistency_weight = consistency_weight
        self.efficiency_weight = efficiency_weight
        self.diversity_weight = diversity_weight
        self.length_penalty_threshold = length_penalty_threshold
        self.length_penalty_factor = length_penalty_factor
        
        # Normalize weights
        total_weight = (accuracy_weight + consistency_weight + 
                       efficiency_weight + diversity_weight)
        if total_weight != 1.0:
            self.accuracy_weight /= total_weight
            self.consistency_weight /= total_weight
            self.efficiency_weight /= total_weight
            self.diversity_weight /= total_weight
    
    def calculate_accuracy(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """
        Calculate accuracy score from evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not evaluation_results:
            return 0.0
        
        correct_count = sum(1 for result in evaluation_results if result.get('is_correct', False))
        return correct_count / len(evaluation_results)
    
    def calculate_consistency(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency score based on response patterns.
        
        Consistency measures how reliably the prompt produces structured responses.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Consistency score between 0 and 1
        """
        if not evaluation_results:
            return 0.0
        
        # Measure consistency in response lengths
        response_lengths = [result.get('response_length', 0) for result in evaluation_results]
        if not response_lengths or max(response_lengths) == 0:
            return 0.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_length = statistics.mean(response_lengths)
        if mean_length == 0:
            return 0.0
        
        std_length = statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0
        cv = std_length / mean_length
        
        # Convert to consistency score (0 = inconsistent, 1 = perfectly consistent)
        consistency_score = max(0.0, 1.0 - cv)
        
        # Bonus for having answers (even if wrong)
        answer_rate = sum(1 for result in evaluation_results 
                         if result.get('predicted_answer') is not None) / len(evaluation_results)
        
        return (consistency_score * 0.7) + (answer_rate * 0.3)
    
    def calculate_efficiency(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """
        Calculate efficiency score based on response length vs accuracy.
        
        More efficient prompts produce correct answers with shorter responses.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Efficiency score between 0 and 1
        """
        if not evaluation_results:
            return 0.0
        
        correct_results = [r for r in evaluation_results if r.get('is_correct', False)]
        if not correct_results:
            return 0.0
        
        # Calculate average response length for correct answers
        correct_lengths = [r.get('response_length', 0) for r in correct_results]
        avg_correct_length = statistics.mean(correct_lengths)
        
        # Efficiency is inversely related to length (with reasonable bounds)
        # Optimal length is around 100-200 characters
        optimal_length = 150
        if avg_correct_length <= optimal_length:
            efficiency = 1.0
        else:
            # Penalty for overly long responses
            excess_length = avg_correct_length - optimal_length
            efficiency = max(0.1, 1.0 - (excess_length / 1000))  # Gradual penalty
        
        return efficiency
    
    def calculate_diversity_bonus(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """
        Calculate diversity bonus based on variety in problem-solving approaches.
        
        This is a simplified version that looks at response length variation.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Diversity bonus between 0 and 1
        """
        if not evaluation_results or len(evaluation_results) < 2:
            return 0.0
        
        # Look at response length diversity for correct answers
        correct_results = [r for r in evaluation_results if r.get('is_correct', False)]
        if len(correct_results) < 2:
            return 0.0
        
        response_lengths = [r.get('response_length', 0) for r in correct_results]
        
        # Calculate normalized standard deviation
        mean_length = statistics.mean(response_lengths)
        if mean_length == 0:
            return 0.0
        
        std_length = statistics.stdev(response_lengths)
        normalized_std = std_length / mean_length
        
        # Moderate diversity is good (not too uniform, not too chaotic)
        optimal_diversity = 0.3
        diversity_score = 1.0 - abs(normalized_std - optimal_diversity) / optimal_diversity
        
        return max(0.0, min(1.0, diversity_score))
    
    def calculate_length_penalty(self, genome: PromptGenome) -> float:
        """
        Calculate penalty for overly long prompts.
        
        Args:
            genome: The prompt genome
            
        Returns:
            Length penalty (0 = no penalty, higher = more penalty)
        """
        prompt_length = genome.length()
        
        if prompt_length <= self.length_penalty_threshold:
            return 0.0
        
        excess_length = prompt_length - self.length_penalty_threshold
        penalty = excess_length * self.length_penalty_factor
        
        return min(penalty, 0.5)  # Cap penalty at 0.5
    
    def calculate_fitness(self, genome: PromptGenome, 
                         evaluation_results: List[Dict[str, Any]]) -> FitnessComponents:
        """
        Calculate overall fitness score for a genome.
        
        Args:
            genome: The prompt genome
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            FitnessComponents object with detailed breakdown
        """
        # Calculate individual components
        accuracy = self.calculate_accuracy(evaluation_results)
        consistency = self.calculate_consistency(evaluation_results)
        efficiency = self.calculate_efficiency(evaluation_results)
        diversity_bonus = self.calculate_diversity_bonus(evaluation_results)
        length_penalty = self.calculate_length_penalty(genome)
        
        # Calculate weighted overall fitness
        overall_fitness = (
            accuracy * self.accuracy_weight +
            consistency * self.consistency_weight +
            efficiency * self.efficiency_weight +
            diversity_bonus * self.diversity_weight -
            length_penalty
        )
        
        # Ensure fitness is non-negative
        overall_fitness = max(0.0, overall_fitness)
        
        return FitnessComponents(
            accuracy=accuracy,
            consistency=consistency,
            efficiency=efficiency,
            diversity_bonus=diversity_bonus,
            length_penalty=length_penalty,
            overall_fitness=overall_fitness
        )
    
    def get_fitness_statistics(self, fitness_scores: List[float]) -> Dict[str, float]:
        """Get statistics for a list of fitness scores."""
        if not fitness_scores:
            return {'count': 0}
        
        return {
            'count': len(fitness_scores),
            'mean': statistics.mean(fitness_scores),
            'median': statistics.median(fitness_scores),
            'std': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0,
            'min': min(fitness_scores),
            'max': max(fitness_scores),
            'range': max(fitness_scores) - min(fitness_scores)
        }


if __name__ == "__main__":
    # Test fitness calculator
    print("Testing fitness calculator...")
    
    # Create test genome
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    test_genome = PromptGenome.from_text("Let's solve this step by step carefully.")
    
    # Create mock evaluation results
    mock_results = [
        {
            'problem_id': 'test_1',
            'is_correct': True,
            'predicted_answer': 42.0,
            'response_length': 150,
            'response': 'Step 1: ... Step 2: ... Answer: 42'
        },
        {
            'problem_id': 'test_2',
            'is_correct': True,
            'predicted_answer': 15.0,
            'response_length': 120,
            'response': 'First, ... Then, ... Answer: 15'
        },
        {
            'problem_id': 'test_3',
            'is_correct': False,
            'predicted_answer': 30.0,
            'response_length': 180,
            'response': 'Let me think... Actually... Answer: 30'
        },
        {
            'problem_id': 'test_4',
            'is_correct': True,
            'predicted_answer': 8.0,
            'response_length': 140,
            'response': 'Breaking this down... Answer: 8'
        }
    ]
    
    # Test fitness calculation
    calculator = FitnessCalculator()
    fitness_components = calculator.calculate_fitness(test_genome, mock_results)
    
    print(f"✅ Test genome: {test_genome}")
    print(f"✅ Accuracy: {fitness_components.accuracy:.3f}")
    print(f"✅ Consistency: {fitness_components.consistency:.3f}")
    print(f"✅ Efficiency: {fitness_components.efficiency:.3f}")
    print(f"✅ Diversity bonus: {fitness_components.diversity_bonus:.3f}")
    print(f"✅ Length penalty: {fitness_components.length_penalty:.3f}")
    print(f"✅ Overall fitness: {fitness_components.overall_fitness:.3f}")
    
    # Test with different scenarios
    print("\nTesting edge cases...")
    
    # Empty results
    empty_fitness = calculator.calculate_fitness(test_genome, [])
    print(f"✅ Empty results fitness: {empty_fitness.overall_fitness:.3f}")
    
    # All wrong answers
    wrong_results = [{'is_correct': False, 'response_length': 100} for _ in range(3)]
    wrong_fitness = calculator.calculate_fitness(test_genome, wrong_results)
    print(f"✅ All wrong fitness: {wrong_fitness.overall_fitness:.3f}")
    
    # Perfect accuracy
    perfect_results = [{'is_correct': True, 'response_length': 150} for _ in range(5)]
    perfect_fitness = calculator.calculate_fitness(test_genome, perfect_results)
    print(f"✅ Perfect accuracy fitness: {perfect_fitness.overall_fitness:.3f}")
    
    print("\n🎯 Fitness calculator tests completed successfully!")
