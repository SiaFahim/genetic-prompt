"""
Convergence detection for genetic algorithm evolution.
"""

import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.population import Population
    from src.utils.config import config
else:
    from .population import Population
    from ..utils.config import config


class ConvergenceReason(Enum):
    """Reasons for convergence detection."""
    FITNESS_PLATEAU = "fitness_plateau"
    DIVERSITY_LOSS = "diversity_loss"
    MAX_GENERATIONS = "max_generations"
    TARGET_FITNESS = "target_fitness"
    STAGNATION = "stagnation"
    NOT_CONVERGED = "not_converged"


@dataclass
class ConvergenceStatus:
    """Status of convergence detection."""
    converged: bool
    reason: ConvergenceReason
    confidence: float  # 0.0 to 1.0
    generations_since_improvement: int
    current_best_fitness: float
    diversity_score: float
    plateau_length: int
    details: Dict[str, Any]


class ConvergenceDetector:
    """Detects convergence in genetic algorithm evolution."""
    
    def __init__(self,
                 fitness_plateau_threshold: float = 0.001,
                 fitness_plateau_generations: int = 10,
                 diversity_threshold: float = 0.05,
                 max_generations: int = 100,
                 target_fitness: Optional[float] = None,
                 stagnation_threshold: int = 20,
                 improvement_threshold: float = 0.01):
        """
        Initialize convergence detector.
        
        Args:
            fitness_plateau_threshold: Minimum fitness improvement to avoid plateau
            fitness_plateau_generations: Generations without improvement for plateau
            diversity_threshold: Minimum diversity to avoid convergence
            max_generations: Maximum generations before forced convergence
            target_fitness: Target fitness for early stopping
            stagnation_threshold: Generations without improvement for stagnation
            improvement_threshold: Minimum improvement to reset stagnation counter
        """
        self.fitness_plateau_threshold = fitness_plateau_threshold
        self.fitness_plateau_generations = fitness_plateau_generations
        self.diversity_threshold = diversity_threshold
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.stagnation_threshold = stagnation_threshold
        self.improvement_threshold = improvement_threshold
        
        # History tracking
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.best_fitness_history: List[float] = []
        self.generation_count = 0
        self.last_improvement_generation = 0
        self.best_fitness_ever = float('-inf')
    
    def update(self, population: Population) -> ConvergenceStatus:
        """
        Update convergence detector with current population state.
        
        Args:
            population: Current population
            
        Returns:
            ConvergenceStatus indicating convergence state
        """
        self.generation_count += 1
        
        # Calculate current metrics
        fitness_stats = population.get_fitness_statistics()
        diversity = population.calculate_diversity()
        current_best = fitness_stats.get('max', float('-inf'))
        current_mean = fitness_stats.get('mean', 0.0)
        
        # Update history
        self.fitness_history.append(current_mean)
        self.diversity_history.append(diversity)
        self.best_fitness_history.append(current_best)
        
        # Check for improvement
        if current_best > self.best_fitness_ever + self.improvement_threshold:
            self.best_fitness_ever = current_best
            self.last_improvement_generation = self.generation_count
        
        # Check convergence conditions
        convergence_status = self._check_convergence(
            current_best, current_mean, diversity, fitness_stats
        )
        
        return convergence_status
    
    def _check_convergence(self, current_best: float, current_mean: float,
                          diversity: float, fitness_stats: Dict[str, float]) -> ConvergenceStatus:
        """Check all convergence conditions."""
        
        generations_since_improvement = self.generation_count - self.last_improvement_generation

        # Check for empty population first (critical validation)
        # Note: This check assumes we have access to population through the calling context
        # The diversity=0.000 case typically indicates empty population
        if diversity == 0.0 and current_best == 0.0 and current_mean == 0.0:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.DIVERSITY_LOSS,
                confidence=1.0,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=0,
                details={
                    'message': 'Population appears to be empty - no genomes to evaluate',
                    'troubleshooting': 'Check population initialization in evolution controller'
                }
            )

        # Check target fitness
        if self.target_fitness is not None and current_best >= self.target_fitness:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.TARGET_FITNESS,
                confidence=1.0,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=0,
                details={'target_fitness': self.target_fitness, 'achieved_fitness': current_best}
            )
        
        # Check maximum generations
        if self.generation_count >= self.max_generations:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.MAX_GENERATIONS,
                confidence=1.0,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=0,
                details={'max_generations': self.max_generations}
            )
        
        # Check stagnation
        if generations_since_improvement >= self.stagnation_threshold:
            confidence = min(1.0, generations_since_improvement / self.stagnation_threshold)
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.STAGNATION,
                confidence=confidence,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=generations_since_improvement,
                details={'stagnation_threshold': self.stagnation_threshold}
            )
        
        # Check diversity loss
        if diversity < self.diversity_threshold:
            confidence = 1.0 - (diversity / self.diversity_threshold)
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.DIVERSITY_LOSS,
                confidence=confidence,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=0,
                details={'diversity_threshold': self.diversity_threshold, 'current_diversity': diversity}
            )
        
        # Check fitness plateau
        plateau_length = self._detect_fitness_plateau()
        if plateau_length >= self.fitness_plateau_generations:
            confidence = min(1.0, plateau_length / self.fitness_plateau_generations)
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.FITNESS_PLATEAU,
                confidence=confidence,
                generations_since_improvement=generations_since_improvement,
                current_best_fitness=current_best,
                diversity_score=diversity,
                plateau_length=plateau_length,
                details={'plateau_threshold': self.fitness_plateau_threshold, 'plateau_length': plateau_length}
            )
        
        # Not converged
        return ConvergenceStatus(
            converged=False,
            reason=ConvergenceReason.NOT_CONVERGED,
            confidence=0.0,
            generations_since_improvement=generations_since_improvement,
            current_best_fitness=current_best,
            diversity_score=diversity,
            plateau_length=plateau_length,
            details={}
        )
    
    def _detect_fitness_plateau(self) -> int:
        """
        Detect fitness plateau by analyzing recent fitness history.
        
        Returns:
            Length of current plateau in generations
        """
        if len(self.best_fitness_history) < 2:
            return 0
        
        plateau_length = 0
        current_fitness = self.best_fitness_history[-1]
        
        # Look backwards for plateau
        for i in range(len(self.best_fitness_history) - 2, -1, -1):
            fitness_diff = current_fitness - self.best_fitness_history[i]
            
            if fitness_diff < self.fitness_plateau_threshold:
                plateau_length += 1
            else:
                break
        
        return plateau_length
    
    def get_convergence_prediction(self) -> Dict[str, Any]:
        """
        Predict likelihood of convergence in near future.
        
        Returns:
            Dictionary with convergence predictions
        """
        if len(self.fitness_history) < 5:
            return {'prediction': 'insufficient_data'}
        
        # Analyze trends
        recent_fitness = self.fitness_history[-5:]
        recent_diversity = self.diversity_history[-5:]
        
        # Fitness trend
        fitness_trend = self._calculate_trend(recent_fitness)
        diversity_trend = self._calculate_trend(recent_diversity)
        
        # Predict convergence likelihood
        convergence_likelihood = 0.0
        
        # Fitness stagnation contributes to convergence
        if abs(fitness_trend) < 0.001:
            convergence_likelihood += 0.3
        
        # Diversity loss contributes to convergence
        if diversity_trend < -0.01:
            convergence_likelihood += 0.4
        
        # Recent plateau contributes to convergence
        recent_plateau = self._detect_fitness_plateau()
        if recent_plateau > 0:
            convergence_likelihood += 0.3 * (recent_plateau / self.fitness_plateau_generations)
        
        convergence_likelihood = min(1.0, convergence_likelihood)
        
        return {
            'convergence_likelihood': convergence_likelihood,
            'fitness_trend': fitness_trend,
            'diversity_trend': diversity_trend,
            'recent_plateau_length': recent_plateau,
            'generations_since_improvement': self.generation_count - self.last_improvement_generation,
            'predicted_convergence_in': max(1, int((1 - convergence_likelihood) * 10))
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def reset(self):
        """Reset convergence detector state."""
        self.fitness_history.clear()
        self.diversity_history.clear()
        self.best_fitness_history.clear()
        self.generation_count = 0
        self.last_improvement_generation = 0
        self.best_fitness_ever = float('-inf')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get convergence detector statistics."""
        return {
            'generation_count': self.generation_count,
            'generations_since_improvement': self.generation_count - self.last_improvement_generation,
            'best_fitness_ever': self.best_fitness_ever,
            'fitness_history_length': len(self.fitness_history),
            'diversity_history_length': len(self.diversity_history),
            'current_plateau_length': self._detect_fitness_plateau(),
            'settings': {
                'fitness_plateau_threshold': self.fitness_plateau_threshold,
                'fitness_plateau_generations': self.fitness_plateau_generations,
                'diversity_threshold': self.diversity_threshold,
                'max_generations': self.max_generations,
                'target_fitness': self.target_fitness,
                'stagnation_threshold': self.stagnation_threshold
            }
        }


if __name__ == "__main__":
    # Test convergence detection
    print("Testing convergence detection...")

    import random

    # Load vocabulary for testing
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create test population
    population = Population(population_size=10)
    population.initialize_random(5, 15)
    
    # Create convergence detector
    detector = ConvergenceDetector(
        fitness_plateau_threshold=0.01,
        fitness_plateau_generations=5,
        diversity_threshold=0.1,
        max_generations=20,
        target_fitness=0.95,
        stagnation_threshold=8
    )
    
    print("✅ Convergence detector initialized")
    
    # Simulate evolution with different scenarios
    scenarios = [
        "improving_fitness",
        "plateau",
        "diversity_loss",
        "target_reached"
    ]
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario.upper()} scenario ---")
        detector.reset()
        
        for generation in range(15):
            # Simulate different fitness patterns
            if scenario == "improving_fitness":
                # Steady improvement
                base_fitness = 0.5 + generation * 0.02
                fitnesses = [base_fitness + random.uniform(-0.1, 0.1) for _ in range(10)]
            elif scenario == "plateau":
                # Plateau after initial improvement
                if generation < 5:
                    base_fitness = 0.5 + generation * 0.05
                else:
                    base_fitness = 0.75  # Plateau
                fitnesses = [base_fitness + random.uniform(-0.05, 0.05) for _ in range(10)]
            elif scenario == "diversity_loss":
                # Decreasing diversity
                base_fitness = 0.6
                diversity_factor = max(0.01, 0.2 - generation * 0.015)
                fitnesses = [base_fitness + random.uniform(-diversity_factor, diversity_factor) for _ in range(10)]
            elif scenario == "target_reached":
                # Reach target fitness
                if generation < 8:
                    base_fitness = 0.7 + generation * 0.03
                else:
                    base_fitness = 0.96  # Above target
                fitnesses = [base_fitness + random.uniform(-0.02, 0.02) for _ in range(10)]
            
            # Set fitnesses
            for genome, fitness in zip(population.genomes, fitnesses):
                genome.set_fitness(max(0.0, fitness))
            
            # Update detector
            status = detector.update(population)
            
            if status.converged:
                print(f"✅ Converged at generation {generation + 1}: {status.reason.value}")
                print(f"   Confidence: {status.confidence:.2f}")
                print(f"   Best fitness: {status.current_best_fitness:.3f}")
                print(f"   Diversity: {status.diversity_score:.3f}")
                break
        else:
            print("✅ No convergence detected in 15 generations")
        
        # Test prediction
        prediction = detector.get_convergence_prediction()
        if 'convergence_likelihood' in prediction:
            print(f"   Convergence likelihood: {prediction['convergence_likelihood']:.2f}")
    
    print("\n🎯 Convergence detection tests completed successfully!")
