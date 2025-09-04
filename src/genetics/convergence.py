"""
Convergence detection for genetic algorithm.
Monitors fitness improvement and detects stagnation.
"""

from typing import List, Dict, Any, Optional
import logging

from src.genetics.genome import PromptGenome
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """Detects convergence and stagnation in genetic algorithm evolution."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize convergence detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Configuration
        ga_config = self.config.get('genetic_algorithm', {})
        self.convergence_threshold = ga_config.get('convergence_threshold', 0.85)
        self.stagnation_generations = ga_config.get('stagnation_generations', 5)
        self.stagnation_improvement_threshold = ga_config.get('stagnation_improvement_threshold', 0.001)
        self.target_accuracy = self.config.get('experiment.target_accuracy', 0.95)

        # Convergence requires sustained performance over multiple generations
        self.convergence_patience = ga_config.get('convergence_patience', 3)  # Require 3 consecutive generations above target
        self.consecutive_high_accuracy_count = 0
        
        # Tracking
        self.fitness_history = []
        self.best_fitness_history = []
        self.accuracy_history = []
        self.generation_count = 0
        
        # State
        self.is_converged = False
        self.is_stagnant = False
        self.convergence_generation = None
        self.stagnation_start_generation = None
    
    def update(self, population: List[PromptGenome], generation: int) -> None:
        """
        Update convergence detector with new generation data.
        
        Args:
            population: Current population
            generation: Current generation number
        """
        self.generation_count = generation
        
        # Get fitness statistics
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if not evaluated_genomes:
            logger.warning(f"No evaluated genomes in generation {generation}")
            return
        
        fitnesses = [g.fitness for g in evaluated_genomes]
        accuracies = [g.accuracy for g in evaluated_genomes if g.accuracy is not None]
        
        # Calculate statistics
        mean_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness = max(fitnesses)
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        best_accuracy = max(accuracies) if accuracies else 0
        
        # Update history
        self.fitness_history.append(mean_fitness)
        self.best_fitness_history.append(best_fitness)
        self.accuracy_history.append(best_accuracy)
        
        # Check for convergence
        self._check_convergence(best_accuracy)
        
        # Check for stagnation
        self._check_stagnation()
        
        logger.debug(f"Generation {generation}: mean_fitness={mean_fitness:.4f}, "
                    f"best_fitness={best_fitness:.4f}, best_accuracy={best_accuracy:.4f}")
    
    def _check_convergence(self, best_accuracy: float) -> None:
        """
        Check if algorithm has converged based on accuracy threshold.
        Requires sustained high performance over multiple generations.

        Args:
            best_accuracy: Best accuracy in current generation
        """
        if not self.is_converged:
            if best_accuracy >= self.target_accuracy:
                self.consecutive_high_accuracy_count += 1
                logger.debug(f"High accuracy achieved: {best_accuracy:.4f} "
                           f"({self.consecutive_high_accuracy_count}/{self.convergence_patience} consecutive)")

                if self.consecutive_high_accuracy_count >= self.convergence_patience:
                    self.is_converged = True
                    self.convergence_generation = self.generation_count - self.convergence_patience + 1
                    logger.info(f"Convergence achieved! Sustained {best_accuracy:.4f} accuracy "
                               f"for {self.convergence_patience} consecutive generations "
                               f"(starting at generation {self.convergence_generation})")
            else:
                # Reset counter if accuracy drops below target
                if self.consecutive_high_accuracy_count > 0:
                    logger.debug(f"Accuracy dropped below target: {best_accuracy:.4f} < {self.target_accuracy:.4f}. "
                               f"Resetting convergence counter.")
                self.consecutive_high_accuracy_count = 0
    
    def _check_stagnation(self) -> None:
        """Check if algorithm has stagnated based on fitness improvement."""
        if len(self.best_fitness_history) < self.stagnation_generations:
            return
        
        # Get recent fitness history
        recent_fitness = self.best_fitness_history[-self.stagnation_generations:]
        
        # Check if improvement is below threshold
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        if fitness_improvement < self.stagnation_improvement_threshold:
            if not self.is_stagnant:
                self.is_stagnant = True
                self.stagnation_start_generation = self.generation_count - self.stagnation_generations + 1
                logger.info(f"Stagnation detected at generation {self.generation_count}! "
                           f"Improvement over last {self.stagnation_generations} generations: "
                           f"{fitness_improvement:.6f}")
        else:
            # Reset stagnation if improvement is detected
            if self.is_stagnant:
                logger.info(f"Stagnation ended at generation {self.generation_count}! "
                           f"Improvement: {fitness_improvement:.6f}")
                self.is_stagnant = False
                self.stagnation_start_generation = None
    
    def should_terminate(self, max_generations: int) -> bool:
        """
        Determine if evolution should terminate.
        
        Args:
            max_generations: Maximum allowed generations
            
        Returns:
            True if evolution should terminate
        """
        # Terminate if converged
        if self.is_converged:
            return True
        
        # Terminate if max generations reached
        if self.generation_count >= max_generations:
            return True
        
        return False
    
    def get_termination_reason(self, max_generations: int) -> str:
        """
        Get reason for termination.
        
        Args:
            max_generations: Maximum allowed generations
            
        Returns:
            Termination reason string
        """
        if self.is_converged:
            return f"Convergence achieved at generation {self.convergence_generation}"
        elif self.generation_count >= max_generations:
            return f"Maximum generations ({max_generations}) reached"
        else:
            return "Evolution ongoing"
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """Get current convergence status."""
        if not self.best_fitness_history:
            return {'status': 'no_data'}
        
        current_best_fitness = self.best_fitness_history[-1]
        current_best_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0
        
        # Calculate improvement trends
        improvement_trend = None
        if len(self.best_fitness_history) >= 2:
            recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-2]
            improvement_trend = 'improving' if recent_improvement > 0.001 else \
                              'declining' if recent_improvement < -0.001 else 'stable'
        
        # Calculate progress toward target
        progress_to_target = current_best_accuracy / self.target_accuracy if self.target_accuracy > 0 else 0
        
        status = {
            'generation': self.generation_count,
            'is_converged': self.is_converged,
            'is_stagnant': self.is_stagnant,
            'convergence_generation': self.convergence_generation,
            'stagnation_start_generation': self.stagnation_start_generation,
            'current_best_fitness': current_best_fitness,
            'current_best_accuracy': current_best_accuracy,
            'target_accuracy': self.target_accuracy,
            'progress_to_target': progress_to_target,
            'improvement_trend': improvement_trend,
            'generations_since_stagnation': (
                self.generation_count - self.stagnation_start_generation 
                if self.stagnation_start_generation is not None else 0
            )
        }
        
        return status
    
    def get_fitness_trends(self, window_size: int = 5) -> Dict[str, Any]:
        """
        Get fitness trend analysis.
        
        Args:
            window_size: Size of moving window for trend analysis
            
        Returns:
            Dictionary with trend statistics
        """
        if len(self.best_fitness_history) < window_size:
            return {'insufficient_data': True}
        
        # Calculate moving averages
        recent_window = self.best_fitness_history[-window_size:]
        older_window = self.best_fitness_history[-2*window_size:-window_size] if len(self.best_fitness_history) >= 2*window_size else []
        
        recent_avg = sum(recent_window) / len(recent_window)
        older_avg = sum(older_window) / len(older_window) if older_window else recent_avg
        
        # Calculate trend metrics
        trend_direction = recent_avg - older_avg
        trend_strength = abs(trend_direction)
        
        # Calculate volatility (standard deviation of recent fitness)
        if len(recent_window) > 1:
            mean_recent = sum(recent_window) / len(recent_window)
            variance = sum((f - mean_recent) ** 2 for f in recent_window) / len(recent_window)
            volatility = variance ** 0.5
        else:
            volatility = 0
        
        trends = {
            'recent_average': recent_avg,
            'older_average': older_avg,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'is_improving': trend_direction > 0.001,
            'is_stable': abs(trend_direction) <= 0.001,
            'is_declining': trend_direction < -0.001
        }
        
        return trends
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive convergence statistics."""
        if not self.fitness_history:
            return {'no_data': True}
        
        stats = {
            'generations_completed': self.generation_count,
            'total_fitness_history_length': len(self.fitness_history),
            'convergence_status': self.get_convergence_status(),
            'fitness_trends': self.get_fitness_trends(),
            'fitness_statistics': {
                'current_mean': self.fitness_history[-1],
                'current_best': self.best_fitness_history[-1],
                'all_time_best': max(self.best_fitness_history),
                'improvement_from_start': (
                    self.best_fitness_history[-1] - self.best_fitness_history[0]
                    if len(self.best_fitness_history) > 1 else 0
                )
            }
        }
        
        return stats
    
    def reset(self) -> None:
        """Reset convergence detector state."""
        self.fitness_history.clear()
        self.best_fitness_history.clear()
        self.accuracy_history.clear()
        self.generation_count = 0
        self.is_converged = False
        self.is_stagnant = False
        self.convergence_generation = None
        self.stagnation_start_generation = None
        self.consecutive_high_accuracy_count = 0

        logger.info("Convergence detector reset")
    
    def save_history(self, filepath: str) -> None:
        """
        Save fitness history to file.
        
        Args:
            filepath: Path to save history
        """
        import json
        
        history_data = {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'accuracy_history': self.accuracy_history,
            'generation_count': self.generation_count,
            'convergence_status': self.get_convergence_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Convergence history saved to {filepath}")
