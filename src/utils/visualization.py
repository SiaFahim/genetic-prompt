"""
Progress visualization for genetic algorithm evolution.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
else:
    from .config import config


class EvolutionVisualizer:
    """Real-time visualization for evolution progress."""
    
    def __init__(self, experiment_name: str, save_plots: bool = True):
        """
        Initialize evolution visualizer.
        
        Args:
            experiment_name: Name of the experiment
            save_plots: Whether to save plots to files
        """
        self.experiment_name = experiment_name
        self.save_plots = save_plots
        
        # Create plots directory
        if save_plots:
            self.plots_dir = config.get_data_dir() / "plots" / experiment_name
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []
        self.diversity_scores = []
        self.evaluation_times = []
        self.convergence_events = []
        
        # Plot configuration
        plt.style.use('default')
        self.colors = {
            'best': '#2E8B57',      # Sea Green
            'mean': '#4169E1',      # Royal Blue
            'diversity': '#FF6347',  # Tomato
            'evaluation': '#9370DB', # Medium Purple
            'convergence': '#FF4500' # Orange Red
        }
        
        # Initialize plots
        self.fig = None
        self.axes = None
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up the plot layout."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Evolution Progress: {self.experiment_name}', fontsize=16)
        
        # Fitness evolution plot
        self.axes[0, 0].set_title('Fitness Evolution')
        self.axes[0, 0].set_xlabel('Generation')
        self.axes[0, 0].set_ylabel('Fitness')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend(['Best Fitness', 'Mean Fitness'])
        
        # Diversity plot
        self.axes[0, 1].set_title('Population Diversity')
        self.axes[0, 1].set_xlabel('Generation')
        self.axes[0, 1].set_ylabel('Diversity Score')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation time plot
        self.axes[1, 0].set_title('Evaluation Performance')
        self.axes[1, 0].set_xlabel('Generation')
        self.axes[1, 0].set_ylabel('Evaluation Time (s)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence monitoring plot
        self.axes[1, 1].set_title('Convergence Monitoring')
        self.axes[1, 1].set_xlabel('Generation')
        self.axes[1, 1].set_ylabel('Fitness Improvement')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_data(self, generation: int, result: Dict[str, Any]):
        """
        Update visualization data with new generation results.
        
        Args:
            generation: Generation number
            result: Generation result dictionary
        """
        self.generations.append(generation)
        self.best_fitness.append(result.get('best_fitness', 0))
        self.mean_fitness.append(result.get('mean_fitness', 0))
        self.diversity_scores.append(result.get('diversity', 0))
        self.evaluation_times.append(result.get('evaluation_time', 0))
        
        # Check for convergence events
        convergence_status = result.get('convergence_status')
        if convergence_status and convergence_status.get('converged', False):
            self.convergence_events.append({
                'generation': generation,
                'reason': convergence_status.get('reason', 'unknown')
            })
    
    def update_plots(self):
        """Update all plots with current data."""
        if not self.generations:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Fitness evolution plot
        self.axes[0, 0].plot(self.generations, self.best_fitness, 
                           color=self.colors['best'], linewidth=2, label='Best Fitness')
        self.axes[0, 0].plot(self.generations, self.mean_fitness, 
                           color=self.colors['mean'], linewidth=2, label='Mean Fitness')
        self.axes[0, 0].set_title('Fitness Evolution')
        self.axes[0, 0].set_xlabel('Generation')
        self.axes[0, 0].set_ylabel('Fitness')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend()
        
        # Add convergence markers
        for event in self.convergence_events:
            self.axes[0, 0].axvline(x=event['generation'], 
                                  color=self.colors['convergence'], 
                                  linestyle='--', alpha=0.7,
                                  label=f"Converged: {event['reason']}")
        
        # Diversity plot
        self.axes[0, 1].plot(self.generations, self.diversity_scores, 
                           color=self.colors['diversity'], linewidth=2)
        self.axes[0, 1].set_title('Population Diversity')
        self.axes[0, 1].set_xlabel('Generation')
        self.axes[0, 1].set_ylabel('Diversity Score')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation time plot
        if len(self.evaluation_times) > 1:
            # Show moving average for smoother visualization
            window_size = min(5, len(self.evaluation_times))
            moving_avg = np.convolve(self.evaluation_times, 
                                   np.ones(window_size)/window_size, mode='valid')
            avg_generations = self.generations[window_size-1:]
            
            self.axes[1, 0].plot(self.generations, self.evaluation_times, 
                               color=self.colors['evaluation'], alpha=0.3, label='Raw')
            self.axes[1, 0].plot(avg_generations, moving_avg, 
                               color=self.colors['evaluation'], linewidth=2, label='Moving Avg')
            self.axes[1, 0].legend()
        else:
            self.axes[1, 0].plot(self.generations, self.evaluation_times, 
                               color=self.colors['evaluation'], linewidth=2)
        
        self.axes[1, 0].set_title('Evaluation Performance')
        self.axes[1, 0].set_xlabel('Generation')
        self.axes[1, 0].set_ylabel('Evaluation Time (s)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence monitoring (fitness improvement)
        if len(self.best_fitness) > 1:
            improvements = np.diff(self.best_fitness)
            improvement_generations = self.generations[1:]
            
            self.axes[1, 1].bar(improvement_generations, improvements, 
                              color=self.colors['best'], alpha=0.7)
            self.axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        self.axes[1, 1].set_title('Fitness Improvement per Generation')
        self.axes[1, 1].set_xlabel('Generation')
        self.axes[1, 1].set_ylabel('Fitness Change')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if enabled
        if self.save_plots:
            plot_file = self.plots_dir / f"evolution_progress_gen_{max(self.generations):04d}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    
    def create_fitness_distribution_plot(self, population_fitness: List[float], 
                                       generation: int) -> str:
        """
        Create fitness distribution plot for current population.
        
        Args:
            population_fitness: List of fitness values for current population
            generation: Current generation number
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(population_fitness, bins=20, alpha=0.7, color=self.colors['best'], 
                edgecolor='black')
        
        # Statistics
        mean_fitness = np.mean(population_fitness)
        std_fitness = np.std(population_fitness)
        
        ax.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_fitness:.3f}')
        ax.axvline(mean_fitness + std_fitness, color='orange', linestyle=':', 
                  label=f'+1 STD: {mean_fitness + std_fitness:.3f}')
        ax.axvline(mean_fitness - std_fitness, color='orange', linestyle=':', 
                  label=f'-1 STD: {mean_fitness - std_fitness:.3f}')
        
        ax.set_title(f'Fitness Distribution - Generation {generation}')
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if self.save_plots:
            plot_file = self.plots_dir / f"fitness_distribution_gen_{generation:04d}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        plt.close()
        return ""
    
    def create_convergence_analysis_plot(self) -> str:
        """
        Create detailed convergence analysis plot.
        
        Returns:
            Path to saved plot file
        """
        if len(self.best_fitness) < 5:
            return ""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Fitness plateau detection
        plateau_threshold = 0.001
        plateau_generations = []
        
        for i in range(1, len(self.best_fitness)):
            improvement = self.best_fitness[i] - self.best_fitness[i-1]
            if improvement < plateau_threshold:
                plateau_generations.append(self.generations[i])
        
        # Top plot: Fitness with plateau markers
        axes[0].plot(self.generations, self.best_fitness, 
                    color=self.colors['best'], linewidth=2, label='Best Fitness')
        
        for gen in plateau_generations:
            axes[0].axvline(x=gen, color='red', alpha=0.3, linestyle='-')
        
        axes[0].set_title('Fitness Evolution with Plateau Detection')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Best Fitness')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Bottom plot: Improvement rate
        if len(self.best_fitness) > 1:
            improvements = np.diff(self.best_fitness)
            improvement_generations = self.generations[1:]
            
            axes[1].plot(improvement_generations, improvements, 
                        color=self.colors['mean'], linewidth=2, marker='o', markersize=4)
            axes[1].axhline(y=plateau_threshold, color='red', linestyle='--', 
                          label=f'Plateau Threshold: {plateau_threshold}')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        axes[1].set_title('Fitness Improvement Rate')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Fitness Improvement')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        if self.save_plots:
            plot_file = self.plots_dir / "convergence_analysis.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            return str(plot_file)
        
        plt.close()
        return ""
    
    def show_plots(self):
        """Display plots in interactive mode."""
        if self.fig:
            plt.show()
    
    def save_final_plots(self):
        """Save final summary plots."""
        if not self.save_plots or not self.generations:
            return
        
        # Update and save main progress plot
        self.update_plots()
        final_plot = self.plots_dir / "final_evolution_progress.png"
        plt.savefig(final_plot, dpi=300, bbox_inches='tight')
        
        # Create convergence analysis
        self.create_convergence_analysis_plot()
        
        print(f"📊 Final plots saved to: {self.plots_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get visualization statistics."""
        if not self.generations:
            return {'no_data': True}
        
        return {
            'total_generations': len(self.generations),
            'best_fitness_final': self.best_fitness[-1] if self.best_fitness else 0,
            'best_fitness_peak': max(self.best_fitness) if self.best_fitness else 0,
            'mean_diversity': np.mean(self.diversity_scores) if self.diversity_scores else 0,
            'total_convergence_events': len(self.convergence_events),
            'plots_directory': str(self.plots_dir) if self.save_plots else None
        }


if __name__ == "__main__":
    # Test visualization system
    print("Testing evolution visualization system...")
    
    # Create test visualizer
    visualizer = EvolutionVisualizer("test_visualization", save_plots=True)
    
    # Generate test data
    import random
    random.seed(42)
    
    for generation in range(1, 21):
        # Simulate evolution data
        best_fitness = min(0.95, 0.3 + generation * 0.03 + random.uniform(-0.02, 0.02))
        mean_fitness = best_fitness - random.uniform(0.1, 0.2)
        diversity = max(0.1, 0.9 - generation * 0.03 + random.uniform(-0.05, 0.05))
        eval_time = 30 + random.uniform(-5, 10)
        
        result = {
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'diversity': diversity,
            'evaluation_time': eval_time
        }
        
        # Add convergence event
        if generation == 15:
            result['convergence_status'] = {
                'converged': True,
                'reason': 'fitness_plateau'
            }
        
        visualizer.update_data(generation, result)
    
    print("✅ Test data generated")
    
    # Update plots
    visualizer.update_plots()
    print("✅ Plots updated")
    
    # Create fitness distribution plot
    test_fitness = [random.uniform(0.4, 0.9) for _ in range(50)]
    dist_plot = visualizer.create_fitness_distribution_plot(test_fitness, 10)
    print(f"✅ Fitness distribution plot: {dist_plot}")
    
    # Create convergence analysis
    conv_plot = visualizer.create_convergence_analysis_plot()
    print(f"✅ Convergence analysis plot: {conv_plot}")
    
    # Save final plots
    visualizer.save_final_plots()
    print("✅ Final plots saved")
    
    # Get statistics
    stats = visualizer.get_statistics()
    print(f"✅ Visualization statistics: {stats}")
    
    print("\n🎯 Visualization system tests completed successfully!")
