"""
Generation management for genetic algorithm.
Tracks generations, best genomes, and evolution statistics.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.genetics.genome import PromptGenome
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class GenerationManager:
    """Manages generation tracking and statistics for genetic algorithm."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize generation manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Paths
        self.results_dir = Path(self.config.get('paths.results_dir', './data/results'))
        self.checkpoints_dir = Path(self.config.get('paths.checkpoints_dir', './data/checkpoints'))
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation tracking
        self.generations = []
        self.best_genomes = []
        self.population_stats = []
        
        # Experiment metadata
        self.experiment_start_time = None
        self.experiment_id = None
        self.config_snapshot = None
    
    def start_experiment(self, experiment_id: Optional[str] = None) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_id: Optional experiment ID, auto-generated if None
            
        Returns:
            Experiment ID
        """
        if experiment_id is None:
            timestamp = int(time.time())
            experiment_id = f"gsm8k_ga_{timestamp}"
        
        self.experiment_id = experiment_id
        self.experiment_start_time = time.time()
        self.config_snapshot = self.config.to_dict()
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return experiment_id
    
    def record_generation(self, generation: int, population: List[PromptGenome],
                         generation_stats: Dict[str, Any]) -> None:
        """
        Record generation data.
        
        Args:
            generation: Generation number
            population: Current population
            generation_stats: Generation statistics
        """
        # Calculate population statistics
        evaluated_genomes = [g for g in population if g.fitness is not None]
        
        if evaluated_genomes:
            fitnesses = [g.fitness for g in evaluated_genomes]
            accuracies = [g.accuracy for g in evaluated_genomes if g.accuracy is not None]
            lengths = [len(g.tokens) for g in evaluated_genomes]
            
            pop_stats = {
                'generation': generation,
                'population_size': len(population),
                'evaluated_count': len(evaluated_genomes),
                'fitness': {
                    'mean': sum(fitnesses) / len(fitnesses),
                    'max': max(fitnesses),
                    'min': min(fitnesses),
                    'std': self._calculate_std(fitnesses)
                },
                'length': {
                    'mean': sum(lengths) / len(lengths),
                    'max': max(lengths),
                    'min': min(lengths)
                },
                'timestamp': time.time()
            }
            
            if accuracies:
                pop_stats['accuracy'] = {
                    'mean': sum(accuracies) / len(accuracies),
                    'max': max(accuracies),
                    'min': min(accuracies)
                }
            
            self.population_stats.append(pop_stats)
            
            # Track best genome
            best_genome = max(evaluated_genomes, key=lambda g: g.fitness)
            best_genome_info = {
                'generation': generation,
                'genome_id': best_genome.genome_id,
                'fitness': best_genome.fitness,
                'accuracy': best_genome.accuracy,
                'length': len(best_genome.tokens),
                'text': best_genome.to_text(),
                'tokens': best_genome.tokens.copy(),
                'metadata': best_genome.metadata.copy(),
                'timestamp': time.time()
            }
            
            self.best_genomes.append(best_genome_info)
        
        # Record generation info
        generation_info = {
            'generation': generation,
            'stats': generation_stats,
            'timestamp': time.time()
        }
        
        self.generations.append(generation_info)
        
        logger.debug(f"Recorded generation {generation} data")
    
    def get_best_genome_overall(self) -> Optional[Dict[str, Any]]:
        """Get the best genome across all generations."""
        if not self.best_genomes:
            return None
        
        return max(self.best_genomes, key=lambda x: x['fitness'])
    
    def get_generation_summary(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific generation.
        
        Args:
            generation: Generation number
            
        Returns:
            Generation summary or None if not found
        """
        for gen_info in self.generations:
            if gen_info['generation'] == generation:
                # Find corresponding population stats
                pop_stats = None
                for stats in self.population_stats:
                    if stats['generation'] == generation:
                        pop_stats = stats
                        break
                
                # Find best genome for this generation
                best_genome = None
                for genome_info in self.best_genomes:
                    if genome_info['generation'] == generation:
                        best_genome = genome_info
                        break
                
                return {
                    'generation': generation,
                    'generation_stats': gen_info['stats'],
                    'population_stats': pop_stats,
                    'best_genome': best_genome,
                    'timestamp': gen_info['timestamp']
                }
        
        return None
    
    def get_fitness_progression(self) -> List[Dict[str, Any]]:
        """Get fitness progression across generations."""
        progression = []
        
        for pop_stats in self.population_stats:
            progression.append({
                'generation': pop_stats['generation'],
                'mean_fitness': pop_stats['fitness']['mean'],
                'max_fitness': pop_stats['fitness']['max'],
                'best_accuracy': pop_stats.get('accuracy', {}).get('max', 0),
                'timestamp': pop_stats['timestamp']
            })
        
        return progression
    
    def get_diversity_metrics(self) -> List[Dict[str, Any]]:
        """Get diversity metrics across generations."""
        diversity_metrics = []
        
        for pop_stats in self.population_stats:
            # Calculate diversity based on fitness standard deviation
            fitness_std = pop_stats['fitness']['std']
            fitness_range = pop_stats['fitness']['max'] - pop_stats['fitness']['min']
            
            diversity_metrics.append({
                'generation': pop_stats['generation'],
                'fitness_std': fitness_std,
                'fitness_range': fitness_range,
                'length_range': pop_stats['length']['max'] - pop_stats['length']['min'],
                'timestamp': pop_stats['timestamp']
            })
        
        return diversity_metrics
    
    def save_checkpoint(self, generation: int, population: List[PromptGenome]) -> str:
        """
        Save checkpoint for current generation.
        
        Args:
            generation: Current generation
            population: Current population
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            'experiment_id': self.experiment_id,
            'generation': generation,
            'timestamp': time.time(),
            'population': [genome.to_dict() for genome in population],
            'generations': self.generations,
            'best_genomes': self.best_genomes,
            'population_stats': self.population_stats,
            'config_snapshot': self.config_snapshot
        }
        
        checkpoint_file = self.checkpoints_dir / f"{self.experiment_id}_gen_{generation:03d}.json"
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Saved checkpoint for generation {generation}: {checkpoint_file}")
        
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restore state
        self.experiment_id = checkpoint_data['experiment_id']
        self.generations = checkpoint_data['generations']
        self.best_genomes = checkpoint_data['best_genomes']
        self.population_stats = checkpoint_data['population_stats']
        self.config_snapshot = checkpoint_data['config_snapshot']
        
        logger.info(f"Loaded checkpoint from {checkpoint_file}")
        
        return checkpoint_data
    
    def save_final_results(self) -> str:
        """
        Save final experiment results.
        
        Returns:
            Path to results file
        """
        if not self.experiment_id:
            raise ValueError("No experiment started")
        
        experiment_end_time = time.time()
        total_time = experiment_end_time - self.experiment_start_time if self.experiment_start_time else 0
        
        results = {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'start_time': self.experiment_start_time,
                'end_time': experiment_end_time,
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'config_snapshot': self.config_snapshot
            },
            'evolution_summary': {
                'total_generations': len(self.generations),
                'best_genome_overall': self.get_best_genome_overall(),
                'fitness_progression': self.get_fitness_progression(),
                'diversity_metrics': self.get_diversity_metrics()
            },
            'detailed_results': {
                'generations': self.generations,
                'best_genomes': self.best_genomes,
                'population_stats': self.population_stats
            }
        }
        
        results_file = self.results_dir / f"{self.experiment_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved final results: {results_file}")
        
        return str(results_file)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics."""
        if not self.generations:
            return {'no_data': True}
        
        best_overall = self.get_best_genome_overall()
        fitness_progression = self.get_fitness_progression()
        
        current_time = time.time()
        elapsed_time = current_time - self.experiment_start_time if self.experiment_start_time else 0
        
        summary = {
            'experiment_id': self.experiment_id,
            'generations_completed': len(self.generations),
            'elapsed_time_minutes': elapsed_time / 60,
            'best_fitness_overall': best_overall['fitness'] if best_overall else 0,
            'best_accuracy_overall': best_overall['accuracy'] if best_overall else 0,
            'current_generation': self.generations[-1]['generation'] if self.generations else 0,
            'fitness_improvement': (
                fitness_progression[-1]['max_fitness'] - fitness_progression[0]['max_fitness']
                if len(fitness_progression) > 1 else 0
            ),
            'avg_time_per_generation': elapsed_time / len(self.generations) if self.generations else 0
        }
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
