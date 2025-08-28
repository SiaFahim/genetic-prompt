"""
Experiment management system for genetic algorithm evolution.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.utils.evolution_logging import EvolutionLogger
    from src.utils.visualization import EvolutionVisualizer
    from src.genetics.evolution import EvolutionConfig
else:
    from .config import config
    from .evolution_logging import EvolutionLogger
    from .visualization import EvolutionVisualizer
    from ..genetics.evolution import EvolutionConfig


def _convert_to_json_serializable(obj):
    """Convert objects to JSON-serializable format, handling enums and dataclasses."""
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dataclass_fields__'):
        # It's a dataclass
        result = {}
        for field_name in obj.__dataclass_fields__:
            field_value = getattr(obj, field_name)
            result[field_name] = _convert_to_json_serializable(field_value)
        return result
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        # Basic types (int, float, str, bool, None)
        return obj


@dataclass
class ExperimentMetadata:
    """Metadata for an evolution experiment."""
    experiment_id: str
    experiment_name: str
    description: str
    created_at: float
    status: str  # 'created', 'running', 'completed', 'failed', 'cancelled'
    config: Dict[str, Any]
    results_dir: str
    logs_dir: str
    plots_dir: str
    total_generations: int = 0
    best_fitness: float = 0.0
    convergence_reason: str = ""
    total_time: float = 0.0
    completed_at: Optional[float] = None


class ExperimentManager:
    """Manages multiple evolution experiments with unique IDs and organization."""
    
    def __init__(self):
        """Initialize experiment manager."""
        # Create experiments directory structure
        self.experiments_dir = config.get_data_dir() / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.experiments_dir / "experiments_metadata.json"
        
        # Load existing experiments
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self._load_experiments()
    
    def create_experiment(self, 
                         experiment_name: str,
                         description: str,
                         evolution_config: EvolutionConfig,
                         experiment_id: Optional[str] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Human-readable name for the experiment
            description: Description of the experiment
            evolution_config: Evolution configuration
            experiment_id: Optional custom experiment ID
            
        Returns:
            Experiment ID
        """
        # Generate unique experiment ID
        if experiment_id is None:
            experiment_id = f"{experiment_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Create experiment directories
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        results_dir = exp_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        logs_dir = exp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create experiment metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            created_at=time.time(),
            status='created',
            config=self._serialize_config(evolution_config),
            results_dir=str(results_dir),
            logs_dir=str(logs_dir),
            plots_dir=str(plots_dir)
        )
        
        # Store experiment
        self.experiments[experiment_id] = metadata
        self._save_experiments()
        
        # Save experiment config
        config_file = exp_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            serializable_metadata = _convert_to_json_serializable(asdict(metadata))
            json.dump(serializable_metadata, f, indent=2)
        
        print(f"📋 Created experiment: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> tuple[EvolutionLogger, EvolutionVisualizer]:
        """
        Start an experiment and return logger and visualizer.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Tuple of (logger, visualizer)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata = self.experiments[experiment_id]
        metadata.status = 'running'
        self._save_experiments()
        
        # Create logger and visualizer
        logger = EvolutionLogger(experiment_id)
        visualizer = EvolutionVisualizer(experiment_id, save_plots=True)
        
        print(f"🚀 Started experiment: {experiment_id}")
        return logger, visualizer
    
    def complete_experiment(self, experiment_id: str, final_results: Dict[str, Any]):
        """
        Mark experiment as completed and save final results.
        
        Args:
            experiment_id: Experiment ID
            final_results: Final experiment results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata = self.experiments[experiment_id]
        metadata.status = 'completed'
        metadata.completed_at = time.time()
        metadata.total_generations = final_results.get('total_generations', 0)
        metadata.best_fitness = final_results.get('best_fitness', 0.0)
        metadata.convergence_reason = final_results.get('convergence_reason', '')
        metadata.total_time = final_results.get('total_time', 0.0)
        
        self._save_experiments()
        
        # Save final results
        results_file = Path(metadata.results_dir) / "final_results.json"
        with open(results_file, 'w') as f:
            serializable_results = _convert_to_json_serializable(final_results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Completed experiment: {experiment_id}")
    
    def fail_experiment(self, experiment_id: str, error_message: str):
        """
        Mark experiment as failed.
        
        Args:
            experiment_id: Experiment ID
            error_message: Error message
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata = self.experiments[experiment_id]
        metadata.status = 'failed'
        metadata.completed_at = time.time()
        
        self._save_experiments()
        
        # Save error information
        error_file = Path(metadata.results_dir) / "error.json"
        with open(error_file, 'w') as f:
            json.dump({
                'error_message': error_message,
                'failed_at': time.time()
            }, f, indent=2)
        
        print(f"❌ Failed experiment: {experiment_id}")
    
    def cancel_experiment(self, experiment_id: str):
        """
        Cancel a running experiment.
        
        Args:
            experiment_id: Experiment ID
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata = self.experiments[experiment_id]
        metadata.status = 'cancelled'
        metadata.completed_at = time.time()
        
        self._save_experiments()
        print(f"🛑 Cancelled experiment: {experiment_id}")
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[ExperimentMetadata]:
        """
        List experiments with optional status filter.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of experiment metadata
        """
        experiments = list(self.experiments.values())
        
        if status_filter:
            experiments = [exp for exp in experiments if exp.status == status_filter]
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """
        Get experiment metadata by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment metadata or None
        """
        return self.experiments.get(experiment_id)
    
    def delete_experiment(self, experiment_id: str, confirm: bool = False):
        """
        Delete an experiment and all its data.
        
        Args:
            experiment_id: Experiment ID
            confirm: Confirmation flag for safety
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete experiment")
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata = self.experiments[experiment_id]
        
        # Remove experiment directory
        exp_dir = self.experiments_dir / experiment_id
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
        
        # Remove from metadata
        del self.experiments[experiment_id]
        self._save_experiments()
        
        print(f"🗑️  Deleted experiment: {experiment_id}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all experiments."""
        total_experiments = len(self.experiments)
        status_counts = {}
        
        for exp in self.experiments.values():
            status_counts[exp.status] = status_counts.get(exp.status, 0) + 1
        
        completed_experiments = [exp for exp in self.experiments.values() 
                               if exp.status == 'completed']
        
        if completed_experiments:
            avg_fitness = sum(exp.best_fitness for exp in completed_experiments) / len(completed_experiments)
            avg_generations = sum(exp.total_generations for exp in completed_experiments) / len(completed_experiments)
            avg_time = sum(exp.total_time for exp in completed_experiments) / len(completed_experiments)
        else:
            avg_fitness = avg_generations = avg_time = 0
        
        return {
            'total_experiments': total_experiments,
            'status_counts': status_counts,
            'completed_experiments': len(completed_experiments),
            'average_best_fitness': avg_fitness,
            'average_generations': avg_generations,
            'average_time': avg_time,
            'experiments_directory': str(self.experiments_dir)
        }
    
    def _serialize_config(self, config: EvolutionConfig) -> Dict[str, Any]:
        """Serialize evolution config to JSON-compatible format."""
        config_dict = asdict(config)
        
        # Convert enums to strings
        for key, value in config_dict.items():
            if hasattr(value, 'value'):  # Enum
                config_dict[key] = value.value
        
        return config_dict
    
    def _load_experiments(self):
        """Load experiments from metadata file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for exp_id, exp_data in data.items():
                    self.experiments[exp_id] = ExperimentMetadata(**exp_data)
                
            except Exception as e:
                print(f"Warning: Could not load experiments metadata: {e}")
    
    def _save_experiments(self):
        """Save experiments to metadata file."""
        try:
            data = {exp_id: _convert_to_json_serializable(asdict(metadata))
                   for exp_id, metadata in self.experiments.items()}

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save experiments metadata: {e}")


if __name__ == "__main__":
    # Test experiment manager
    print("Testing experiment manager...")
    
    # Create manager
    manager = ExperimentManager()
    
    # Create test experiment
    from src.genetics.evolution import EvolutionConfig
    test_config = EvolutionConfig(
        population_size=20,
        max_generations=50,
        target_fitness=0.8
    )
    
    exp_id = manager.create_experiment(
        experiment_name="test_experiment",
        description="Test experiment for validation",
        evolution_config=test_config
    )
    print(f"✅ Created experiment: {exp_id}")
    
    # Start experiment
    logger, visualizer = manager.start_experiment(exp_id)
    print("✅ Started experiment")
    
    # Simulate completion
    final_results = {
        'best_fitness': 0.85,
        'total_generations': 25,
        'convergence_reason': 'target_reached',
        'total_time': 120.5
    }
    manager.complete_experiment(exp_id, final_results)
    print("✅ Completed experiment")
    
    # List experiments
    experiments = manager.list_experiments()
    print(f"✅ Found {len(experiments)} experiments")
    
    # Get summary
    summary = manager.get_experiment_summary()
    print(f"✅ Experiment summary: {summary}")
    
    print("\n🎯 Experiment manager tests completed successfully!")
