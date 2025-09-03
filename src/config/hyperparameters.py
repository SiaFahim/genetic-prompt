"""
Centralized hyperparameter configuration for the genetic algorithm system.

This module provides a comprehensive configuration system that centralizes all
genetic algorithm hyperparameters, provides validation, and supports both
programmatic and file-based configuration updates.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""
    pass


@dataclass
class ParameterSpec:
    """Specification for a hyperparameter including validation rules."""
    name: str
    description: str
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[Any]] = None
    parameter_type: type = float
    category: str = "general"
    
    def validate(self, value: Any) -> Any:
        """Validate and convert parameter value."""
        if value is None:
            return self.default_value
            
        # Type conversion
        try:
            if self.parameter_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = bool(value)
            else:
                value = self.parameter_type(value)
        except (ValueError, TypeError) as e:
            raise ParameterValidationError(
                f"Parameter '{self.name}': Cannot convert {value} to {self.parameter_type.__name__}: {e}"
            )
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            raise ParameterValidationError(
                f"Parameter '{self.name}': Value {value} is below minimum {self.min_value}"
            )
        if self.max_value is not None and value > self.max_value:
            raise ParameterValidationError(
                f"Parameter '{self.name}': Value {value} is above maximum {self.max_value}"
            )
        
        # Valid values validation
        if self.valid_values is not None and value not in self.valid_values:
            raise ParameterValidationError(
                f"Parameter '{self.name}': Value {value} not in valid values {self.valid_values}"
            )
        
        return value


@dataclass
class HyperparameterConfig:
    """Centralized configuration for all genetic algorithm hyperparameters."""
    
    # Core Evolution Parameters
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_size: int = 5
    tournament_size: int = 3
    
    # Convergence & Termination
    target_fitness: Optional[float] = 0.85
    convergence_patience: int = 20
    fitness_plateau_threshold: float = 0.001
    fitness_plateau_generations: int = 10
    diversity_threshold: float = 0.05
    stagnation_threshold: int = 20
    improvement_threshold: float = 0.01
    
    # Mutation Parameters
    semantic_prob: float = 0.9
    insertion_rate: float = 0.05
    deletion_rate: float = 0.05
    swap_rate: float = 0.05
    duplication_rate: float = 0.02
    max_insertions: int = 3
    min_genome_length: int = 3
    max_genome_length: int = 50
    
    # Selection Parameters
    selection_pressure: float = 1.5
    elite_ratio: float = 0.1
    
    # Semantic Neighborhoods
    n_neighbors: int = 50
    neighbor_count: int = 5
    
    # Genome Initialization
    min_initial_length: int = 5
    max_initial_length: int = 20
    
    # Evaluation Parameters
    max_problems: int = 100
    batch_size: int = 10
    api_timeout: int = 30
    max_retries: int = 3
    
    # Model Parameters
    temperature: float = 0.0
    max_tokens: int = 150
    
    # System Parameters
    use_cache: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    enable_logging: bool = True
    random_seed: Optional[int] = None
    
    # Performance Parameters
    parallel_evaluation: bool = False
    max_workers: int = 4
    memory_limit_mb: int = 1024

    # Async Evaluation Parameters
    enable_async_evaluation: bool = True
    async_batch_size: int = 20
    max_concurrent_requests: int = 10
    genome_batch_size: int = 10
    max_concurrent_genomes: int = 5
    rate_limit_per_minute: int = 3500
    async_retry_attempts: int = 3
    async_base_delay: float = 1.0
    async_max_delay: float = 60.0
    async_timeout: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate_all()
    
    def validate_all(self) -> None:
        """Validate all parameters using their specifications."""
        specs = self.get_parameter_specs()
        
        for param_name, spec in specs.items():
            current_value = getattr(self, param_name)
            try:
                validated_value = spec.validate(current_value)
                setattr(self, param_name, validated_value)
            except ParameterValidationError as e:
                logger.error(f"Validation failed for {param_name}: {e}")
                raise
    
    @classmethod
    def get_parameter_specs(cls) -> Dict[str, ParameterSpec]:
        """Get parameter specifications for all hyperparameters."""
        return {
            # Core Evolution Parameters
            'population_size': ParameterSpec(
                'population_size', 'Number of genomes in population',
                50, min_value=5, max_value=500, parameter_type=int, category='evolution'
            ),
            'max_generations': ParameterSpec(
                'max_generations', 'Maximum number of generations',
                100, min_value=1, max_value=1000, parameter_type=int, category='evolution'
            ),
            'crossover_rate': ParameterSpec(
                'crossover_rate', 'Probability of crossover operation',
                0.8, min_value=0.0, max_value=1.0, parameter_type=float, category='evolution'
            ),
            'mutation_rate': ParameterSpec(
                'mutation_rate', 'Probability of mutation operation',
                0.2, min_value=0.0, max_value=1.0, parameter_type=float, category='evolution'
            ),
            'elite_size': ParameterSpec(
                'elite_size', 'Number of best genomes to preserve',
                5, min_value=0, max_value=50, parameter_type=int, category='evolution'
            ),
            'tournament_size': ParameterSpec(
                'tournament_size', 'Size of tournament for selection',
                3, min_value=1, max_value=10, parameter_type=int, category='selection'
            ),
            
            # Convergence & Termination
            'target_fitness': ParameterSpec(
                'target_fitness', 'Target fitness for early stopping',
                0.85, min_value=0.0, max_value=1.0, parameter_type=float, category='convergence'
            ),
            'convergence_patience': ParameterSpec(
                'convergence_patience', 'Generations to wait before convergence',
                20, min_value=1, max_value=100, parameter_type=int, category='convergence'
            ),
            'fitness_plateau_threshold': ParameterSpec(
                'fitness_plateau_threshold', 'Minimum fitness improvement to avoid plateau',
                0.001, min_value=0.0, max_value=0.1, parameter_type=float, category='convergence'
            ),
            'fitness_plateau_generations': ParameterSpec(
                'fitness_plateau_generations', 'Generations without improvement for plateau',
                10, min_value=1, max_value=50, parameter_type=int, category='convergence'
            ),
            'diversity_threshold': ParameterSpec(
                'diversity_threshold', 'Minimum diversity to avoid convergence',
                0.05, min_value=0.0, max_value=1.0, parameter_type=float, category='convergence'
            ),
            'stagnation_threshold': ParameterSpec(
                'stagnation_threshold', 'Generations without improvement for stagnation',
                20, min_value=1, max_value=100, parameter_type=int, category='convergence'
            ),
            'improvement_threshold': ParameterSpec(
                'improvement_threshold', 'Minimum improvement to reset stagnation counter',
                0.01, min_value=0.0, max_value=0.1, parameter_type=float, category='convergence'
            ),
            
            # Mutation Parameters
            'semantic_prob': ParameterSpec(
                'semantic_prob', 'Probability of using semantic neighbor vs random token',
                0.9, min_value=0.0, max_value=1.0, parameter_type=float, category='mutation'
            ),
            'insertion_rate': ParameterSpec(
                'insertion_rate', 'Probability of insertion at each position',
                0.05, min_value=0.0, max_value=1.0, parameter_type=float, category='mutation'
            ),
            'deletion_rate': ParameterSpec(
                'deletion_rate', 'Probability of deleting each token',
                0.05, min_value=0.0, max_value=1.0, parameter_type=float, category='mutation'
            ),
            'swap_rate': ParameterSpec(
                'swap_rate', 'Probability of swapping adjacent tokens',
                0.05, min_value=0.0, max_value=1.0, parameter_type=float, category='mutation'
            ),
            'duplication_rate': ParameterSpec(
                'duplication_rate', 'Probability of duplicating tokens',
                0.02, min_value=0.0, max_value=1.0, parameter_type=float, category='mutation'
            ),
            'max_insertions': ParameterSpec(
                'max_insertions', 'Maximum number of insertions per mutation',
                3, min_value=1, max_value=10, parameter_type=int, category='mutation'
            ),
            'min_genome_length': ParameterSpec(
                'min_genome_length', 'Minimum allowed genome length',
                3, min_value=1, max_value=20, parameter_type=int, category='genome'
            ),
            'max_genome_length': ParameterSpec(
                'max_genome_length', 'Maximum allowed genome length',
                50, min_value=10, max_value=200, parameter_type=int, category='genome'
            ),
            
            # Selection Parameters
            'selection_pressure': ParameterSpec(
                'selection_pressure', 'Selection pressure for rank-based selection',
                1.5, min_value=1.0, max_value=3.0, parameter_type=float, category='selection'
            ),
            'elite_ratio': ParameterSpec(
                'elite_ratio', 'Ratio of elite individuals to preserve',
                0.1, min_value=0.0, max_value=0.5, parameter_type=float, category='selection'
            ),
            
            # Semantic Neighborhoods
            'n_neighbors': ParameterSpec(
                'n_neighbors', 'Number of semantic neighbors per token',
                50, min_value=5, max_value=200, parameter_type=int, category='semantic'
            ),
            'neighbor_count': ParameterSpec(
                'neighbor_count', 'Number of neighbors to sample for mutation',
                5, min_value=1, max_value=20, parameter_type=int, category='semantic'
            ),
            
            # Genome Initialization
            'min_initial_length': ParameterSpec(
                'min_initial_length', 'Minimum length for random genome initialization',
                5, min_value=1, max_value=20, parameter_type=int, category='initialization'
            ),
            'max_initial_length': ParameterSpec(
                'max_initial_length', 'Maximum length for random genome initialization',
                20, min_value=5, max_value=100, parameter_type=int, category='initialization'
            ),

            # Evaluation Parameters
            'max_problems': ParameterSpec(
                'max_problems', 'Maximum number of problems for evaluation',
                100, min_value=1, max_value=1000, parameter_type=int, category='evaluation'
            ),
            'batch_size': ParameterSpec(
                'batch_size', 'Batch size for evaluation',
                10, min_value=1, max_value=100, parameter_type=int, category='evaluation'
            ),
            'api_timeout': ParameterSpec(
                'api_timeout', 'API request timeout in seconds',
                30, min_value=5, max_value=300, parameter_type=int, category='evaluation'
            ),
            'max_retries': ParameterSpec(
                'max_retries', 'Maximum number of API retries',
                3, min_value=0, max_value=10, parameter_type=int, category='evaluation'
            ),

            # Model Parameters
            'temperature': ParameterSpec(
                'temperature', 'Model temperature for generation',
                0.0, min_value=0.0, max_value=2.0, parameter_type=float, category='model'
            ),
            'max_tokens': ParameterSpec(
                'max_tokens', 'Maximum tokens for model generation',
                150, min_value=10, max_value=4000, parameter_type=int, category='model'
            ),

            # System Parameters
            'use_cache': ParameterSpec(
                'use_cache', 'Enable evaluation caching',
                True, parameter_type=bool, category='system'
            ),
            'save_checkpoints': ParameterSpec(
                'save_checkpoints', 'Enable checkpoint saving',
                True, parameter_type=bool, category='system'
            ),
            'checkpoint_interval': ParameterSpec(
                'checkpoint_interval', 'Generations between checkpoints',
                10, min_value=1, max_value=100, parameter_type=int, category='system'
            ),
            'enable_logging': ParameterSpec(
                'enable_logging', 'Enable detailed logging',
                True, parameter_type=bool, category='system'
            ),
            'random_seed': ParameterSpec(
                'random_seed', 'Random seed for reproducibility',
                None, min_value=0, max_value=2**32-1, parameter_type=int, category='system'
            ),

            # Performance Parameters
            'parallel_evaluation': ParameterSpec(
                'parallel_evaluation', 'Enable parallel evaluation',
                False, parameter_type=bool, category='performance'
            ),
            'max_workers': ParameterSpec(
                'max_workers', 'Maximum worker threads for parallel evaluation',
                4, min_value=1, max_value=32, parameter_type=int, category='performance'
            ),
            'memory_limit_mb': ParameterSpec(
                'memory_limit_mb', 'Memory limit in megabytes',
                1024, min_value=128, max_value=16384, parameter_type=int, category='performance'
            ),

            # Async Evaluation Parameters
            'enable_async_evaluation': ParameterSpec(
                'enable_async_evaluation', 'Enable asynchronous evaluation pipeline',
                True, parameter_type=bool, category='async'
            ),
            'async_batch_size': ParameterSpec(
                'async_batch_size', 'Batch size for async problem processing',
                20, min_value=1, max_value=100, parameter_type=int, category='async'
            ),
            'max_concurrent_requests': ParameterSpec(
                'max_concurrent_requests', 'Maximum concurrent API requests per batch',
                10, min_value=1, max_value=50, parameter_type=int, category='async'
            ),
            'genome_batch_size': ParameterSpec(
                'genome_batch_size', 'Number of genomes to process concurrently',
                10, min_value=1, max_value=50, parameter_type=int, category='async'
            ),
            'max_concurrent_genomes': ParameterSpec(
                'max_concurrent_genomes', 'Maximum concurrent genome evaluations',
                5, min_value=1, max_value=20, parameter_type=int, category='async'
            ),
            'rate_limit_per_minute': ParameterSpec(
                'rate_limit_per_minute', 'API rate limit per minute',
                3500, min_value=100, max_value=10000, parameter_type=int, category='async'
            ),
            'async_retry_attempts': ParameterSpec(
                'async_retry_attempts', 'Number of retry attempts for failed requests',
                3, min_value=1, max_value=10, parameter_type=int, category='async'
            ),
            'async_base_delay': ParameterSpec(
                'async_base_delay', 'Base delay for exponential backoff (seconds)',
                1.0, min_value=0.1, max_value=10.0, parameter_type=float, category='async'
            ),
            'async_max_delay': ParameterSpec(
                'async_max_delay', 'Maximum delay for exponential backoff (seconds)',
                60.0, min_value=1.0, max_value=300.0, parameter_type=float, category='async'
            ),
            'async_timeout': ParameterSpec(
                'async_timeout', 'Timeout for async API requests (seconds)',
                30, min_value=5, max_value=120, parameter_type=int, category='async'
            ),
        }
    
    def update_parameters(self, updates: Dict[str, Any]) -> None:
        """Update multiple parameters with validation."""
        specs = self.get_parameter_specs()
        
        for param_name, value in updates.items():
            if param_name not in specs:
                logger.warning(f"Unknown parameter: {param_name}")
                continue
                
            spec = specs[param_name]
            try:
                validated_value = spec.validate(value)
                setattr(self, param_name, validated_value)
                logger.info(f"Updated {param_name}: {value} -> {validated_value}")
            except ParameterValidationError as e:
                logger.error(f"Failed to update {param_name}: {e}")
                raise
    
    def get_parameter_info(self, param_name: str) -> Optional[ParameterSpec]:
        """Get information about a specific parameter."""
        specs = self.get_parameter_specs()
        return specs.get(param_name)
    
    def get_parameters_by_category(self, category: str) -> Dict[str, Any]:
        """Get all parameters in a specific category."""
        specs = self.get_parameter_specs()
        result = {}
        
        for param_name, spec in specs.items():
            if spec.category == category:
                result[param_name] = getattr(self, param_name)
        
        return result
    
    def get_all_categories(self) -> List[str]:
        """Get all parameter categories."""
        specs = self.get_parameter_specs()
        categories = set(spec.category for spec in specs.values())
        return sorted(categories)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_info in self.__dataclass_fields__.values():
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterConfig':
        """Create configuration from dictionary."""
        # Filter out unknown parameters
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'HyperparameterConfig':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls.from_dict(data)
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def get_modified_parameters(self, reference_config: 'HyperparameterConfig') -> Dict[str, Tuple[Any, Any]]:
        """Get parameters that differ from a reference configuration."""
        modified = {}
        
        for field_name in self.__dataclass_fields__.keys():
            current_value = getattr(self, field_name)
            reference_value = getattr(reference_config, field_name)
            
            if current_value != reference_value:
                modified[field_name] = (reference_value, current_value)
        
        return modified


# Global configuration instance
_global_config: Optional[HyperparameterConfig] = None


def get_hyperparameter_config() -> HyperparameterConfig:
    """Get the global hyperparameter configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = HyperparameterConfig()
    return _global_config


def set_hyperparameter_config(config: HyperparameterConfig) -> None:
    """Set the global hyperparameter configuration instance."""
    global _global_config
    _global_config = config


def update_hyperparameters(updates: Dict[str, Any]) -> None:
    """Update the global hyperparameter configuration."""
    config = get_hyperparameter_config()
    config.update_parameters(updates)


def reset_hyperparameters() -> None:
    """Reset hyperparameters to default values."""
    global _global_config
    _global_config = HyperparameterConfig()
