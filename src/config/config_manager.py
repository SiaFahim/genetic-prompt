"""
Configuration manager for loading and managing hyperparameter configurations.

This module provides utilities for loading configurations from various sources,
validating them, and providing easy access throughout the application.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import asdict

from .hyperparameters import HyperparameterConfig, ParameterValidationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages hyperparameter configurations with validation and persistence."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        if config_dir is None:
            # Default to project config directory
            from ..utils.config import config
            config_dir = config.project_root / "configs"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_config: Optional[HyperparameterConfig] = None
        self._config_history: List[HyperparameterConfig] = []
    
    def load_config(self, source: Union[str, Path, Dict[str, Any]]) -> HyperparameterConfig:
        """
        Load configuration from various sources.
        
        Args:
            source: Configuration source (file path, dict, or preset name)
            
        Returns:
            Loaded and validated configuration
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            # Check if it's a preset name
            if not source_path.exists() and not source_path.is_absolute():
                preset_path = self.config_dir / f"{source}.json"
                if preset_path.exists():
                    source_path = preset_path
                else:
                    # Try to load as preset
                    return self.load_preset(source)
            
            # Load from file
            if source_path.suffix.lower() == '.json':
                with open(source_path, 'r') as f:
                    data = json.load(f)
            elif source_path.suffix.lower() in ['.yml', '.yaml']:
                with open(source_path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")
            
            logger.info(f"Loaded configuration from {source_path}")
            
        elif isinstance(source, dict):
            data = source
            logger.info("Loaded configuration from dictionary")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Create and validate configuration
        try:
            config = HyperparameterConfig.from_dict(data)
            self._current_config = config
            self._config_history.append(config)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def load_preset(self, preset_name: str) -> HyperparameterConfig:
        """Load a predefined configuration preset."""
        presets = self.get_available_presets()
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        config = presets[preset_name]
        self._current_config = config
        self._config_history.append(config)
        
        logger.info(f"Loaded preset configuration: {preset_name}")
        return config
    
    def get_available_presets(self) -> Dict[str, HyperparameterConfig]:
        """Get all available configuration presets."""
        presets = {}
        
        # Default preset
        presets['default'] = HyperparameterConfig()
        
        # Quick test preset
        presets['quick_test'] = HyperparameterConfig(
            population_size=10,
            max_generations=15,
            max_problems=20,
            convergence_patience=5,
            checkpoint_interval=5
        )
        
        # Standard preset
        presets['standard'] = HyperparameterConfig(
            population_size=50,
            max_generations=100,
            max_problems=100,
            target_fitness=0.85,
            convergence_patience=20
        )
        
        # Thorough preset
        presets['thorough'] = HyperparameterConfig(
            population_size=100,
            max_generations=200,
            max_problems=200,
            target_fitness=0.9,
            convergence_patience=30,
            checkpoint_interval=5
        )
        
        # High exploration preset
        presets['high_exploration'] = HyperparameterConfig(
            population_size=75,
            max_generations=150,
            mutation_rate=0.4,
            crossover_rate=0.6,
            tournament_size=2,
            diversity_threshold=0.1
        )
        
        # High exploitation preset
        presets['high_exploitation'] = HyperparameterConfig(
            population_size=50,
            max_generations=100,
            mutation_rate=0.1,
            crossover_rate=0.9,
            elite_size=10,
            tournament_size=5
        )
        
        # Ablation study presets
        presets['no_crossover'] = HyperparameterConfig(
            crossover_rate=0.0,
            mutation_rate=0.5
        )
        
        presets['no_mutation'] = HyperparameterConfig(
            crossover_rate=0.9,
            mutation_rate=0.0
        )
        
        # Large population preset
        presets['large_population'] = HyperparameterConfig(
            population_size=150,
            max_generations=50,
            elite_size=15
        )
        
        # Fast convergence preset
        presets['fast_convergence'] = HyperparameterConfig(
            population_size=30,
            max_generations=50,
            target_fitness=0.8,
            convergence_patience=10,
            fitness_plateau_generations=5
        )
        
        return presets
    
    def save_config(self, config: HyperparameterConfig, name: str, 
                   description: Optional[str] = None) -> Path:
        """Save configuration to file."""
        filename = f"{name}.json"
        filepath = self.config_dir / filename
        
        # Add metadata
        config_data = config.to_dict()
        config_data['_metadata'] = {
            'name': name,
            'description': description or f"Configuration: {name}",
            'created_by': 'ConfigurationManager',
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved as {name} to {filepath}")
        return filepath
    
    def create_custom_config(self, base_preset: str, 
                           modifications: Dict[str, Any]) -> HyperparameterConfig:
        """Create a custom configuration based on a preset with modifications."""
        base_config = self.load_preset(base_preset)
        
        # Create a copy and apply modifications
        config_dict = base_config.to_dict()
        config_dict.update(modifications)
        
        custom_config = HyperparameterConfig.from_dict(config_dict)
        
        logger.info(f"Created custom configuration based on {base_preset} with {len(modifications)} modifications")
        return custom_config
    
    def get_current_config(self) -> Optional[HyperparameterConfig]:
        """Get the currently active configuration."""
        return self._current_config
    
    def get_config_history(self) -> List[HyperparameterConfig]:
        """Get the history of loaded configurations."""
        return self._config_history.copy()
    
    def validate_config(self, config: HyperparameterConfig) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []
        
        # Check for common issues
        if config.population_size < 10:
            warnings.append("Small population size may lead to poor diversity")
        
        if config.elite_size >= config.population_size * 0.5:
            warnings.append("Elite size is very large relative to population")
        
        if config.mutation_rate + config.crossover_rate > 1.2:
            warnings.append("Combined mutation and crossover rates are very high")
        
        if config.max_generations < 20:
            warnings.append("Low generation count may not allow sufficient evolution")
        
        if config.tournament_size >= config.population_size:
            warnings.append("Tournament size should be smaller than population size")
        
        if config.target_fitness and config.target_fitness > 0.95:
            warnings.append("Very high target fitness may be difficult to achieve")
        
        # Check parameter relationships
        if config.min_genome_length >= config.max_genome_length:
            warnings.append("Minimum genome length should be less than maximum")
        
        if config.min_initial_length >= config.max_initial_length:
            warnings.append("Minimum initial length should be less than maximum")
        
        if config.fitness_plateau_generations >= config.convergence_patience:
            warnings.append("Fitness plateau generations should be less than convergence patience")
        
        return warnings
    
    def get_parameter_summary(self, config: Optional[HyperparameterConfig] = None) -> Dict[str, Any]:
        """Get a summary of configuration parameters by category."""
        if config is None:
            config = self._current_config
        
        if config is None:
            return {}
        
        summary = {}
        categories = config.get_all_categories()
        
        for category in categories:
            category_params = config.get_parameters_by_category(category)
            summary[category] = category_params
        
        return summary
    
    def export_config_template(self, filepath: Union[str, Path]) -> None:
        """Export a configuration template with all parameters and descriptions."""
        filepath = Path(filepath)
        
        specs = HyperparameterConfig.get_parameter_specs()
        template = {}
        
        # Group by category
        categories = {}
        for param_name, spec in specs.items():
            if spec.category not in categories:
                categories[spec.category] = {}
            
            param_info = {
                'value': spec.default_value,
                'description': spec.description,
                'type': spec.parameter_type.__name__,
            }
            
            if spec.min_value is not None:
                param_info['min_value'] = spec.min_value
            if spec.max_value is not None:
                param_info['max_value'] = spec.max_value
            if spec.valid_values is not None:
                param_info['valid_values'] = spec.valid_values
            
            categories[spec.category][param_name] = param_info
        
        template['_template_info'] = {
            'description': 'Hyperparameter configuration template',
            'version': '1.0',
            'categories': list(categories.keys())
        }
        
        template.update(categories)
        
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Configuration template exported to {filepath}")


# Global configuration manager instance
_global_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ConfigurationManager()
    return _global_manager


def load_hyperparameter_config(source: Union[str, Path, Dict[str, Any]]) -> HyperparameterConfig:
    """Load hyperparameter configuration using the global manager."""
    manager = get_config_manager()
    return manager.load_config(source)


def get_preset_configs() -> Dict[str, HyperparameterConfig]:
    """Get all available preset configurations."""
    manager = get_config_manager()
    return manager.get_available_presets()
