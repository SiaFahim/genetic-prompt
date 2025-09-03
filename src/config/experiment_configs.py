"""
Experiment configuration system for GSM8K genetic algorithm evolution.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.selection import SelectionMethod
    from src.genetics.crossover import CrossoverType
    from src.genetics.mutation import MutationType
    from src.utils.config import config
else:
    from ..genetics.selection import SelectionMethod
    from ..genetics.crossover import CrossoverType
    from ..genetics.mutation import MutationType
    from ..utils.config import config


class ExperimentType(Enum):
    """Types of experiments."""
    QUICK_TEST = "quick_test"
    STANDARD = "standard"
    THOROUGH = "thorough"
    ABLATION_STUDY = "ablation_study"
    PARAMETER_SWEEP = "parameter_sweep"
    BASELINE_COMPARISON = "baseline_comparison"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Basic info
    name: str
    description: str
    experiment_type: ExperimentType
    
    # Evolution parameters
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elite_size: int = 5
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    crossover_type: CrossoverType = CrossoverType.SINGLE_POINT
    mutation_type: MutationType = MutationType.SEMANTIC
    target_fitness: Optional[float] = 0.85
    convergence_patience: int = 20
    adaptive_parameters: bool = True
    
    # Evaluation parameters
    max_problems: int = 100
    use_cache: bool = True
    batch_size: int = 10

    # Async evaluation parameters
    enable_async_evaluation: bool = True
    async_batch_size: int = 20
    max_concurrent_requests: int = 10
    genome_batch_size: int = 10
    max_concurrent_genomes: int = 5
    rate_limit_per_minute: int = 3500
    
    # Seed parameters
    seed_strategy: str = "balanced"  # balanced, diverse, random
    custom_seeds: Optional[List[str]] = None
    
    # Monitoring parameters
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    enable_visualization: bool = True
    performance_monitoring: bool = True
    
    # API parameters
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 150
    
    # Output parameters
    save_results: bool = True
    save_plots: bool = True
    verbose: bool = True


class ConfigurationManager:
    """Manages experiment configurations and presets."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.configs_dir = config.get_data_dir() / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predefined configurations
        self.presets = self._create_preset_configurations()
    
    def _create_preset_configurations(self) -> Dict[str, ExperimentConfig]:
        """Create predefined experiment configurations."""
        presets = {}
        
        # Quick Test Configuration
        presets['quick_test'] = ExperimentConfig(
            name="Quick Test",
            description="Fast test run for system validation",
            experiment_type=ExperimentType.QUICK_TEST,
            population_size=10,
            max_generations=15,
            max_problems=20,
            target_fitness=0.7,
            convergence_patience=5,
            checkpoint_interval=5
        )
        
        # Standard Configuration
        presets['standard'] = ExperimentConfig(
            name="Standard Evolution",
            description="Standard GSM8K evolution experiment",
            experiment_type=ExperimentType.STANDARD,
            population_size=50,
            max_generations=100,
            max_problems=100,
            target_fitness=0.85,
            convergence_patience=20
        )
        
        # Thorough Configuration
        presets['thorough'] = ExperimentConfig(
            name="Thorough Evolution",
            description="Comprehensive evolution with large population",
            experiment_type=ExperimentType.THOROUGH,
            population_size=100,
            max_generations=200,
            max_problems=200,
            target_fitness=0.9,
            convergence_patience=30,
            checkpoint_interval=5
        )
        
        # Ablation Study - No Crossover
        presets['ablation_no_crossover'] = ExperimentConfig(
            name="Ablation: No Crossover",
            description="Evolution with mutation only (no crossover)",
            experiment_type=ExperimentType.ABLATION_STUDY,
            population_size=50,
            max_generations=100,
            crossover_rate=0.0,
            mutation_rate=0.5,
            max_problems=100
        )
        
        # Ablation Study - No Mutation
        presets['ablation_no_mutation'] = ExperimentConfig(
            name="Ablation: No Mutation",
            description="Evolution with crossover only (no mutation)",
            experiment_type=ExperimentType.ABLATION_STUDY,
            population_size=50,
            max_generations=100,
            crossover_rate=0.9,
            mutation_rate=0.0,
            max_problems=100
        )
        
        # Parameter Sweep - High Mutation
        presets['high_mutation'] = ExperimentConfig(
            name="High Mutation Rate",
            description="Evolution with high mutation rate",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            population_size=50,
            max_generations=100,
            mutation_rate=0.4,
            max_problems=100
        )
        
        # Parameter Sweep - Large Population
        presets['large_population'] = ExperimentConfig(
            name="Large Population",
            description="Evolution with large population size",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            population_size=150,
            max_generations=50,
            max_problems=100,
            elite_size=15
        )
        
        # Baseline Comparison - Random Search
        presets['random_search'] = ExperimentConfig(
            name="Random Search Baseline",
            description="Random search baseline for comparison",
            experiment_type=ExperimentType.BASELINE_COMPARISON,
            population_size=50,
            max_generations=100,
            crossover_rate=0.0,
            mutation_rate=1.0,
            selection_method=SelectionMethod.TOURNAMENT,
            tournament_size=1,  # Random selection
            max_problems=100
        )
        
        return presets
    
    def get_preset(self, preset_name: str) -> Optional[ExperimentConfig]:
        """Get a preset configuration by name."""
        return self.presets.get(preset_name)
    
    def list_presets(self) -> List[str]:
        """List all available preset names."""
        return list(self.presets.keys())
    
    def get_preset_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all presets."""
        info = {}
        for name, config in self.presets.items():
            info[name] = {
                'name': config.name,
                'description': config.description,
                'type': config.experiment_type.value,
                'population_size': config.population_size,
                'max_generations': config.max_generations,
                'max_problems': config.max_problems
            }
        return info
    
    def create_custom_config(self, base_preset: str, 
                           modifications: Dict[str, Any]) -> ExperimentConfig:
        """
        Create custom configuration based on preset with modifications.
        
        Args:
            base_preset: Name of base preset
            modifications: Dictionary of parameter modifications
            
        Returns:
            Modified ExperimentConfig
        """
        if base_preset not in self.presets:
            raise ValueError(f"Unknown preset: {base_preset}")
        
        # Get base config
        base_config = self.presets[base_preset]
        config_dict = asdict(base_config)
        
        # Apply modifications
        for key, value in modifications.items():
            if key in config_dict:
                # Handle enum conversions
                if key == 'selection_method' and isinstance(value, str):
                    config_dict[key] = SelectionMethod(value)
                elif key == 'crossover_type' and isinstance(value, str):
                    config_dict[key] = CrossoverType(value)
                elif key == 'mutation_type' and isinstance(value, str):
                    config_dict[key] = MutationType(value)
                elif key == 'experiment_type' and isinstance(value, str):
                    config_dict[key] = ExperimentType(value)
                else:
                    config_dict[key] = value
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        
        return ExperimentConfig(**config_dict)
    
    def save_config(self, config: ExperimentConfig, filename: str):
        """Save configuration to file."""
        config_dict = asdict(config)
        
        # Convert enums to strings for JSON serialization
        for key, value in config_dict.items():
            if hasattr(value, 'value'):  # Enum
                config_dict[key] = value.value
        
        config_file = self.configs_dir / f"{filename}.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved: {config_file}")
    
    def load_config(self, filename: str) -> Optional[ExperimentConfig]:
        """Load configuration from file."""
        config_file = self.configs_dir / f"{filename}.json"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Convert string enums back to enum objects
            if 'selection_method' in config_dict:
                config_dict['selection_method'] = SelectionMethod(config_dict['selection_method'])
            if 'crossover_type' in config_dict:
                config_dict['crossover_type'] = CrossoverType(config_dict['crossover_type'])
            if 'mutation_type' in config_dict:
                config_dict['mutation_type'] = MutationType(config_dict['mutation_type'])
            if 'experiment_type' in config_dict:
                config_dict['experiment_type'] = ExperimentType(config_dict['experiment_type'])
            
            return ExperimentConfig(**config_dict)
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return None
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Population size validation
        if config.population_size < 5:
            errors.append("Population size must be at least 5")
        if config.population_size > 500:
            errors.append("Population size should not exceed 500")
        
        # Generation validation
        if config.max_generations < 1:
            errors.append("Max generations must be at least 1")
        
        # Rate validations
        if not 0 <= config.crossover_rate <= 1:
            errors.append("Crossover rate must be between 0 and 1")
        if not 0 <= config.mutation_rate <= 1:
            errors.append("Mutation rate must be between 0 and 1")
        
        # Elite size validation
        if config.elite_size >= config.population_size:
            errors.append("Elite size must be less than population size")
        if config.elite_size < 0:
            errors.append("Elite size must be non-negative")
        
        # Tournament size validation
        if config.tournament_size > config.population_size:
            errors.append("Tournament size cannot exceed population size")
        if config.tournament_size < 1:
            errors.append("Tournament size must be at least 1")
        
        # Target fitness validation
        if config.target_fitness is not None:
            if not 0 <= config.target_fitness <= 1:
                errors.append("Target fitness must be between 0 and 1")
        
        # Problem count validation
        if config.max_problems < 1:
            errors.append("Max problems must be at least 1")
        if config.max_problems > 1000:
            errors.append("Max problems should not exceed 1000 for performance")
        
        return errors
    
    def get_config_summary(self, config: ExperimentConfig) -> str:
        """Get human-readable configuration summary."""
        summary = []
        summary.append(f"📋 {config.name}")
        summary.append(f"   {config.description}")
        summary.append(f"   Type: {config.experiment_type.value}")
        summary.append(f"   Population: {config.population_size}")
        summary.append(f"   Generations: {config.max_generations}")
        summary.append(f"   Problems: {config.max_problems}")
        summary.append(f"   Crossover: {config.crossover_rate:.1%}")
        summary.append(f"   Mutation: {config.mutation_rate:.1%}")
        summary.append(f"   Selection: {config.selection_method.value}")
        summary.append(f"   Model: {config.model_name}")
        summary.append(f"   Target Fitness: {config.target_fitness}")

        return "\n".join(summary)


if __name__ == "__main__":
    # Test configuration system
    print("Testing experiment configuration system...")
    
    manager = ConfigurationManager()
    
    # Test preset listing
    presets = manager.list_presets()
    print(f"✅ Available presets: {len(presets)}")
    for preset in presets:
        print(f"   - {preset}")
    
    # Test preset retrieval
    standard_config = manager.get_preset('standard')
    assert standard_config is not None, "Standard preset not found"
    print(f"✅ Standard config loaded: {standard_config.name}")
    
    # Test configuration validation
    errors = manager.validate_config(standard_config)
    assert len(errors) == 0, f"Standard config validation failed: {errors}"
    print("✅ Standard config validation passed")
    
    # Test custom configuration creation
    custom_config = manager.create_custom_config('standard', {
        'population_size': 30,
        'max_generations': 50,
        'mutation_rate': 0.3
    })
    assert custom_config.population_size == 30, "Custom modification failed"
    print("✅ Custom configuration creation works")
    
    # Test configuration summary
    summary = manager.get_config_summary(standard_config)
    assert "Standard Evolution" in summary, "Summary missing config name"
    print("✅ Configuration summary generated")
    
    # Test save/load
    test_filename = "test_config"
    manager.save_config(custom_config, test_filename)
    loaded_config = manager.load_config(test_filename)
    assert loaded_config is not None, "Failed to load saved config"
    assert loaded_config.population_size == 30, "Loaded config differs from saved"
    print("✅ Configuration save/load works")
    
    print("\n🎯 Configuration system tests completed successfully!")
