"""
Configuration management for GSM8K Genetic Algorithm system.
Loads configuration from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Try to import python-dotenv, install if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    # Continue without dotenv - environment variables should still work

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the GSM8K genetic algorithm system."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize configuration from JSON file and environment variables.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables where applicable
        self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration."""
        
        # API Keys
        if os.getenv('OPENAI_API_KEY'):
            config['api'] = config.get('api', {})
            config['api']['openai_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('ANTHROPIC_API_KEY'):
            config['api'] = config.get('api', {})
            config['api']['anthropic_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        # Model configuration
        if os.getenv('MODEL_NAME'):
            config['model']['name'] = os.getenv('MODEL_NAME')
        
        # Experiment settings
        if os.getenv('EXPERIMENT_NAME'):
            config['experiment']['name'] = os.getenv('EXPERIMENT_NAME')
        
        if os.getenv('RANDOM_SEED'):
            config['experiment']['random_seed'] = int(os.getenv('RANDOM_SEED'))
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            config['logging']['level'] = os.getenv('LOG_LEVEL')
        
        # Cache settings
        if os.getenv('ENABLE_CACHE'):
            config['evaluation']['cache_enabled'] = os.getenv('ENABLE_CACHE').lower() == 'true'
        
        if os.getenv('CACHE_DIR'):
            config['paths']['cache_dir'] = os.getenv('CACHE_DIR')
    
    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        required_sections = ['experiment', 'model', 'genetic_algorithm', 'evaluation']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check for API key
        api_config = self._config.get('api', {})
        if not (api_config.get('openai_key') or api_config.get('anthropic_key')):
            logger.warning("No API key found in configuration or environment variables")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self) -> str:
        """Get the appropriate API key based on model configuration."""
        model_name = self.get('model.name', '').lower()
        
        if 'gpt' in model_name or 'openai' in model_name:
            api_key = self.get('api.openai_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration or environment")
            return api_key
        
        elif 'claude' in model_name or 'anthropic' in model_name:
            api_key = self.get('api.anthropic_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found in configuration or environment")
            return api_key
        
        else:
            # Default to OpenAI
            api_key = self.get('api.openai_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("No API key found for the specified model")
            return api_key
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get complete model configuration."""
        return self.get('model', {})
    
    def get_ga_config(self) -> Dict[str, Any]:
        """Get genetic algorithm configuration."""
        return self.get('genetic_algorithm', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get all configured paths."""
        return self.get('paths', {})
    
    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        paths = self.get_paths()
        
        for path_name, path_value in paths.items():
            if path_value:
                Path(path_value).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path_value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the complete configuration as a dictionary."""
        return self._config.copy()
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save to, defaults to original config path
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2)
        
        logger.info(f"Configuration saved to: {save_path}")


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "configs/experiment_config.json") -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: str = "configs/experiment_config.json") -> Config:
    """
    Reload the global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
