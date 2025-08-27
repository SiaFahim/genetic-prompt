"""
Configuration management for GSM8K Genetic Algorithm Project.
Handles loading environment variables and configuration settings.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for the GSM8K genetic algorithm project."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file = env_file or self.project_root / ".env"
        self._load_env_file()
        
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv('OPENAI_API_KEY')
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key from environment."""
        return os.getenv('ANTHROPIC_API_KEY')
    
    @property
    def default_model(self) -> str:
        """Get default model name."""
        return os.getenv('DEFAULT_MODEL', 'gpt-4')
    
    @property
    def temperature(self) -> float:
        """Get temperature for LLM generation."""
        return float(os.getenv('TEMPERATURE', '0.0'))
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens for LLM generation."""
        return int(os.getenv('MAX_TOKENS', '200'))
    
    @property
    def random_seed(self) -> int:
        """Get random seed for reproducibility."""
        return int(os.getenv('RANDOM_SEED', '42'))
    
    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    
    def get_data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        cache_dir = self.get_data_dir() / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_checkpoints_dir(self) -> Path:
        """Get checkpoints directory path."""
        return self.get_data_dir() / "checkpoints"
    
    def get_results_dir(self) -> Path:
        """Get results directory path."""
        return self.get_data_dir() / "results"
    
    def get_seeds_dir(self) -> Path:
        """Get seeds directory path."""
        return self.get_data_dir() / "seeds"

    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        logs_dir = self.get_data_dir() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'default_model': self.default_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'random_seed': self.random_seed,
            'cache_enabled': self.cache_enabled,
            'data_dir': str(self.get_data_dir()),
            'cache_dir': str(self.get_cache_dir()),
            'checkpoints_dir': str(self.get_checkpoints_dir()),
            'results_dir': str(self.get_results_dir()),
            'seeds_dir': str(self.get_seeds_dir()),
        }


# Global configuration instance
config = Config()
