"""
Comprehensive logging system for genetic algorithm evolution.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
else:
    from .config import config


class EvolutionLogger:
    """Comprehensive logging system for evolution experiments."""
    
    def __init__(self, experiment_name: str, log_level: str = "INFO"):
        """
        Initialize evolution logger.
        
        Args:
            experiment_name: Name of the experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.experiment_name = experiment_name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logs directory
        self.logs_dir = config.get_logs_dir()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific log directory
        self.experiment_logs_dir = self.logs_dir / experiment_name
        self.experiment_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # Experiment metadata
        self.experiment_start_time = time.time()
        self.generation_logs = []
        self.performance_logs = []
        self.error_logs = []
    
    def _setup_loggers(self):
        """Set up different types of loggers."""
        
        # Main evolution logger
        self.evolution_logger = logging.getLogger(f"evolution_{self.experiment_name}")
        self.evolution_logger.setLevel(self.log_level)
        self.evolution_logger.handlers.clear()
        
        # Evolution log file with rotation
        evolution_handler = RotatingFileHandler(
            self.experiment_logs_dir / "evolution.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        evolution_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        evolution_handler.setFormatter(evolution_formatter)
        self.evolution_logger.addHandler(evolution_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.evolution_logger.addHandler(console_handler)
        
        # Performance logger (separate file)
        self.performance_logger = logging.getLogger(f"performance_{self.experiment_name}")
        self.performance_logger.setLevel(logging.DEBUG)
        self.performance_logger.handlers.clear()
        
        performance_handler = RotatingFileHandler(
            self.experiment_logs_dir / "performance.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        performance_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        performance_handler.setFormatter(performance_formatter)
        self.performance_logger.addHandler(performance_handler)
        
        # Error logger (separate file)
        self.error_logger = logging.getLogger(f"error_{self.experiment_name}")
        self.error_logger.setLevel(logging.WARNING)
        self.error_logger.handlers.clear()
        
        error_handler = RotatingFileHandler(
            self.experiment_logs_dir / "errors.log",
            maxBytes=2*1024*1024,  # 2MB
            backupCount=2
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.evolution_logger.info(f"🧬 Starting evolution experiment: {self.experiment_name}")
        self.evolution_logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Save configuration to file
        config_file = self.experiment_logs_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'start_time': self.experiment_start_time,
                'config': config
            }, f, indent=2)
    
    def log_generation(self, generation: int, result: Dict[str, Any]):
        """Log generation results."""
        self.evolution_logger.info(
            f"Generation {generation}: "
            f"best_fitness={result.get('best_fitness', 0):.3f}, "
            f"mean_fitness={result.get('mean_fitness', 0):.3f}, "
            f"diversity={result.get('diversity', 0):.3f}"
        )
        
        # Store detailed generation log
        generation_log = {
            'generation': generation,
            'timestamp': time.time(),
            'result': result
        }
        self.generation_logs.append(generation_log)
        
        # Save generation details to JSON
        if generation % 10 == 0:  # Save every 10 generations
            self._save_generation_logs()
    
    def log_convergence(self, reason: str, generation: int, best_fitness: float):
        """Log convergence event."""
        self.evolution_logger.info(
            f"🎯 Convergence detected at generation {generation}: {reason} "
            f"(best_fitness={best_fitness:.3f})"
        )
    
    def log_performance(self, metric: str, value: float, context: Optional[str] = None):
        """Log performance metrics."""
        message = f"{metric}: {value}"
        if context:
            message += f" ({context})"
        
        self.performance_logger.debug(message)
        
        # Store performance log
        perf_log = {
            'timestamp': time.time(),
            'metric': metric,
            'value': value,
            'context': context
        }
        self.performance_logs.append(perf_log)
    
    def log_api_usage(self, api_calls: int, tokens_used: int, cost_estimate: float):
        """Log API usage statistics."""
        self.performance_logger.info(
            f"API Usage - Calls: {api_calls}, Tokens: {tokens_used}, "
            f"Estimated Cost: ${cost_estimate:.4f}"
        )
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """Log errors with context."""
        error_message = f"Error: {str(error)}"
        if context:
            error_message = f"{context} - {error_message}"
        
        self.error_logger.error(error_message, exc_info=True)
        
        # Store error log
        error_log = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        self.error_logs.append(error_log)
    
    def log_genome_details(self, genome_id: str, fitness: float, prompt_text: str):
        """Log detailed genome information."""
        self.evolution_logger.debug(
            f"Genome {genome_id[:8]}: fitness={fitness:.3f}, "
            f"prompt='{prompt_text[:50]}...'"
        )
    
    def log_selection_stats(self, method: str, stats: Dict[str, Any]):
        """Log selection statistics."""
        self.evolution_logger.debug(f"Selection ({method}): {stats}")
    
    def log_mutation_stats(self, mutations_applied: int, success_rate: float):
        """Log mutation statistics."""
        self.performance_logger.debug(
            f"Mutations: {mutations_applied} applied, "
            f"success_rate={success_rate:.2%}"
        )
    
    def log_evaluation_batch(self, batch_size: int, eval_time: float, cache_hits: int):
        """Log evaluation batch statistics."""
        self.performance_logger.info(
            f"Evaluation batch: {batch_size} genomes, "
            f"time={eval_time:.1f}s, cache_hits={cache_hits}"
        )
    
    def log_experiment_end(self, final_results: Dict[str, Any]):
        """Log experiment completion."""
        total_time = time.time() - self.experiment_start_time
        
        self.evolution_logger.info(
            f"🏆 Experiment completed in {total_time:.1f}s: "
            f"best_fitness={final_results.get('best_fitness', 0):.3f}"
        )
        
        # Save final results
        results_file = self.experiment_logs_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'total_time': total_time,
                'final_results': final_results,
                'end_time': time.time()
            }, f, indent=2)
        
        # Save all logs
        self._save_all_logs()
    
    def _save_generation_logs(self):
        """Save generation logs to file."""
        logs_file = self.experiment_logs_dir / "generation_logs.json"
        with open(logs_file, 'w') as f:
            json.dump(self.generation_logs, f, indent=2)
    
    def _save_all_logs(self):
        """Save all accumulated logs."""
        # Save generation logs
        self._save_generation_logs()
        
        # Save performance logs
        perf_file = self.experiment_logs_dir / "performance_logs.json"
        with open(perf_file, 'w') as f:
            json.dump(self.performance_logs, f, indent=2)
        
        # Save error logs
        error_file = self.experiment_logs_dir / "error_logs.json"
        with open(error_file, 'w') as f:
            json.dump(self.error_logs, f, indent=2)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logged information."""
        return {
            'experiment_name': self.experiment_name,
            'start_time': self.experiment_start_time,
            'total_generations': len(self.generation_logs),
            'total_performance_logs': len(self.performance_logs),
            'total_errors': len(self.error_logs),
            'log_directory': str(self.experiment_logs_dir)
        }


# Global logger instance
_global_logger: Optional[EvolutionLogger] = None


def get_logger(experiment_name: str = "default", log_level: str = "INFO") -> EvolutionLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None or _global_logger.experiment_name != experiment_name:
        _global_logger = EvolutionLogger(experiment_name, log_level)
    
    return _global_logger


def log_info(message: str, experiment_name: str = "default"):
    """Quick info logging function."""
    logger = get_logger(experiment_name)
    logger.evolution_logger.info(message)


def log_error(error: Exception, context: str = None, experiment_name: str = "default"):
    """Quick error logging function."""
    logger = get_logger(experiment_name)
    logger.log_error(error, context)


if __name__ == "__main__":
    # Test logging system
    print("Testing evolution logging system...")
    
    # Create test logger
    logger = EvolutionLogger("test_experiment", "DEBUG")
    
    # Test experiment start
    test_config = {
        'population_size': 50,
        'max_generations': 100,
        'mutation_rate': 0.2
    }
    logger.log_experiment_start(test_config)
    print("✅ Experiment start logged")
    
    # Test generation logging
    for gen in range(5):
        result = {
            'best_fitness': 0.5 + gen * 0.1,
            'mean_fitness': 0.3 + gen * 0.05,
            'diversity': 0.8 - gen * 0.1
        }
        logger.log_generation(gen + 1, result)
    print("✅ Generation logs created")
    
    # Test performance logging
    logger.log_performance("evaluation_time", 45.2, "batch_size_10")
    logger.log_api_usage(100, 5000, 0.25)
    print("✅ Performance logs created")
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.log_error(e, "test_context")
    print("✅ Error logs created")
    
    # Test experiment end
    final_results = {
        'best_fitness': 0.85,
        'total_generations': 5,
        'convergence_reason': 'test_complete'
    }
    logger.log_experiment_end(final_results)
    print("✅ Experiment end logged")
    
    # Test summary
    summary = logger.get_log_summary()
    print(f"✅ Log summary: {summary}")
    
    print("\n🎯 Logging system tests completed successfully!")
