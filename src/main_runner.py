"""
Main experiment runner for GSM8K genetic algorithm evolution.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import all system components
from src.utils.config import config
from src.utils.evolution_logging import EvolutionLogger
from src.utils.visualization import EvolutionVisualizer
from src.utils.experiment_manager import ExperimentManager
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.dataset import gsm8k_dataset
from src.embeddings.vocabulary import vocabulary
from src.seeds.seed_manager import SeedManager
from src.genetics.evolution import EvolutionController, EvolutionConfig
from src.evaluation.pipeline import EvaluationPipeline


class GSM8KExperimentRunner:
    """Main runner for GSM8K genetic algorithm experiments."""
    
    def __init__(self, experiment_config: Dict[str, Any]):
        """
        Initialize experiment runner.
        
        Args:
            experiment_config: Complete experiment configuration
        """
        self.config = experiment_config
        self.experiment_id = None
        
        # System components
        self.experiment_manager = ExperimentManager()
        self.seed_manager = SeedManager()
        self.logger = None
        self.visualizer = None
        self.performance_monitor = None
        self.evolution_controller = None
        
        # Results
        self.final_results = None
        self.best_prompt = None
        self.experiment_success = False
    
    def setup_experiment(self) -> bool:
        """
        Set up the experiment with all necessary components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            print("🔧 Setting up GSM8K evolution experiment...")
            
            # Load vocabulary
            print("📚 Loading vocabulary...")
            vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
            if vocab_file.exists():
                vocabulary.load_vocabulary(vocab_file)
                print(f"   ✅ Loaded vocabulary with {len(vocabulary.token_to_id)} tokens")
            else:
                print("   ⚠️  Vocabulary not found, creating basic vocabulary...")
                vocabulary._create_basic_vocabulary()
                print(f"   ✅ Created basic vocabulary with {len(vocabulary.token_to_id)} tokens")
            
            # Load dataset
            print("📊 Loading GSM8K dataset...")
            try:
                problems = gsm8k_dataset.get_primary_eval_set()
                print(f"   ✅ Loaded {len(problems)} evaluation problems")
            except Exception as e:
                print(f"   ❌ Failed to load dataset: {e}")
                return False
            
            # Create experiment
            print("🧪 Creating experiment...")
            evolution_config = EvolutionConfig(**self.config.get('evolution', {}))
            
            self.experiment_id = self.experiment_manager.create_experiment(
                experiment_name=self.config.get('name', 'gsm8k_evolution'),
                description=self.config.get('description', 'GSM8K prompt evolution experiment'),
                evolution_config=evolution_config
            )
            print(f"   ✅ Created experiment: {self.experiment_id}")
            
            # Initialize monitoring components
            print("📊 Initializing monitoring...")
            self.logger, self.visualizer = self.experiment_manager.start_experiment(self.experiment_id)
            self.performance_monitor = PerformanceMonitor(self.experiment_id)
            print("   ✅ Monitoring initialized")
            
            # Create evaluation pipeline
            print("⚡ Setting up evaluation pipeline...")
            self.evaluation_pipeline = EvaluationPipeline(
                use_cache=self.config.get('use_cache', True),
                max_problems=self.config.get('max_problems', 100)
            )
            print("   ✅ Evaluation pipeline ready")
            
            # Get seed prompts
            print("🌱 Loading seed prompts...")
            seed_strategy = self.config.get('seed_strategy', 'balanced')
            seed_count = evolution_config.population_size
            
            if seed_strategy == 'balanced':
                seed_prompts = [seed.text for seed in 
                              self.seed_manager.create_balanced_subset(seed_count)]
            elif seed_strategy == 'diverse':
                seed_prompts = [seed.text for seed in 
                              self.seed_manager.create_balanced_subset(seed_count, strategy='diverse')]
            else:
                seed_prompts = [seed.text for seed in 
                              self.seed_manager.get_base_seeds()[:seed_count]]
            
            print(f"   ✅ Loaded {len(seed_prompts)} seed prompts")
            
            # Create evolution controller
            print("🧬 Initializing evolution controller...")
            self.evolution_controller = EvolutionController(
                config=evolution_config,
                evaluation_pipeline=self.evaluation_pipeline,
                seed_prompts=seed_prompts,
                progress_callback=self._progress_callback
            )
            print("   ✅ Evolution controller ready")
            
            # Log experiment start
            self.logger.log_experiment_start(self.config)
            
            print("✅ Experiment setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Experiment setup failed: {e}")
            traceback.print_exc()
            return False
    
    def run_experiment(self) -> bool:
        """
        Run the complete evolution experiment.
        
        Returns:
            True if experiment completed successfully, False otherwise
        """
        if not self.experiment_id:
            print("❌ Experiment not set up. Call setup_experiment() first.")
            return False
        
        try:
            print(f"\n🚀 Starting GSM8K evolution experiment: {self.experiment_id}")
            print("=" * 60)
            
            # Start performance monitoring
            self.performance_monitor.take_snapshot()
            
            # Run evolution
            start_time = time.time()
            self.final_results = self.evolution_controller.run_evolution()
            total_time = time.time() - start_time
            
            # Extract best prompt
            self.best_prompt = self.final_results.get('best_genome')
            best_fitness = self.final_results.get('best_fitness', 0.0)
            
            print(f"\n🏆 Evolution completed!")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Best fitness: {best_fitness:.3f}")
            print(f"   Generations: {self.final_results.get('total_generations', 0)}")
            print(f"   Convergence: {self.final_results.get('convergence_reason', 'unknown')}")
            
            if self.best_prompt:
                print(f"   Best prompt: {self.best_prompt[:100]}...")
            
            # Final monitoring snapshot
            final_metrics = self.performance_monitor.take_snapshot()
            
            # Log completion
            self.logger.log_experiment_end(self.final_results)
            
            # Save final visualizations
            self.visualizer.save_final_plots()
            
            # Complete experiment
            self.experiment_manager.complete_experiment(self.experiment_id, self.final_results)
            
            # Save performance report
            self.performance_monitor.save_performance_report()
            
            self.experiment_success = True
            print("✅ Experiment completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            traceback.print_exc()
            
            # Log failure
            if self.logger:
                self.logger.log_error(e, "experiment_execution")
            
            # Mark experiment as failed
            if self.experiment_manager and self.experiment_id:
                self.experiment_manager.fail_experiment(self.experiment_id, str(e))
            
            return False
    
    def _progress_callback(self, result):
        """Progress callback for evolution controller."""
        # Update visualizer
        if self.visualizer:
            self.visualizer.update_data(result.generation, {
                'best_fitness': result.best_fitness,
                'mean_fitness': result.mean_fitness,
                'diversity': result.diversity,
                'evaluation_time': result.evaluation_time,
                'convergence_status': {
                    'converged': result.convergence_status.converged,
                    'reason': result.convergence_status.reason.value
                } if result.convergence_status.converged else None
            })
            
            # Update plots every 5 generations
            if result.generation % 5 == 0:
                self.visualizer.update_plots()
        
        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_evaluation_time(result.evaluation_time)
            self.performance_monitor.take_snapshot()
        
        # Check for performance alerts
        if self.performance_monitor and result.generation % 10 == 0:
            alerts = self.performance_monitor.check_performance_alerts()
            for alert in alerts:
                if self.logger:
                    self.logger.evolution_logger.warning(f"Performance Alert: {alert}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        if not self.experiment_success:
            return {'status': 'failed', 'experiment_id': self.experiment_id}
        
        summary = {
            'status': 'completed',
            'experiment_id': self.experiment_id,
            'config': self.config,
            'results': self.final_results,
            'best_prompt': self.best_prompt
        }
        
        # Add monitoring statistics
        if self.performance_monitor:
            summary['performance'] = self.performance_monitor.get_performance_summary()
        
        if self.visualizer:
            summary['visualization'] = self.visualizer.get_statistics()
        
        return summary
    
    def cleanup(self):
        """Clean up experiment resources."""
        if self.performance_monitor:
            self.performance_monitor.save_performance_report()
        
        if self.visualizer:
            self.visualizer.save_final_plots()
        
        print("🧹 Experiment cleanup completed")


def run_gsm8k_experiment(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a complete GSM8K evolution experiment.
    
    Args:
        config_dict: Experiment configuration dictionary
        
    Returns:
        Experiment summary dictionary
    """
    runner = GSM8KExperimentRunner(config_dict)
    
    try:
        # Setup
        if not runner.setup_experiment():
            return {'status': 'setup_failed'}
        
        # Run
        if not runner.run_experiment():
            return {'status': 'execution_failed'}
        
        # Get results
        summary = runner.get_experiment_summary()
        
        return summary
        
    finally:
        runner.cleanup()


if __name__ == "__main__":
    # Example experiment configuration
    example_config = {
        'name': 'gsm8k_test_run',
        'description': 'Test run of GSM8K genetic algorithm evolution',
        'evolution': {
            'population_size': 20,
            'max_generations': 30,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elite_size': 3,
            'target_fitness': 0.8,
            'convergence_patience': 10
        },
        'seed_strategy': 'balanced',
        'use_cache': True,
        'max_problems': 50
    }
    
    print("🧪 Running example GSM8K evolution experiment...")
    
    # Run experiment
    results = run_gsm8k_experiment(example_config)
    
    print(f"\n📊 Experiment Results:")
    print(f"   Status: {results.get('status', 'unknown')}")
    
    if results.get('status') == 'completed':
        print(f"   Best Fitness: {results.get('results', {}).get('best_fitness', 0):.3f}")
        print(f"   Generations: {results.get('results', {}).get('total_generations', 0)}")
        print(f"   Best Prompt: {results.get('best_prompt', 'None')[:100]}...")
    
    print("\n🎯 Example experiment completed!")
