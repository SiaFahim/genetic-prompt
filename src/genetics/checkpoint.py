"""
Checkpointing system for genetic algorithm evolution.
"""

import json
import pickle
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import asdict

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.genetics.genome import PromptGenome
    from src.genetics.population import Population
    from src.genetics.evolution import EvolutionConfig, GenerationResult
    from src.utils.config import config
else:
    from .genome import PromptGenome
    from .population import Population
    from .evolution import EvolutionConfig, GenerationResult
    from ..utils.config import config


class CheckpointManager:
    """Manages saving and loading evolution checkpoints."""
    
    def __init__(self, experiment_name: str = "gsm8k_evolution"):
        """
        Initialize checkpoint manager.
        
        Args:
            experiment_name: Name of the experiment for checkpoint organization
        """
        self.experiment_name = experiment_name
        self.checkpoint_dir = config.get_checkpoints_dir() / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files
        self.latest_checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'created': time.time(),
            'last_checkpoint': None,
            'total_checkpoints': 0,
            'checkpoint_history': []
        }
        
        # Load existing metadata
        self._load_metadata()
    
    def save_checkpoint(self, 
                       population: Population,
                       generation_results: List[GenerationResult],
                       best_genome: Optional[PromptGenome],
                       evolution_config: EvolutionConfig,
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Save evolution checkpoint.
        
        Args:
            population: Current population
            generation_results: List of generation results
            best_genome: Best genome found so far
            evolution_config: Evolution configuration
            additional_data: Additional data to save
            
        Returns:
            Checkpoint filename
        """
        current_generation = population.generation
        timestamp = time.time()
        
        # Create checkpoint filename
        checkpoint_filename = f"checkpoint_gen_{current_generation:04d}_{int(timestamp)}.json"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'metadata': {
                'experiment_name': self.experiment_name,
                'generation': current_generation,
                'timestamp': timestamp,
                'checkpoint_version': '1.0'
            },
            'population': {
                'generation': population.generation,
                'population_size': len(population),
                'genomes': [
                    {
                        'genome_id': genome.genome_id,
                        'token_ids': genome.token_ids,
                        'fitness': genome.fitness,
                        'generation': genome.generation,
                        'age': genome.age,
                        'evaluation_count': genome.evaluation_count,
                        'parent_ids': genome.parent_ids,
                        'mutation_history': genome.mutation_history
                    }
                    for genome in population.genomes
                ]
            },
            'best_genome': {
                'genome_id': best_genome.genome_id,
                'token_ids': best_genome.token_ids,
                'fitness': best_genome.fitness,
                'prompt_text': best_genome.to_text()
            } if best_genome else None,
            'generation_results': [
                {
                    'generation': result.generation,
                    'best_fitness': result.best_fitness,
                    'mean_fitness': result.mean_fitness,
                    'diversity': result.diversity,
                    'evaluation_time': result.evaluation_time,
                    'evolution_time': result.evolution_time,
                    'convergence_status': {
                        'converged': result.convergence_status.converged,
                        'reason': result.convergence_status.reason.value,
                        'confidence': result.convergence_status.confidence
                    }
                }
                for result in generation_results
            ],
            'evolution_config': self._serialize_config(evolution_config),
            'additional_data': additional_data or {}
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Update latest checkpoint link
        with open(self.latest_checkpoint_file, 'w') as f:
            json.dump({'latest_checkpoint': checkpoint_filename}, f)
        
        # Update metadata
        self.metadata['last_checkpoint'] = checkpoint_filename
        self.metadata['total_checkpoints'] += 1
        self.metadata['checkpoint_history'].append({
            'filename': checkpoint_filename,
            'generation': current_generation,
            'timestamp': timestamp,
            'best_fitness': best_genome.fitness if best_genome else None
        })
        
        # Keep only last 10 checkpoint entries in history
        if len(self.metadata['checkpoint_history']) > 10:
            self.metadata['checkpoint_history'] = self.metadata['checkpoint_history'][-10:]
        
        self._save_metadata()

        print(f"💾 Checkpoint saved: {checkpoint_filename}")
        return checkpoint_filename

    def _serialize_config(self, config: EvolutionConfig) -> Dict[str, Any]:
        """Serialize evolution config to JSON-compatible format."""
        config_dict = asdict(config)

        # Convert enums to strings
        if 'selection_method' in config_dict:
            config_dict['selection_method'] = config_dict['selection_method'].value
        if 'crossover_type' in config_dict:
            config_dict['crossover_type'] = config_dict['crossover_type'].value
        if 'mutation_type' in config_dict:
            config_dict['mutation_type'] = config_dict['mutation_type'].value

        return config_dict
    
    def load_checkpoint(self, checkpoint_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load evolution checkpoint.
        
        Args:
            checkpoint_filename: Specific checkpoint to load (None for latest)
            
        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_filename is None:
            # Load latest checkpoint
            if not self.latest_checkpoint_file.exists():
                raise FileNotFoundError("No checkpoints found")
            
            with open(self.latest_checkpoint_file, 'r') as f:
                latest_info = json.load(f)
                checkpoint_filename = latest_info['latest_checkpoint']
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_filename}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        print(f"📂 Checkpoint loaded: {checkpoint_filename}")
        return checkpoint_data
    
    def restore_population(self, checkpoint_data: Dict[str, Any]) -> Population:
        """
        Restore population from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            Restored Population object
        """
        pop_data = checkpoint_data['population']
        
        # Create population
        population = Population(pop_data['population_size'])
        population.generation = pop_data['generation']
        
        # Restore genomes
        population.genomes = []
        for genome_data in pop_data['genomes']:
            genome = PromptGenome.from_tokens(
                genome_data['token_ids'],
                genome_id=genome_data['genome_id']
            )
            genome.fitness = genome_data['fitness']
            genome.generation = genome_data['generation']
            genome.age = genome_data['age']
            genome.evaluation_count = genome_data['evaluation_count']
            genome.parent_ids = genome_data['parent_ids']
            genome.mutation_history = genome_data['mutation_history']
            
            population.genomes.append(genome)
        
        return population
    
    def restore_best_genome(self, checkpoint_data: Dict[str, Any]) -> Optional[PromptGenome]:
        """
        Restore best genome from checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data dictionary
            
        Returns:
            Restored best genome or None
        """
        best_data = checkpoint_data.get('best_genome')
        if not best_data:
            return None
        
        genome = PromptGenome.from_tokens(
            best_data['token_ids'],
            genome_id=best_data['genome_id']
        )
        genome.fitness = best_data['fitness']
        
        return genome
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                checkpoints.append({
                    'filename': checkpoint_file.name,
                    'generation': data['metadata']['generation'],
                    'timestamp': data['metadata']['timestamp'],
                    'best_fitness': (data['best_genome']['fitness'] 
                                   if data['best_genome'] else None),
                    'population_size': len(data['population']['genomes'])
                })
            except Exception as e:
                print(f"Warning: Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by generation
        checkpoints.sort(key=lambda x: x['generation'])
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_count:
            return
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-keep_count]
        
        for checkpoint in to_remove:
            checkpoint_path = self.checkpoint_dir / checkpoint['filename']
            try:
                checkpoint_path.unlink()
                print(f"🗑️  Removed old checkpoint: {checkpoint['filename']}")
            except Exception as e:
                print(f"Warning: Could not remove checkpoint {checkpoint['filename']}: {e}")
        
        # Update metadata
        self.metadata['checkpoint_history'] = [
            entry for entry in self.metadata['checkpoint_history']
            if entry['filename'] not in [c['filename'] for c in to_remove]
        ]
        self._save_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    loaded_metadata = json.load(f)
                    self.metadata.update(loaded_metadata)
            except Exception as e:
                print(f"Warning: Could not load checkpoint metadata: {e}")
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint metadata: {e}")
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return {'total_checkpoints': 0}
        
        return {
            'total_checkpoints': len(checkpoints),
            'latest_generation': max(c['generation'] for c in checkpoints),
            'best_fitness_overall': max(c['best_fitness'] for c in checkpoints if c['best_fitness']),
            'checkpoint_dir': str(self.checkpoint_dir),
            'disk_usage_mb': sum(
                (self.checkpoint_dir / c['filename']).stat().st_size 
                for c in checkpoints
            ) / (1024 * 1024)
        }


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing checkpoint manager...")
    
    # Load vocabulary for testing
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create test data
    manager = CheckpointManager("test_experiment")
    
    # Create test population
    population = Population(5)
    population.initialize_random(5, 10)
    population.generation = 10
    
    # Set some fitness values
    for i, genome in enumerate(population.genomes):
        genome.set_fitness(0.5 + i * 0.1)
    
    # Create test generation results
    from src.genetics.convergence import ConvergenceStatus, ConvergenceReason
    test_convergence = ConvergenceStatus(
        converged=False,
        reason=ConvergenceReason.NOT_CONVERGED,
        confidence=0.0,
        generations_since_improvement=5,
        current_best_fitness=0.9,
        diversity_score=0.7,
        plateau_length=0,
        details={}
    )
    
    generation_results = [
        GenerationResult(
            generation=10,
            best_fitness=0.9,
            mean_fitness=0.7,
            diversity=0.7,
            convergence_status=test_convergence,
            evaluation_time=30.0,
            evolution_time=5.0,
            population_stats={}
        )
    ]
    
    # Test saving checkpoint
    best_genome = population.get_best_genome()
    evolution_config = EvolutionConfig()
    
    checkpoint_filename = manager.save_checkpoint(
        population, generation_results, best_genome, evolution_config
    )
    print(f"✅ Checkpoint saved: {checkpoint_filename}")
    
    # Test loading checkpoint
    loaded_data = manager.load_checkpoint()
    print(f"✅ Checkpoint loaded: generation {loaded_data['metadata']['generation']}")
    
    # Test restoring population
    restored_population = manager.restore_population(loaded_data)
    print(f"✅ Population restored: {len(restored_population)} genomes")
    
    # Test restoring best genome
    restored_best = manager.restore_best_genome(loaded_data)
    print(f"✅ Best genome restored: fitness={restored_best.fitness}")
    
    # Test listing checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"✅ Found {len(checkpoints)} checkpoints")
    
    # Test statistics
    stats = manager.get_checkpoint_statistics()
    print(f"✅ Checkpoint statistics: {stats}")
    
    print("\n🎯 Checkpoint manager tests completed successfully!")
