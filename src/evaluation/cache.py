"""
Persistent caching system for evaluation results.
"""

import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.genetics.genome import PromptGenome
    from src.evaluation.fitness import FitnessComponents
else:
    from ..utils.config import config
    from ..genetics.genome import PromptGenome
    from ..evaluation.fitness import FitnessComponents


@dataclass
class CacheEntry:
    """Represents a cached evaluation result."""
    prompt_hash: str
    prompt_text: str
    evaluation_results: List[Dict[str, Any]]
    fitness_components: Dict[str, Any]
    timestamp: float
    model_info: Dict[str, str]
    problem_set_hash: str


class EvaluationCache:
    """Persistent cache for evaluation results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize evaluation cache.
        
        Args:
            cache_dir: Directory to store cache files (defaults to config cache dir)
        """
        self.cache_dir = cache_dir or config.get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.evaluation_cache_file = self.cache_dir / "evaluation_cache.json"
        self.fitness_cache_file = self.cache_dir / "fitness_cache.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # In-memory caches
        self.evaluation_cache = {}
        self.fitness_cache = {}
        self.metadata = {
            'created': time.time(),
            'last_updated': time.time(),
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Load existing caches
        self.load_caches()
    
    def _create_prompt_hash(self, prompt_text: str) -> str:
        """Create hash for prompt text."""
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    
    def _create_problem_set_hash(self, problems: List[Dict[str, Any]]) -> str:
        """Create hash for problem set."""
        # Create a stable hash based on problem IDs and questions
        problem_data = []
        for problem in problems:
            problem_data.append({
                'id': problem.get('id', ''),
                'question': problem.get('question', ''),
                'final_answer': problem.get('final_answer')
            })
        
        # Sort to ensure consistent ordering
        problem_data.sort(key=lambda x: x['id'])
        
        problem_str = json.dumps(problem_data, sort_keys=True)
        return hashlib.sha256(problem_str.encode()).hexdigest()[:16]
    
    def get_cache_key(self, genome: PromptGenome, problems: List[Dict[str, Any]], 
                     model_info: Dict[str, str]) -> str:
        """
        Generate cache key for a genome-problems-model combination.
        
        Args:
            genome: The prompt genome
            problems: List of problems
            model_info: Model configuration info
            
        Returns:
            Cache key string
        """
        prompt_hash = self._create_prompt_hash(genome.to_text())
        problem_set_hash = self._create_problem_set_hash(problems)
        model_hash = hashlib.md5(json.dumps(model_info, sort_keys=True).encode()).hexdigest()[:8]
        
        return f"{prompt_hash}_{problem_set_hash}_{model_hash}"
    
    def has_evaluation(self, cache_key: str) -> bool:
        """Check if evaluation results are cached."""
        return cache_key in self.evaluation_cache
    
    def get_evaluation(self, cache_key: str) -> Optional[CacheEntry]:
        """
        Get cached evaluation results.
        
        Args:
            cache_key: Cache key
            
        Returns:
            CacheEntry if found, None otherwise
        """
        if cache_key in self.evaluation_cache:
            self.metadata['cache_hits'] += 1
            entry_data = self.evaluation_cache[cache_key]
            return CacheEntry(**entry_data)
        
        self.metadata['cache_misses'] += 1
        return None
    
    def store_evaluation(self, cache_key: str, genome: PromptGenome, 
                        problems: List[Dict[str, Any]], 
                        evaluation_results: List[Dict[str, Any]],
                        fitness_components: FitnessComponents,
                        model_info: Dict[str, str]):
        """
        Store evaluation results in cache.
        
        Args:
            cache_key: Cache key
            genome: The prompt genome
            problems: List of problems
            evaluation_results: Evaluation results
            fitness_components: Fitness components
            model_info: Model configuration info
        """
        entry = CacheEntry(
            prompt_hash=self._create_prompt_hash(genome.to_text()),
            prompt_text=genome.to_text(),
            evaluation_results=evaluation_results,
            fitness_components=asdict(fitness_components),
            timestamp=time.time(),
            model_info=model_info,
            problem_set_hash=self._create_problem_set_hash(problems)
        )
        
        self.evaluation_cache[cache_key] = asdict(entry)
        self.metadata['total_entries'] = len(self.evaluation_cache)
        self.metadata['last_updated'] = time.time()
    
    def get_fitness(self, genome: PromptGenome) -> Optional[float]:
        """
        Get cached fitness for a genome.
        
        Args:
            genome: The prompt genome
            
        Returns:
            Cached fitness or None
        """
        prompt_hash = self._create_prompt_hash(genome.to_text())
        return self.fitness_cache.get(prompt_hash)
    
    def store_fitness(self, genome: PromptGenome, fitness: float):
        """
        Store fitness in cache.
        
        Args:
            genome: The prompt genome
            fitness: Fitness score
        """
        prompt_hash = self._create_prompt_hash(genome.to_text())
        self.fitness_cache[prompt_hash] = fitness
    
    def save_caches(self):
        """Save caches to disk."""
        try:
            # Save evaluation cache
            with open(self.evaluation_cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f, indent=2)
            
            # Save fitness cache
            with open(self.fitness_cache_file, 'w') as f:
                json.dump(self.fitness_cache, f, indent=2)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Cache saved: {len(self.evaluation_cache)} evaluation entries, "
                  f"{len(self.fitness_cache)} fitness entries")
            
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_caches(self):
        """Load caches from disk."""
        try:
            # Load evaluation cache
            if self.evaluation_cache_file.exists():
                with open(self.evaluation_cache_file, 'r') as f:
                    self.evaluation_cache = json.load(f)
            
            # Load fitness cache
            if self.fitness_cache_file.exists():
                with open(self.fitness_cache_file, 'r') as f:
                    self.fitness_cache = json.load(f)
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    loaded_metadata = json.load(f)
                    self.metadata.update(loaded_metadata)
            
            print(f"Cache loaded: {len(self.evaluation_cache)} evaluation entries, "
                  f"{len(self.fitness_cache)} fitness entries")
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Reset caches on error
            self.evaluation_cache = {}
            self.fitness_cache = {}
    
    def clear_cache(self):
        """Clear all caches."""
        self.evaluation_cache.clear()
        self.fitness_cache.clear()
        self.metadata['total_entries'] = 0
        self.metadata['cache_hits'] = 0
        self.metadata['cache_misses'] = 0
        self.metadata['last_updated'] = time.time()
        
        # Remove cache files
        for cache_file in [self.evaluation_cache_file, self.fitness_cache_file, self.metadata_file]:
            if cache_file.exists():
                cache_file.unlink()
        
        print("Cache cleared")
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """
        Remove cache entries older than specified days.
        
        Args:
            max_age_days: Maximum age in days
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        # Clean evaluation cache
        old_keys = []
        for key, entry_data in self.evaluation_cache.items():
            if current_time - entry_data.get('timestamp', 0) > max_age_seconds:
                old_keys.append(key)
        
        for key in old_keys:
            del self.evaluation_cache[key]
        
        if old_keys:
            print(f"Removed {len(old_keys)} old cache entries")
            self.metadata['total_entries'] = len(self.evaluation_cache)
            self.metadata['last_updated'] = time.time()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.metadata['cache_hits'] + self.metadata['cache_misses']
        hit_rate = (self.metadata['cache_hits'] / total_requests 
                   if total_requests > 0 else 0.0)
        
        # Calculate cache size on disk
        cache_size_bytes = 0
        for cache_file in [self.evaluation_cache_file, self.fitness_cache_file, self.metadata_file]:
            if cache_file.exists():
                cache_size_bytes += cache_file.stat().st_size
        
        return {
            'evaluation_entries': len(self.evaluation_cache),
            'fitness_entries': len(self.fitness_cache),
            'total_requests': total_requests,
            'cache_hits': self.metadata['cache_hits'],
            'cache_misses': self.metadata['cache_misses'],
            'hit_rate': hit_rate,
            'cache_size_bytes': cache_size_bytes,
            'cache_size_mb': cache_size_bytes / (1024 * 1024),
            'created': self.metadata['created'],
            'last_updated': self.metadata['last_updated']
        }


# Global cache instance
evaluation_cache = EvaluationCache()


if __name__ == "__main__":
    # Test caching system
    print("Testing evaluation cache...")
    
    # Load vocabulary for testing
    from src.embeddings.vocabulary import vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create test cache
    test_cache = EvaluationCache()
    
    # Create test data
    test_genome = PromptGenome.from_text("Solve this step by step.")
    test_problems = [
        {'id': 'test_1', 'question': 'What is 2+2?', 'final_answer': 4.0},
        {'id': 'test_2', 'question': 'What is 3*5?', 'final_answer': 15.0}
    ]
    test_model_info = {'model': 'gpt-4', 'temperature': 0.0}
    
    # Test cache key generation
    cache_key = test_cache.get_cache_key(test_genome, test_problems, test_model_info)
    print(f"✅ Cache key: {cache_key}")
    
    # Test cache miss
    cached_entry = test_cache.get_evaluation(cache_key)
    print(f"✅ Cache miss (expected): {cached_entry is None}")
    
    # Create mock evaluation results
    from src.evaluation.fitness import FitnessComponents
    mock_results = [
        {'problem_id': 'test_1', 'is_correct': True, 'response_length': 50},
        {'problem_id': 'test_2', 'is_correct': False, 'response_length': 60}
    ]
    mock_fitness = FitnessComponents(
        accuracy=0.5, consistency=0.8, efficiency=0.9, 
        diversity_bonus=0.1, length_penalty=0.0, overall_fitness=0.65
    )
    
    # Store in cache
    test_cache.store_evaluation(cache_key, test_genome, test_problems, 
                               mock_results, mock_fitness, test_model_info)
    print("✅ Stored evaluation in cache")
    
    # Test cache hit
    cached_entry = test_cache.get_evaluation(cache_key)
    print(f"✅ Cache hit: {cached_entry is not None}")
    if cached_entry:
        print(f"   Cached fitness: {cached_entry.fitness_components['overall_fitness']}")
    
    # Test fitness caching
    test_cache.store_fitness(test_genome, 0.75)
    cached_fitness = test_cache.get_fitness(test_genome)
    print(f"✅ Fitness cache: {cached_fitness}")
    
    # Test statistics
    stats = test_cache.get_cache_statistics()
    print(f"✅ Cache statistics: {stats}")
    
    # Test save/load
    test_cache.save_caches()
    print("✅ Cache saved to disk")
    
    # Create new cache instance and load
    test_cache2 = EvaluationCache(test_cache.cache_dir)
    cached_entry2 = test_cache2.get_evaluation(cache_key)
    print(f"✅ Cache loaded from disk: {cached_entry2 is not None}")
    
    print("\n🎯 Evaluation cache tests completed successfully!")
