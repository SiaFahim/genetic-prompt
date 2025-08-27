"""
Seed prompt management system for loading, organizing, and managing collections.
"""

import json
import random
import time
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import asdict

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.seeds.prompt_categories import PromptCategory, PromptCategoryManager
    from src.seeds.base_seeds import BaseSeedCollection, SeedPrompt
    from src.seeds.seed_validation import SeedValidator
    from src.utils.config import config
else:
    from .prompt_categories import PromptCategory, PromptCategoryManager
    from .base_seeds import BaseSeedCollection, SeedPrompt
    from .seed_validation import SeedValidator
    from ..utils.config import config


class SeedManager:
    """Manages seed prompt collections with loading, validation, and selection."""
    
    def __init__(self):
        """Initialize seed manager."""
        self.seeds_dir = config.get_seeds_dir()
        self.seeds_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.category_manager = PromptCategoryManager()
        self.validator = SeedValidator()
        
        # Seed collections
        self.base_collection = BaseSeedCollection()
        self.custom_collections: Dict[str, List[SeedPrompt]] = {}
        
        # Load any existing custom collections
        self._load_custom_collections()
    
    def get_base_seeds(self) -> List[SeedPrompt]:
        """Get the base seed collection."""
        return self.base_collection.get_all_seeds()
    
    def get_seeds_by_category(self, category: PromptCategory, 
                             collection_name: str = "base") -> List[SeedPrompt]:
        """
        Get seeds filtered by category.
        
        Args:
            category: Category to filter by
            collection_name: Name of collection ("base" or custom name)
            
        Returns:
            List of seeds in the specified category
        """
        if collection_name == "base":
            return self.base_collection.get_seeds_by_category(category)
        elif collection_name in self.custom_collections:
            return [seed for seed in self.custom_collections[collection_name] 
                   if seed.category == category]
        else:
            return []
    
    def create_balanced_subset(self, size: int, 
                              collection_name: str = "base",
                              strategy: str = "balanced") -> List[SeedPrompt]:
        """
        Create a balanced subset of seeds.
        
        Args:
            size: Number of seeds to select
            collection_name: Collection to select from
            strategy: Selection strategy ("balanced", "random", "diverse")
            
        Returns:
            List of selected seeds
        """
        # Get source collection
        if collection_name == "base":
            source_seeds = self.base_collection.get_all_seeds()
        elif collection_name in self.custom_collections:
            source_seeds = self.custom_collections[collection_name]
        else:
            raise ValueError(f"Collection not found: {collection_name}")
        
        if size >= len(source_seeds):
            return source_seeds.copy()
        
        if strategy == "balanced":
            return self._create_balanced_selection(source_seeds, size)
        elif strategy == "random":
            return random.sample(source_seeds, size)
        elif strategy == "diverse":
            return self._create_diverse_selection(source_seeds, size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _create_balanced_selection(self, seeds: List[SeedPrompt], size: int) -> List[SeedPrompt]:
        """Create balanced selection maintaining category proportions."""
        # Get category distribution
        category_counts = {}
        for seed in seeds:
            category_counts[seed.category] = category_counts.get(seed.category, 0) + 1
        
        # Calculate target counts for each category
        total_seeds = len(seeds)
        target_counts = {}
        
        for category, count in category_counts.items():
            proportion = count / total_seeds
            target_count = max(1, round(proportion * size))
            target_counts[category] = target_count
        
        # Adjust if total exceeds size
        total_target = sum(target_counts.values())
        if total_target > size:
            # Reduce largest categories first
            while total_target > size:
                largest_category = max(target_counts, key=target_counts.get)
                target_counts[largest_category] -= 1
                total_target -= 1
        
        # Select seeds from each category
        selected_seeds = []
        seeds_by_category = {}
        
        for seed in seeds:
            if seed.category not in seeds_by_category:
                seeds_by_category[seed.category] = []
            seeds_by_category[seed.category].append(seed)
        
        for category, target_count in target_counts.items():
            if category in seeds_by_category:
                available_seeds = seeds_by_category[category]
                selected_count = min(target_count, len(available_seeds))
                selected_seeds.extend(random.sample(available_seeds, selected_count))
        
        return selected_seeds
    
    def _create_diverse_selection(self, seeds: List[SeedPrompt], size: int) -> List[SeedPrompt]:
        """Create diverse selection maximizing text diversity."""
        if size >= len(seeds):
            return seeds.copy()
        
        selected = []
        remaining = seeds.copy()
        
        # Start with a random seed
        first_seed = random.choice(remaining)
        selected.append(first_seed)
        remaining.remove(first_seed)
        
        # Greedily select most diverse remaining seeds
        while len(selected) < size and remaining:
            best_seed = None
            best_diversity = -1
            
            for candidate in remaining:
                # Calculate minimum similarity to selected seeds
                min_similarity = float('inf')
                for selected_seed in selected:
                    similarity = self._calculate_text_similarity(
                        candidate.text, selected_seed.text
                    )
                    min_similarity = min(min_similarity, similarity)
                
                # Higher diversity score for lower similarity
                diversity_score = 1.0 - min_similarity
                
                if diversity_score > best_diversity:
                    best_diversity = diversity_score
                    best_seed = candidate
            
            if best_seed:
                selected.append(best_seed)
                remaining.remove(best_seed)
        
        return selected
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        import re
        
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def save_collection(self, seeds: List[SeedPrompt], collection_name: str,
                       description: str = "") -> str:
        """
        Save a seed collection to file.
        
        Args:
            seeds: List of seeds to save
            collection_name: Name for the collection
            description: Optional description
            
        Returns:
            Path to saved file
        """
        collection_data = {
            'name': collection_name,
            'description': description,
            'created_at': time.time(),
            'seed_count': len(seeds),
            'seeds': [
                {
                    'id': seed.id,
                    'text': seed.text,
                    'category': seed.category.value,
                    'description': seed.description,
                    'expected_strength': seed.expected_strength,
                    'variations': seed.variations
                }
                for seed in seeds
            ]
        }
        
        # Save to file
        collection_file = self.seeds_dir / f"{collection_name}.json"
        with open(collection_file, 'w') as f:
            json.dump(collection_data, f, indent=2)
        
        # Add to custom collections
        self.custom_collections[collection_name] = seeds
        
        print(f"💾 Saved collection '{collection_name}' with {len(seeds)} seeds")
        return str(collection_file)
    
    def load_collection(self, collection_name: str) -> Optional[List[SeedPrompt]]:
        """
        Load a seed collection from file.
        
        Args:
            collection_name: Name of collection to load
            
        Returns:
            List of seeds or None if not found
        """
        collection_file = self.seeds_dir / f"{collection_name}.json"
        
        if not collection_file.exists():
            return None
        
        try:
            with open(collection_file, 'r') as f:
                data = json.load(f)
            
            seeds = []
            for seed_data in data['seeds']:
                seed = SeedPrompt(
                    id=seed_data['id'],
                    text=seed_data['text'],
                    category=PromptCategory(seed_data['category']),
                    description=seed_data['description'],
                    expected_strength=seed_data['expected_strength'],
                    variations=seed_data['variations']
                )
                seeds.append(seed)
            
            self.custom_collections[collection_name] = seeds
            print(f"📂 Loaded collection '{collection_name}' with {len(seeds)} seeds")
            return seeds
            
        except Exception as e:
            print(f"❌ Error loading collection '{collection_name}': {e}")
            return None
    
    def _load_custom_collections(self):
        """Load all custom collections from the seeds directory."""
        for collection_file in self.seeds_dir.glob("*.json"):
            collection_name = collection_file.stem
            if collection_name != "base":  # Skip base collection
                self.load_collection(collection_name)
    
    def list_collections(self) -> Dict[str, Dict[str, Any]]:
        """List all available collections with metadata."""
        collections = {
            'base': {
                'name': 'Base Collection',
                'description': 'Default high-quality seed prompts',
                'seed_count': len(self.base_collection.get_all_seeds()),
                'validation_score': None  # Could add validation here
            }
        }
        
        for name, seeds in self.custom_collections.items():
            collections[name] = {
                'name': name,
                'description': f'Custom collection with {len(seeds)} seeds',
                'seed_count': len(seeds),
                'validation_score': None
            }
        
        return collections
    
    def validate_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Validate a specific collection.
        
        Args:
            collection_name: Name of collection to validate
            
        Returns:
            Validation metrics or None if collection not found
        """
        if collection_name == "base":
            seeds = self.base_collection.get_all_seeds()
        elif collection_name in self.custom_collections:
            seeds = self.custom_collections[collection_name]
        else:
            return None
        
        metrics = self.validator.validate_collection(seeds)
        return asdict(metrics)
    
    def get_collection_statistics(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a collection."""
        if collection_name == "base":
            seeds = self.base_collection.get_all_seeds()
        elif collection_name in self.custom_collections:
            seeds = self.custom_collections[collection_name]
        else:
            return None
        
        # Category distribution
        category_dist = {}
        for seed in seeds:
            category_dist[seed.category.value] = category_dist.get(seed.category.value, 0) + 1
        
        # Length statistics
        lengths = [len(seed.text) for seed in seeds]
        word_counts = [len(seed.text.split()) for seed in seeds]
        
        return {
            'total_seeds': len(seeds),
            'category_distribution': category_dist,
            'length_stats': {
                'char_mean': sum(lengths) / len(lengths),
                'char_min': min(lengths),
                'char_max': max(lengths),
                'word_mean': sum(word_counts) / len(word_counts),
                'word_min': min(word_counts),
                'word_max': max(word_counts)
            }
        }


if __name__ == "__main__":
    # Test seed manager
    print("Testing seed manager...")
    
    import time
    
    # Create manager
    manager = SeedManager()
    
    # Test base collection access
    base_seeds = manager.get_base_seeds()
    print(f"✅ Base collection: {len(base_seeds)} seeds")
    
    # Test category filtering
    step_seeds = manager.get_seeds_by_category(PromptCategory.STEP_BY_STEP)
    print(f"✅ Step-by-step seeds: {len(step_seeds)}")
    
    # Test balanced subset creation
    balanced_subset = manager.create_balanced_subset(20, strategy="balanced")
    print(f"✅ Balanced subset: {len(balanced_subset)} seeds")
    
    # Test diverse subset creation
    diverse_subset = manager.create_balanced_subset(15, strategy="diverse")
    print(f"✅ Diverse subset: {len(diverse_subset)} seeds")
    
    # Test collection saving
    test_collection = balanced_subset[:10]
    saved_path = manager.save_collection(
        test_collection, 
        "test_collection", 
        "Test collection for validation"
    )
    print(f"✅ Collection saved: {Path(saved_path).name}")
    
    # Test collection loading
    loaded_seeds = manager.load_collection("test_collection")
    print(f"✅ Collection loaded: {len(loaded_seeds) if loaded_seeds else 0} seeds")
    
    # Test collection listing
    collections = manager.list_collections()
    print(f"✅ Available collections: {list(collections.keys())}")
    
    # Test validation
    validation = manager.validate_collection("base")
    if validation:
        print(f"✅ Base collection validation score: {validation['overall_score']:.3f}")
    
    # Test statistics
    stats = manager.get_collection_statistics("base")
    if stats:
        print(f"✅ Base collection statistics: {stats['total_seeds']} seeds")
    
    print("\n🎯 Seed manager tests completed successfully!")
