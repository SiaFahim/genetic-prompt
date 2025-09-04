"""
Semantic utilities for genetic algorithm operations.
Provides high-level interface for semantic mutations and operations.
"""

import random
from typing import List, Optional, Dict, Any
import logging

from src.embeddings.neighborhoods import SemanticNeighborhoods
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class SemanticMutator:
    """High-level interface for semantic mutations in genetic algorithm."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize semantic mutator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.neighborhoods = SemanticNeighborhoods(config_path)
        
        # Configuration
        self.semantic_prob = self.config.get('mutation.semantic_neighbor_prob', 0.9)
        self.fallback_enabled = self.config.get('embeddings.fallback_enabled', True)
        
        # Cache for performance
        self._vocabulary = None
        self._vocab_set = None
    
    def get_vocabulary(self) -> List[str]:
        """Get the complete vocabulary."""
        if self._vocabulary is None:
            self._vocabulary = self.neighborhoods.get_vocabulary()
            self._vocab_set = set(self._vocabulary)
        
        return self._vocabulary
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        if self._vocab_set is None:
            self.get_vocabulary()  # This will set _vocab_set
        
        return word in self._vocab_set
    
    def mutate_word(self, word: str, semantic_prob: Optional[float] = None) -> str:
        """
        Mutate a single word using semantic neighborhoods.
        
        Args:
            word: Word to mutate
            semantic_prob: Probability of semantic vs random mutation (overrides default)
            
        Returns:
            Mutated word
        """
        if semantic_prob is None:
            semantic_prob = self.semantic_prob
        
        # Try semantic mutation first
        if random.random() < semantic_prob:
            neighbors = self.neighborhoods.get_neighbors(word)
            if neighbors:
                return random.choice(neighbors)
        
        # Fallback to random word
        if self.fallback_enabled:
            vocabulary = self.get_vocabulary()
            return random.choice(vocabulary)
        
        # If no fallback, return original word
        return word
    
    def mutate_tokens(self, tokens: List[str], mutation_prob: float = 0.002,
                     semantic_prob: Optional[float] = None) -> List[str]:
        """
        Mutate a list of tokens.
        
        Args:
            tokens: List of tokens to mutate
            mutation_prob: Probability of mutating each token
            semantic_prob: Probability of semantic vs random mutation
            
        Returns:
            List of mutated tokens
        """
        mutated_tokens = []
        mutations_made = 0
        
        for token in tokens:
            if random.random() < mutation_prob:
                mutated_token = self.mutate_word(token, semantic_prob)
                mutated_tokens.append(mutated_token)
                mutations_made += 1
            else:
                mutated_tokens.append(token)
        
        logger.debug(f"Mutated {mutations_made}/{len(tokens)} tokens")
        return mutated_tokens
    
    def get_similar_words(self, word: str, count: int = 5) -> List[str]:
        """
        Get similar words to a given word.
        
        Args:
            word: Reference word
            count: Number of similar words to return
            
        Returns:
            List of similar words
        """
        return self.neighborhoods.get_neighbors(word, count=count)
    
    def get_diverse_words(self, reference_words: List[str], count: int = 10) -> List[str]:
        """
        Get words that are diverse from reference words.
        
        Args:
            reference_words: List of words to be different from
            count: Number of diverse words to return
            
        Returns:
            List of diverse words
        """
        vocabulary = self.get_vocabulary()
        return self.neighborhoods.get_diverse_words(reference_words, vocabulary, count)
    
    def calculate_diversity(self, word_lists: List[List[str]]) -> float:
        """
        Calculate diversity between multiple word lists.
        
        Args:
            word_lists: List of word lists to compare
            
        Returns:
            Diversity score (0 = identical, 1 = completely different)
        """
        if len(word_lists) < 2:
            return 0.0
        
        # Calculate pairwise Jaccard distances
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(word_lists)):
            for j in range(i + 1, len(word_lists)):
                set1 = set(word_lists[i])
                set2 = set(word_lists[j])
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    jaccard_distance = 1 - jaccard_similarity
                    total_distance += jaccard_distance
                    comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def find_replacement_candidates(self, original_word: str, context_words: List[str],
                                  count: int = 5) -> List[str]:
        """
        Find good replacement candidates for a word given context.
        
        Args:
            original_word: Word to replace
            context_words: Context words to consider
            count: Number of candidates to return
            
        Returns:
            List of replacement candidates
        """
        # Get semantic neighbors of the original word
        neighbors = self.neighborhoods.get_neighbors(original_word, count=count * 2)
        
        if not neighbors:
            # Fallback to random words
            vocabulary = self.get_vocabulary()
            return random.sample(vocabulary, min(count, len(vocabulary)))
        
        # If we have context, try to find words that fit well
        if context_words:
            # Score candidates based on similarity to context
            candidate_scores = []
            
            for candidate in neighbors:
                total_similarity = 0.0
                valid_comparisons = 0
                
                for context_word in context_words:
                    if context_word != original_word:  # Don't compare with self
                        similarity = self.neighborhoods.get_similarity(candidate, context_word)
                        if similarity is not None:
                            total_similarity += similarity
                            valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_similarity = total_similarity / valid_comparisons
                    candidate_scores.append((candidate, avg_similarity))
            
            if candidate_scores:
                # Sort by average similarity to context (descending)
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                return [word for word, _ in candidate_scores[:count]]
        
        # If no context or context-based scoring failed, return top neighbors
        return neighbors[:count]
    
    def validate_mutation(self, original_tokens: List[str], 
                         mutated_tokens: List[str]) -> Dict[str, Any]:
        """
        Validate a mutation and return statistics.
        
        Args:
            original_tokens: Original token list
            mutated_tokens: Mutated token list
            
        Returns:
            Dictionary with validation statistics
        """
        if len(original_tokens) != len(mutated_tokens):
            logger.warning("Token lists have different lengths")
        
        changes = 0
        semantic_changes = 0
        invalid_words = 0
        
        min_length = min(len(original_tokens), len(mutated_tokens))
        
        for i in range(min_length):
            original = original_tokens[i]
            mutated = mutated_tokens[i]
            
            if original != mutated:
                changes += 1
                
                # Check if it's a semantic change
                if self.is_valid_word(mutated):
                    neighbors = self.neighborhoods.get_neighbors(original)
                    if mutated in neighbors:
                        semantic_changes += 1
                else:
                    invalid_words += 1
        
        stats = {
            'total_tokens': len(original_tokens),
            'changes_made': changes,
            'semantic_changes': semantic_changes,
            'random_changes': changes - semantic_changes,
            'invalid_words': invalid_words,
            'change_rate': changes / len(original_tokens) if original_tokens else 0,
            'semantic_rate': semantic_changes / changes if changes > 0 else 0
        }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the semantic system."""
        vocabulary = self.get_vocabulary()
        
        # Sample some words to check neighborhood quality
        sample_size = min(100, len(vocabulary))
        sample_words = random.sample(vocabulary, sample_size)
        
        total_neighbors = 0
        words_with_neighbors = 0
        
        for word in sample_words:
            neighbors = self.neighborhoods.get_neighbors(word)
            if neighbors:
                total_neighbors += len(neighbors)
                words_with_neighbors += 1
        
        avg_neighbors = total_neighbors / words_with_neighbors if words_with_neighbors > 0 else 0
        
        stats = {
            'vocabulary_size': len(vocabulary),
            'sample_size': sample_size,
            'words_with_neighbors': words_with_neighbors,
            'avg_neighbors_per_word': avg_neighbors,
            'semantic_probability': self.semantic_prob,
            'fallback_enabled': self.fallback_enabled
        }
        
        return stats
