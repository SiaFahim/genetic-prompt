"""
Random seed prompt generator for genetic algorithm experiments.
Generates completely random token sequences for testing convergence from pure randomness.
"""

import random
import logging
from typing import List, Optional, Set
from pathlib import Path

from src.embeddings.glove_loader import GloVeLoader
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class RandomSeedGenerator:
    """Generates random seed prompts from vocabulary tokens."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize random seed generator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.glove_loader = GloVeLoader(config_path)
        
        # Configuration
        self.vocabulary_size = self.config.get('genetic_algorithm.random_seed_vocabulary_size', 1000)
        self.length_range = self.config.get('genetic_algorithm.random_seed_length_range', [5, 20])
        self.random_seed = self.config.get('experiment.random_seed', 42)
        
        # Vocabulary cache
        self._vocabulary = None
        self._filtered_vocabulary = None
        
        logger.info(f"RandomSeedGenerator initialized with vocab_size={self.vocabulary_size}, "
                   f"length_range={self.length_range}")
    
    def _load_vocabulary(self) -> List[str]:
        """
        Load vocabulary from GloVe embeddings.
        
        Returns:
            List of vocabulary words
        """
        if self._vocabulary is not None:
            return self._vocabulary
        
        try:
            # Load embeddings to get vocabulary
            word_to_idx, idx_to_word, embeddings = self.glove_loader.load_embeddings()
            
            # Get vocabulary words (limited by vocabulary_size)
            vocab_words = list(word_to_idx.keys())[:self.vocabulary_size]
            
            self._vocabulary = vocab_words
            logger.info(f"Loaded vocabulary of {len(vocab_words)} words")
            
            return vocab_words
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary from embeddings: {e}")
            # Fallback to basic English words
            return self._get_fallback_vocabulary()
    
    def _get_fallback_vocabulary(self) -> List[str]:
        """
        Get fallback vocabulary if embeddings are not available.
        
        Returns:
            List of basic English words
        """
        logger.warning("Using fallback vocabulary - embeddings not available")
        
        # Basic English words for fallback
        fallback_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "was", "are", "been", "has", "had", "were", "said",
            "each", "which", "their", "time", "will", "about", "if", "up", "out", "many",
            "then", "them", "these", "so", "some", "her", "would", "make", "like", "into",
            "him", "has", "two", "more", "very", "what", "know", "just", "first", "get",
            "over", "think", "where", "much", "go", "well", "were", "been", "have", "had",
            "who", "oil", "sit", "now", "find", "long", "down", "day", "did", "get",
            "come", "made", "may", "part", "number", "way", "use", "her", "many", "oil",
            "water", "words", "very", "after", "move", "right", "boy", "old", "too",
            "same", "tell", "does", "set", "three", "want", "air", "well", "also",
            "play", "small", "end", "put", "home", "read", "hand", "port", "large",
            "spell", "add", "even", "land", "here", "must", "big", "high", "such",
            "follow", "act", "why", "ask", "men", "change", "went", "light", "kind",
            "off", "need", "house", "picture", "try", "us", "again", "animal", "point",
            "mother", "world", "near", "build", "self", "earth", "father", "head",
            "stand", "own", "page", "should", "country", "found", "answer", "school",
            "grow", "study", "still", "learn", "plant", "cover", "food", "sun", "four",
            "between", "state", "keep", "eye", "never", "last", "let", "thought", "city",
            "tree", "cross", "farm", "hard", "start", "might", "story", "saw", "far",
            "sea", "draw", "left", "late", "run", "don't", "while", "press", "close",
            "night", "real", "life", "few", "north", "open", "seem", "together", "next",
            "white", "children", "begin", "got", "walk", "example", "ease", "paper",
            "group", "always", "music", "those", "both", "mark", "often", "letter",
            "until", "mile", "river", "car", "feet", "care", "second", "book", "carry",
            "took", "science", "eat", "room", "friend", "began", "idea", "fish", "mountain",
            "stop", "once", "base", "hear", "horse", "cut", "sure", "watch", "color",
            "face", "wood", "main", "enough", "plain", "girl", "usual", "young", "ready",
            "above", "ever", "red", "list", "though", "feel", "talk", "bird", "soon",
            "body", "dog", "family", "direct", "leave", "song", "measure", "door", "product",
            "black", "short", "numeral", "class", "wind", "question", "happen", "complete",
            "ship", "area", "half", "rock", "order", "fire", "south", "problem", "piece",
            "told", "knew", "pass", "since", "top", "whole", "king", "space", "heard",
            "best", "hour", "better", "during", "hundred", "five", "remember", "step",
            "early", "hold", "west", "ground", "interest", "reach", "fast", "verb",
            "sing", "listen", "six", "table", "travel", "less", "morning", "ten",
            "simple", "several", "vowel", "toward", "war", "lay", "against", "pattern",
            "slow", "center", "love", "person", "money", "serve", "appear", "road",
            "map", "rain", "rule", "govern", "pull", "cold", "notice", "voice", "unit",
            "power", "town", "fine", "certain", "fly", "fall", "lead", "cry", "dark",
            "machine", "note", "wait", "plan", "figure", "star", "box", "noun", "field",
            "rest", "correct", "able", "pound", "done", "beauty", "drive", "stood",
            "contain", "front", "teach", "week", "final", "gave", "green", "oh", "quick",
            "develop", "ocean", "warm", "free", "minute", "strong", "special", "mind",
            "behind", "clear", "tail", "produce", "fact", "street", "inch", "multiply",
            "nothing", "course", "stay", "wheel", "full", "force", "blue", "object",
            "decide", "surface", "deep", "moon", "island", "foot", "system", "busy",
            "test", "record", "boat", "common", "gold", "possible", "plane", "stead",
            "dry", "wonder", "laugh", "thousands", "ago", "ran", "check", "game", "shape",
            "equate", "hot", "miss", "brought", "heat", "snow", "tire", "bring", "yes",
            "distant", "fill", "east", "paint", "language", "among"
        ]
        
        # Limit to requested vocabulary size
        return fallback_words[:self.vocabulary_size]
    
    def _filter_vocabulary(self, vocabulary: List[str]) -> List[str]:
        """
        Filter vocabulary to remove problematic tokens.
        
        Args:
            vocabulary: Raw vocabulary list
            
        Returns:
            Filtered vocabulary list
        """
        if self._filtered_vocabulary is not None:
            return self._filtered_vocabulary
        
        # Filter out tokens that might cause issues
        filtered = []
        excluded_patterns = {
            # Very short tokens (might be punctuation)
            lambda w: len(w) < 2,
            # Tokens with special characters
            lambda w: not w.isalpha(),
            # Very long tokens (might be corrupted)
            lambda w: len(w) > 15,
            # Common punctuation that got through
            lambda w: w in {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}'}
        }
        
        for word in vocabulary:
            # Apply all filters
            if not any(filter_func(word) for filter_func in excluded_patterns):
                filtered.append(word)
        
        self._filtered_vocabulary = filtered
        logger.info(f"Filtered vocabulary from {len(vocabulary)} to {len(filtered)} words")
        
        return filtered
    
    def generate_random_seed_prompts(self, num_seeds: int = 50) -> List[str]:
        """
        Generate random seed prompts from vocabulary.
        
        Args:
            num_seeds: Number of random seed prompts to generate
            
        Returns:
            List of random seed prompt strings
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        logger.info(f"Generating {num_seeds} random seed prompts...")
        
        # Load and filter vocabulary
        vocabulary = self._load_vocabulary()
        filtered_vocab = self._filter_vocabulary(vocabulary)
        
        if len(filtered_vocab) < 10:
            raise ValueError(f"Insufficient vocabulary size: {len(filtered_vocab)}. Need at least 10 words.")
        
        random_seeds = []
        
        for i in range(num_seeds):
            # Generate random length within range
            length = random.randint(self.length_range[0], self.length_range[1])
            
            # Sample random tokens
            tokens = random.choices(filtered_vocab, k=length)
            
            # Join into prompt string
            prompt = " ".join(tokens)
            random_seeds.append(prompt)
        
        logger.info(f"Generated {len(random_seeds)} random seed prompts")
        logger.debug(f"Example random seeds: {random_seeds[:3]}")
        
        return random_seeds
    
    def validate_random_seeds(self, seeds: List[str]) -> bool:
        """
        Validate that random seeds are properly formatted.
        
        Args:
            seeds: List of seed prompts to validate
            
        Returns:
            True if all seeds are valid
        """
        if not seeds:
            logger.error("No seeds provided for validation")
            return False
        
        for i, seed in enumerate(seeds):
            # Check basic format
            if not isinstance(seed, str):
                logger.error(f"Seed {i} is not a string: {type(seed)}")
                return False
            
            if not seed.strip():
                logger.error(f"Seed {i} is empty or whitespace only")
                return False
            
            # Check token count
            tokens = seed.split()
            if len(tokens) < self.length_range[0] or len(tokens) > self.length_range[1]:
                logger.error(f"Seed {i} has invalid length {len(tokens)}, expected {self.length_range}")
                return False
        
        logger.info(f"Validated {len(seeds)} random seed prompts successfully")
        return True
