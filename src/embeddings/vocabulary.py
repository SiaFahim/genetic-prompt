"""
Token vocabulary management for the genetic algorithm.
Handles tokenization and token-ID mappings.
"""

import pickle
import json
from typing import Dict, List, Optional, Set
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import config
    from src.utils.dataset import gsm8k_dataset
else:
    from ..utils.config import config
    from ..utils.dataset import gsm8k_dataset


class TokenVocabulary:
    """Manages token vocabulary and ID mappings."""
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize token vocabulary.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_frequencies = {}
        self.vocab_built = False
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab')

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word tokenization
        tokens = word_tokenize(text.lower())
        # Filter out non-alphabetic tokens and very short tokens
        tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
        return tokens
    
    def build_vocabulary_from_dataset(self):
        """Build vocabulary from the GSM8K dataset."""
        print("Building vocabulary from GSM8K dataset...")
        
        # Load dataset
        try:
            raw_dataset = gsm8k_dataset.load_raw_dataset()
            train_data = raw_dataset['train']
            test_data = raw_dataset['test']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating basic vocabulary instead...")
            self._create_basic_vocabulary()
            return
        
        # Count token frequencies
        token_counts = {}
        total_texts = 0
        
        # Process training data
        for item in train_data:
            question_tokens = self.tokenize_text(item['question'])
            answer_tokens = self.tokenize_text(item['answer'])
            
            for token in question_tokens + answer_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            total_texts += 1
            if total_texts % 1000 == 0:
                print(f"Processed {total_texts} training examples...")
        
        # Process test data
        for item in test_data:
            question_tokens = self.tokenize_text(item['question'])
            answer_tokens = self.tokenize_text(item['answer'])
            
            for token in question_tokens + answer_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            total_texts += 1
        
        print(f"Processed {total_texts} total examples")
        print(f"Found {len(token_counts)} unique tokens")
        
        # Select top tokens by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        selected_tokens = sorted_tokens[:self.vocab_size]
        
        # Build mappings
        self.token_frequencies = dict(selected_tokens)
        
        # Add special tokens
        special_tokens = ['<UNK>', '<PAD>', '<START>', '<END>']
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Add vocabulary tokens
        for i, (token, freq) in enumerate(selected_tokens):
            token_id = i + len(special_tokens)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.token_to_id)} tokens")
        print(f"Most frequent tokens: {[token for token, _ in selected_tokens[:10]]}")
    
    def _create_basic_vocabulary(self):
        """Create a basic vocabulary for demonstration."""
        basic_tokens = [
            # Special tokens
            '<UNK>', '<PAD>', '<START>', '<END>',
            # Common words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
            # Math-related words
            'problem', 'solve', 'calculate', 'find', 'answer', 'question', 'step', 'solution',
            'number', 'value', 'result', 'total', 'sum', 'difference', 'product', 'quotient',
            'add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'times', 'equals',
            'more', 'less', 'than', 'equal', 'greater', 'smaller', 'bigger', 'larger',
            # Common verbs
            'make', 'take', 'give', 'get', 'go', 'come', 'see', 'know', 'think', 'say',
            'tell', 'ask', 'work', 'play', 'run', 'walk', 'sit', 'stand', 'look', 'find',
            # Adjectives
            'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'new', 'old',
            'young', 'right', 'wrong', 'true', 'false', 'same', 'different', 'easy', 'hard'
        ]
        
        # Extend to vocab_size
        while len(basic_tokens) < self.vocab_size:
            basic_tokens.append(f"token_{len(basic_tokens)}")
        
        # Build mappings
        for i, token in enumerate(basic_tokens[:self.vocab_size]):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            self.token_frequencies[token] = max(1, 1000 - i)  # Decreasing frequency
        
        self.vocab_built = True
        print(f"Created basic vocabulary with {len(self.token_to_id)} tokens")
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary_from_dataset() first.")
        
        tokens = self.tokenize_text(text)
        token_ids = []
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0))
            token_ids.append(token_id)
        
        return token_ids
    
    def decode_ids(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary_from_dataset() first.")
        
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<UNK>')
            if token not in ['<PAD>', '<START>', '<END>']:
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def get_token_frequency(self, token: str) -> int:
        """Get frequency of a token."""
        return self.token_frequencies.get(token, 0)
    
    def get_random_token_id(self) -> int:
        """Get a random token ID from vocabulary."""
        import random
        return random.randint(0, len(self.token_to_id) - 1)
    
    def save_vocabulary(self, filepath: Path):
        """Save vocabulary to file."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_frequencies': self.token_frequencies,
            'vocab_built': self.vocab_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: Path):
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.token_frequencies = data['token_frequencies']
        self.vocab_built = data['vocab_built']
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(self.token_to_id)}")


# Global vocabulary instance
vocabulary = TokenVocabulary()


if __name__ == "__main__":
    # Test vocabulary building
    print("Testing token vocabulary system...")
    
    # Build vocabulary
    vocabulary.build_vocabulary_from_dataset()
    
    # Test encoding/decoding
    test_text = "Let's solve this step by step to find the answer."
    print(f"\nOriginal text: {test_text}")
    
    token_ids = vocabulary.encode_text(test_text)
    print(f"Token IDs: {token_ids}")
    
    decoded_text = vocabulary.decode_ids(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    # Show some vocabulary statistics
    print(f"\nVocabulary statistics:")
    print(f"Total tokens: {len(vocabulary.token_to_id)}")
    print(f"Most frequent tokens:")
    sorted_by_freq = sorted(vocabulary.token_frequencies.items(), 
                           key=lambda x: x[1], reverse=True)
    for token, freq in sorted_by_freq[:10]:
        print(f"  {token}: {freq}")
    
    # Save vocabulary
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    vocabulary.save_vocabulary(vocab_file)
    
    print("\n✅ Token vocabulary system test completed!")
