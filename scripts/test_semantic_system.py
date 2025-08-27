#!/usr/bin/env python3
"""
Test the complete semantic system for the genetic algorithm.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import config
from src.embeddings.semantic_utils import SimpleWordEmbeddings
from src.embeddings.vocabulary import TokenVocabulary
from src.embeddings.neighborhoods import SemanticNeighborhoods


def test_complete_semantic_system():
    """Test the complete semantic system integration."""
    print("🧪 Testing Complete Semantic System")
    print("=" * 50)
    
    data_dir = config.get_data_dir()
    embeddings_dir = data_dir / "embeddings"
    
    # Test 1: Load vocabulary
    print("\n1. Testing Vocabulary System...")
    vocabulary = TokenVocabulary()
    vocab_file = embeddings_dir / "vocabulary.pkl"
    
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print(f"✅ Loaded vocabulary with {len(vocabulary.token_to_id)} tokens")
    else:
        print("❌ Vocabulary file not found")
        return False
    
    # Test 2: Load embeddings
    print("\n2. Testing Embeddings System...")
    embeddings = SimpleWordEmbeddings()
    embeddings_file = embeddings_dir / "word_embeddings.pkl"
    
    if embeddings_file.exists():
        embeddings.load_embeddings(embeddings_file)
        print(f"✅ Loaded embeddings for {embeddings.vocab_size} words")
    else:
        print("❌ Embeddings file not found")
        return False
    
    # Test 3: Load neighborhoods
    print("\n3. Testing Neighborhoods System...")
    neighborhoods = SemanticNeighborhoods()
    neighborhoods_file = embeddings_dir / "semantic_neighborhoods.pkl"
    
    if neighborhoods_file.exists():
        neighborhoods.load_neighborhoods(neighborhoods_file)
        neighborhoods.vocabulary = vocabulary  # Set vocabulary reference
        print(f"✅ Loaded neighborhoods for {len(neighborhoods.neighborhoods)} tokens")
    else:
        print("❌ Neighborhoods file not found")
        return False
    
    # Test 4: Text encoding/decoding
    print("\n4. Testing Text Processing...")
    test_texts = [
        "Let's solve this step by step.",
        "Calculate the total number of items.",
        "Find the answer to this problem.",
        "Think through this carefully."
    ]
    
    for text in test_texts:
        token_ids = vocabulary.encode_text(text)
        decoded = vocabulary.decode_ids(token_ids)
        print(f"  Original: {text}")
        print(f"  Encoded:  {token_ids}")
        print(f"  Decoded:  {decoded}")
        print()
    
    # Test 5: Semantic mutations
    print("5. Testing Semantic Mutations...")
    test_prompt = "Let's solve this problem step by step to find the answer."
    token_ids = vocabulary.encode_text(test_prompt)
    
    print(f"Original prompt: {test_prompt}")
    print(f"Token IDs: {token_ids}")
    
    # Perform semantic mutations
    for i in range(3):
        mutated_ids = token_ids.copy()
        
        # Mutate 2-3 random tokens
        import random
        n_mutations = random.randint(2, 3)
        positions = random.sample(range(len(mutated_ids)), n_mutations)
        
        for pos in positions:
            original_id = mutated_ids[pos]
            neighbor_id = neighborhoods.get_neighbor(original_id, semantic_prob=0.8)
            mutated_ids[pos] = neighbor_id
        
        mutated_text = vocabulary.decode_ids(mutated_ids)
        print(f"  Mutation {i+1}: {mutated_text}")
    
    # Test 6: Neighborhood quality
    print("\n6. Testing Neighborhood Quality...")
    test_words = ['problem', 'solve', 'calculate', 'answer', 'step', 'find']
    
    for word in test_words:
        if word in vocabulary.token_to_id:
            token_id = vocabulary.token_to_id[word]
            neighbors = neighborhoods.get_multiple_neighbors(token_id, 5)
            neighbor_words = [vocabulary.id_to_token[nid] for nid in neighbors]
            print(f"  '{word}' -> {neighbor_words}")
    
    # Test 7: System statistics
    print("\n7. System Statistics...")
    vocab_stats = {
        'vocabulary_size': len(vocabulary.token_to_id),
        'embedding_coverage': len([w for w in vocabulary.token_to_id.keys() 
                                 if embeddings.get_vector(w) is not None]),
        'most_frequent_tokens': list(vocabulary.token_to_id.keys())[:10]
    }
    
    neighborhood_stats = neighborhoods.get_neighborhood_stats()
    
    print("  Vocabulary Statistics:")
    for key, value in vocab_stats.items():
        print(f"    {key}: {value}")
    
    print("  Neighborhood Statistics:")
    for key, value in neighborhood_stats.items():
        print(f"    {key}: {value}")
    
    # Test 8: Performance test
    print("\n8. Performance Test...")
    import time
    
    # Test encoding speed
    start_time = time.time()
    for _ in range(100):
        token_ids = vocabulary.encode_text("This is a test sentence for performance.")
    encoding_time = time.time() - start_time
    
    # Test mutation speed
    start_time = time.time()
    for _ in range(100):
        for token_id in token_ids:
            neighbor = neighborhoods.get_neighbor(token_id)
    mutation_time = time.time() - start_time
    
    print(f"  Encoding 100 sentences: {encoding_time:.3f}s")
    print(f"  Getting 100 neighbors: {mutation_time:.3f}s")
    
    print("\n✅ All semantic system tests completed successfully!")
    return True


if __name__ == "__main__":
    success = test_complete_semantic_system()
    if success:
        print("\n🎉 Semantic system is ready for genetic algorithm!")
    else:
        print("\n❌ Semantic system tests failed!")
        sys.exit(1)
