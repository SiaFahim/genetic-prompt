# Semantic Neighboring Mutation System Analysis
## 1. Semantic Neighboring Implementation Details

### Core Architecture
The semantic neighboring system is built around three main components:

**A. Word Embeddings (`src/embeddings/semantic_utils.py`)**
- Uses **GloVe embeddings** (Global Vectors for Word Representation) with 100-dimensional vectors
- Falls back to **random normalized embeddings** if GloVe files aren't available
- Implements **cosine similarity** calculation for measuring semantic distance between words

````python path=src/embeddings/semantic_utils.py mode=EXCERPT
def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
````

**B. Token Vocabulary (`src/embeddings/vocabulary.py`)**
- Converts text to **token IDs** and back using NLTK word tokenization
- Builds vocabulary from the GSM8K dataset (up to 10,000 most frequent tokens)
- Handles special tokens like `<UNK>`, `<PAD>`, `<START>`, `<END>`

**C. Semantic Neighborhoods (`src/embeddings/neighborhoods.py`)**
- Pre-computes **50 nearest neighbors** for each token using cosine similarity
- Stores neighborhoods as `Dict[token_id, List[token_id]]` for fast lookup
- Falls back to random neighbors when embeddings aren't available

````python path=src/embeddings/neighborhoods.py mode=EXCERPT
def _find_semantic_neighbors(self, target_token: str, target_vec: np.ndarray,
                           vocabulary: TokenVocabulary, 
                           embeddings: SimpleWordEmbeddings) -> List[int]:
    """Find semantic neighbors for a token."""
    similarities = []
    
    for other_token, other_token_id in vocabulary.token_to_id.items():
        # Skip self and special tokens
        if (other_token == target_token or 
            (other_token.startswith('<') and other_token.endswith('>'))):
            continue
        
        other_vec = embeddings.get_vector(other_token)
        if other_vec is not None:
            sim = embeddings.cosine_similarity(target_vec, other_vec)
            similarities.append((other_token_id, sim))
    
    # Sort by similarity and return top neighbors
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbor_ids = [token_id for token_id, _ in similarities[:self.n_neighbors]]
````

## 2. Integration with Mutation Process

### Semantic Mutation Strategy
The system implements **semantic mutation** as the primary mutation type, which works as follows:

**A. Token-Level Mutation (`src/genetics/mutation.py`)**
- Each token in a genome has a **mutation_rate** probability (default 10%) of being mutated
- When a token is selected for mutation, there's a **semantic_prob** probability (default 90%) of using a semantic neighbor vs. a random token
- This creates a balance between **exploitation** (semantic similarity) and **exploration** (randomness)

````python path=src/genetics/mutation.py mode=EXCERPT
def semantic_mutation(genome: PromptGenome, mutation_rate: float = 0.1, 
                     semantic_prob: float = 0.9) -> PromptGenome:
    """
    Perform semantic mutation using neighborhood information.
    """
    for i in range(len(mutated_genome.token_ids)):
        if random.random() < mutation_rate:
            original_token_id = mutated_genome.token_ids[i]
            
            # Get semantic neighbor
            if semantic_neighborhoods.neighborhoods_built:
                new_token_id = semantic_neighborhoods.get_neighbor(
                    original_token_id, semantic_prob
                )
            else:
                # Fallback to random mutation
                new_token_id = vocabulary.get_random_token_id()
````

**B. Neighbor Selection Process**
The `get_neighbor()` method implements the exploration vs. exploitation balance:

````python path=src/embeddings/neighborhoods.py mode=EXCERPT
def get_neighbor(self, token_id: int, semantic_prob: float = 0.9) -> int:
    """Get a neighbor for mutation."""
    # Check if we have neighbors for this token
    if token_id not in self.neighborhoods or not self.neighborhoods[token_id]:
        return self.vocabulary.get_random_token_id()
    
    # Decide whether to use semantic or random neighbor
    if random.random() < semantic_prob:
        # Use semantic neighbor (exploitation)
        return random.choice(self.neighborhoods[token_id])
    else:
        # Use random token (exploration)
        return self.vocabulary.get_random_token_id()
````

### Adaptive Parameters
The evolution system (`src/genetics/evolution.py`) dynamically adjusts mutation rates based on population diversity:

````python path=src/genetics/evolution.py mode=EXCERPT
# Adjust mutation rate based on diversity
if recent_diversity < 0.1:
    # Low diversity - increase mutation
    self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.1)
elif recent_diversity > 0.8:
    # High diversity - decrease mutation
    self.current_mutation_rate = max(0.05, self.current_mutation_rate * 0.9)
````

## 3. Technical Implementation

### Key Classes and Data Flow

**A. Global Instances**
- `vocabulary` (global instance in `src/embeddings/vocabulary.py:255`)
- `semantic_neighborhoods` (global instance in `src/embeddings/neighborhoods.py:232`)

**B. Data Flow Process:**
1. **Initialization**: Vocabulary is built from GSM8K dataset
2. **Embedding Loading**: GloVe embeddings are loaded or demo embeddings created
3. **Neighborhood Building**: For each token, find 50 most similar tokens using cosine similarity
4. **Genome Representation**: Prompts are stored as lists of token IDs in `PromptGenome` objects
5. **Mutation**: During evolution, tokens are replaced with semantic neighbors

**C. Configuration Parameters (`src/config/experiment_configs.py`)**
- `mutation_rate: float = 0.2` - Probability of applying mutation to an individual
- `mutation_type: MutationType = MutationType.SEMANTIC` - Default mutation strategy
- `n_neighbors: int = 50` - Number of semantic neighbors stored per token
- `semantic_prob: float = 0.9` - Probability of using semantic vs. random neighbor

### Performance Optimizations
- **Pre-computed neighborhoods**: All similarities calculated once during initialization
- **Fast lookup**: O(1) neighbor retrieval using dictionary storage
- **Fallback mechanisms**: Random mutations when semantic data unavailable

## 4. File References and Code Locations

### Core Implementation Files:
- **`src/embeddings/semantic_utils.py`** (Lines 25-177): Word embedding management and similarity calculations
- **`src/embeddings/neighborhoods.py`** (Lines 25-187): Semantic neighborhood building and neighbor selection
- **`src/embeddings/vocabulary.py`** (Lines 25-251): Token vocabulary and text encoding/decoding
- **`src/genetics/mutation.py`** (Lines 36-84): Semantic mutation implementation
- **`src/genetics/genome.py`** (Lines 26-99): Genome representation as token sequences

### Integration Points:
- **`src/genetics/evolution.py`** (Lines 84, 250, 268-280): Evolution configuration and adaptive parameters
- **`src/main_runner.py`** (Lines 66-72, 103): System initialization and configuration
- **`src/config/experiment_configs.py`** (Lines 49, 54): Default parameters and experiment presets

### Testing and Validation:
- **`scripts/test_semantic_system.py`** (Lines 82-104): Semantic mutation testing
- **`scripts/test_genetic_operations.py`**: Integration testing with genetic operations

## Summary

The semantic neighboring system creates a **biologically-inspired mutation mechanism** that maintains semantic coherence while allowing for evolutionary exploration. By using pre-computed word embeddings and cosine similarity, the system can intelligently replace words with semantically similar alternatives, leading to more meaningful prompt variations than pure random mutations. The 90/10 split between semantic and random neighbors provides an optimal balance between exploitation of semantic relationships and exploration of the search space.
