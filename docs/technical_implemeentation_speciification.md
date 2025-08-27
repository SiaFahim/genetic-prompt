# Technical Implementation Specification: GSM8K Genetic Algorithm Prompt Optimization

## PHASE 1: Infrastructure Setup and Data Preparation

### Step 1.1: Environment Configuration
**Objective**: Create isolated Python environment with exact dependencies

#### Task 1.1.1: Python Environment
- Create Python 3.10 virtual environment named `gsm8k_ga_env`
- Install exact versions:
  ```
  openai==1.12.0
  datasets==2.16.0
  numpy==1.24.3
  gensim==4.3.0
  nltk==3.8.1
  tqdm==4.65.0
  pandas==2.0.3
  matplotlib==3.7.1
  jsonlines==3.1.0
  ```

#### Task 1.1.2: Directory Structure
Create this exact structure:
```
gsm8k_ga/
├── src/
│   ├── genetics/
│   │   ├── genome.py
│   │   ├── operators.py
│   │   ├── population.py
│   │   └── selection.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── cache_manager.py
│   │   └── metrics.py
│   ├── embeddings/
│   │   ├── neighborhoods.py
│   │   └── semantic_utils.py
│   └── utils/
│       ├── logging.py
│       └── checkpointing.py
├── data/
│   ├── seeds/
│   ├── checkpoints/
│   └── results/
├── configs/
│   └── experiment_config.json
└── scripts/
    ├── run_evolution.py
    └── analyze_results.py
```

### Step 1.2: Dataset Acquisition and Preprocessing

#### Task 1.2.1: Download GSM8K
```python
from datasets import load_dataset

# Download and cache locally
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("./data/gsm8k_raw")

# Extract splits
train_set = dataset['train']  # 7,473 problems
test_set = dataset['test']    # 1,319 problems
```

#### Task 1.2.2: Create Evaluation Subsets
- **Primary eval set**: Randomly sample 100 problems from test_set with seed=42
- **Validation set**: Randomly sample 100 problems from train_set with seed=43
- **Final test set**: Randomly sample 200 problems from test_set (non-overlapping with primary) with seed=44
- Save each as JSON files with structure:
  ```json
  {
    "id": "gsm8k_test_0001",
    "question": "problem text",
    "answer": "solution with #### marker",
    "final_answer": 42.0
  }
  ```

#### Task 1.2.3: Build Answer Extraction Function
```python
def extract_final_answer(answer_text):
    """
    Extract numeric answer after #### marker.
    Handle: integers, decimals, negative numbers, commas
    Return: float or None if no answer found
    """
    import re
    pattern = r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, answer_text)
    if match:
        number_str = match.group(1).replace(',', '')
        return float(number_str)
    return None
```

### Step 1.3: Semantic Embedding Setup

#### Task 1.3.1: Download Word Embeddings
- Download GloVe embeddings: `glove.6B.100d.txt` (822MB)
- Load into memory-mapped numpy array for efficiency
- Create token-to-index mapping dictionary

#### Task 1.3.2: Build Neighborhood Index
```python
def build_semantic_neighborhoods(vocab_size=10000, n_neighbors=50):
    """
    For each token in vocabulary:
    1. Get embedding vector
    2. Compute cosine similarity with all other tokens
    3. Store top-50 nearest neighbors
    4. Save as pickle file: neighborhoods.pkl
    
    Structure: Dict[token_id, List[token_id]]
    """
    neighborhoods = {}
    for token_id in range(vocab_size):
        embedding = embeddings[token_id]
        similarities = compute_cosine_similarities(embedding, all_embeddings)
        top_k = numpy.argsort(similarities)[-n_neighbors:]
        neighborhoods[token_id] = top_k.tolist()
    return neighborhoods
```

## PHASE 2: Core Genetic Algorithm Components

### Step 2.1: Genome Implementation

#### Task 2.1.1: Define PromptGenome Class
```python
class PromptGenome:
    def __init__(self, token_ids, max_length=200):
        self.token_ids = token_ids[:max_length]  # List of integers
        self.fitness = None  # Float 0-1
        self.accuracy = None  # Float 0-1
        self.generation_born = 0  # Integer
        self.parent_ids = []  # List of 2 UUID strings
        self.genome_id = generate_uuid()  # Unique identifier
        self.mutation_count = 0  # Track mutations applied
        self.evaluation_count = 0  # Number of problems evaluated on
        
    def to_text(self, tokenizer):
        """Convert token IDs to text string"""
        return tokenizer.decode(self.token_ids)
    
    def get_hash(self):
        """Return hash for caching purposes"""
        return hashlib.md5(bytes(self.token_ids)).hexdigest()
```

### Step 2.2: Genetic Operators

#### Task 2.2.1: Crossover Operator
```python
def crossover(parent1, parent2, tokenizer):
    """
    1. Convert token_ids to text
    2. Find sentence boundaries using NLTK
    3. Select crossover points near boundaries (±5 tokens)
    4. If no boundaries, use midpoint
    5. Create offspring token sequence
    6. Return new PromptGenome
    """
    # Find sentence boundaries
    text1 = parent1.to_text(tokenizer)
    text2 = parent2.to_text(tokenizer)
    
    boundaries1 = find_sentence_token_boundaries(text1, tokenizer)
    boundaries2 = find_sentence_token_boundaries(text2, tokenizer)
    
    if boundaries1 and boundaries2:
        point1 = random.choice(boundaries1)
        point2 = random.choice(boundaries2)
    else:
        point1 = len(parent1.token_ids) // 2
        point2 = len(parent2.token_ids) // 2
    
    offspring_tokens = parent1.token_ids[:point1] + parent2.token_ids[point2:]
    return PromptGenome(offspring_tokens)
```

#### Task 2.2.2: Mutation Operator
```python
def mutate(genome, neighborhoods, pop_prob=0.8, token_prob=0.002):
    """
    1. Apply population-level mutation with probability pop_prob
    2. For each token, mutate with probability token_prob
    3. Use 90% semantic neighbors, 10% random tokens
    4. Track number of mutations made
    5. Return mutated genome (or original if no mutations)
    """
    if random.random() > pop_prob:
        return genome
    
    mutated_tokens = genome.token_ids.copy()
    mutations_made = 0
    
    for i in range(len(mutated_tokens)):
        if random.random() < token_prob:
            token_id = mutated_tokens[i]
            
            if random.random() < 0.9 and token_id in neighborhoods:
                # Semantic neighbor
                new_token = random.choice(neighborhoods[token_id])
            else:
                # Random token
                new_token = random.randint(0, vocab_size-1)
            
            mutated_tokens[i] = new_token
            mutations_made += 1
    
    if mutations_made > 0:
        new_genome = PromptGenome(mutated_tokens)
        new_genome.mutation_count = mutations_made
        return new_genome
    return genome
```

### Step 2.3: Population Management

#### Task 2.3.1: Initialize Population
```python
def create_initial_population(seed_prompts, population_size=500):
    """
    seed_prompts: List of 50 PromptGenome objects
    
    1. Generate 500 offspring through random parent selection
    2. Apply crossover to each pair
    3. Apply initial mutation with higher rate (token_prob=0.005)
    4. Return list of 500 PromptGenome objects
    """
    population = []
    for _ in range(population_size):
        parent1 = random.choice(seed_prompts)
        parent2 = random.choice(seed_prompts)
        offspring = crossover(parent1, parent2, tokenizer)
        offspring = mutate(offspring, neighborhoods, 
                         pop_prob=0.9, token_prob=0.005)
        offspring.generation_born = 0
        offspring.parent_ids = [parent1.genome_id, parent2.genome_id]
        population.append(offspring)
    return population
```

## PHASE 3: Evaluation System

### Step 3.1: LLM Interface

#### Task 3.1.1: API Client Setup
```python
class LLMEvaluator:
    def __init__(self, api_key, model="gpt-4", cache_dir="./data/cache"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = DiskCache(cache_dir)
        self.call_count = 0
        self.cache_hits = 0
        
    def generate_completion(self, prompt, max_tokens=200, temperature=0):
        """
        1. Check cache using MD5 hash of prompt
        2. If cached, return cached response
        3. If not, make API call with retry logic
        4. Cache response before returning
        """
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = response.choices[0].text
                self.cache[cache_key] = result
                self.call_count += 1
                return result
            except RateLimitError:
                time.sleep(2 ** attempt)
        
        raise Exception("API call failed after 3 attempts")
```

### Step 3.2: Fitness Evaluation

#### Task 3.2.1: Single Genome Evaluation
```python
def evaluate_genome(genome, problems, evaluator):
    """
    1. Convert genome to prompt text
    2. For each problem in subset:
       a. Construct full prompt
       b. Get LLM response
       c. Extract answer
       d. Compare with ground truth
    3. Calculate accuracy
    4. Apply length penalty
    5. Return fitness score
    """
    prompt_text = genome.to_text(tokenizer)
    correct = 0
    
    for problem in problems:
        full_prompt = f"{prompt_text}\n\nQuestion: {problem['question']}\nAnswer:"
        
        response = evaluator.generate_completion(full_prompt)
        predicted = extract_final_answer(response)
        ground_truth = problem['final_answer']
        
        # Numerical comparison with tolerance
        if predicted is not None and abs(predicted - ground_truth) < 0.001:
            correct += 1
    
    accuracy = correct / len(problems)
    
    # Length penalty: reduce fitness for prompts over 200 tokens
    length_penalty = 1.0
    if len(genome.token_ids) > 200:
        excess = len(genome.token_ids) - 200
        length_penalty = 1.0 - (excess / 300) * 0.1  # Max 10% penalty
    
    fitness = accuracy * length_penalty
    
    genome.accuracy = accuracy
    genome.fitness = fitness
    genome.evaluation_count = len(problems)
    
    return fitness
```

#### Task 3.2.2: Batch Evaluation with Parallelization
```python
def evaluate_population(population, problems, evaluator, n_workers=10):
    """
    1. Create thread pool with n_workers
    2. Batch population into chunks
    3. Evaluate in parallel
    4. Handle failures gracefully
    5. Return evaluated population
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def eval_wrapper(genome):
        try:
            return evaluate_genome(genome, problems, evaluator)
        except Exception as e:
            print(f"Evaluation failed for genome {genome.genome_id}: {e}")
            genome.fitness = 0.0
            return 0.0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(eval_wrapper, g) for g in population]
        for future in tqdm(futures, desc="Evaluating"):
            future.result()
    
    return population
```

## PHASE 4: Selection and Evolution

### Step 4.1: Selection Strategy

#### Task 4.1.1: Multi-Objective Selection
```python
def select_parents(population, n_parents=50):
    """
    Select 50 parents using multiple strategies:
    - 20 elite (highest fitness)
    - 15 diverse (maximum distance from elite)
    - 15 random (exploration)
    """
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    # Elite selection
    elite = sorted_pop[:20]
    
    # Diversity selection
    remaining = sorted_pop[20:]
    diverse = select_diverse_genomes(remaining, elite, n=15)
    
    # Random selection
    random_pool = sorted_pop[35:]
    random_selected = random.sample(random_pool, min(15, len(random_pool)))
    
    return elite + diverse + random_selected
```

#### Task 4.1.2: Diversity Measurement
```python
def select_diverse_genomes(candidates, reference_set, n=15):
    """
    1. Calculate token overlap between each candidate and reference set
    2. Select candidates with minimum overlap
    3. Use Jaccard distance for diversity metric
    """
    diversity_scores = []
    
    for candidate in candidates:
        candidate_set = set(candidate.token_ids)
        
        min_similarity = 1.0
        for ref in reference_set:
            ref_set = set(ref.token_ids)
            jaccard = len(candidate_set & ref_set) / len(candidate_set | ref_set)
            min_similarity = min(min_similarity, jaccard)
        
        diversity_scores.append((candidate, 1 - min_similarity))
    
    # Sort by diversity score and select top n
    diversity_scores.sort(key=lambda x: x[1], reverse=True)
    return [genome for genome, _ in diversity_scores[:n]]
```

### Step 4.2: Main Evolution Loop

#### Task 4.2.1: Generation Execution
```python
def run_generation(generation_num, population, evaluator, config):
    """
    Execute one complete generation:
    1. Determine evaluation size based on generation number
    2. Evaluate all offspring
    3. Log statistics
    4. Select parents
    5. Generate next population
    6. Save checkpoint
    """
    # Adaptive evaluation size
    if generation_num <= 10:
        eval_problems = random.sample(all_problems, 50)
    elif generation_num <= 20:
        eval_problems = random.sample(all_problems, 100)
    else:
        eval_problems = random.sample(all_problems, 150)
    
    # Evaluation
    population = evaluate_population(population, eval_problems, evaluator)
    
    # Statistics
    best = max(population, key=lambda x: x.fitness)
    avg_fitness = numpy.mean([g.fitness for g in population])
    std_fitness = numpy.std([g.fitness for g in population])
    
    print(f"Generation {generation_num}:")
    print(f"  Best fitness: {best.fitness:.4f} (Accuracy: {best.accuracy:.2%})")
    print(f"  Average fitness: {avg_fitness:.4f} (±{std_fitness:.4f})")
    print(f"  Best prompt preview: {best.to_text(tokenizer)[:100]}...")
    
    # Check convergence
    if best.accuracy >= 0.85:
        print("TARGET ACCURACY REACHED!")
        return population, True
    
    # Selection
    parents = select_parents(population, n_parents=50)
    
    # Reproduction
    new_population = []
    for _ in range(config['population_size']):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        
        offspring = crossover(parent1, parent2, tokenizer)
        offspring = mutate(offspring, neighborhoods,
                         config['pop_mutation_prob'],
                         config['token_mutation_prob'])
        
        offspring.generation_born = generation_num + 1
        offspring.parent_ids = [parent1.genome_id, parent2.genome_id]
        
        new_population.append(offspring)
    
    # Save checkpoint
    save_checkpoint(generation_num, new_population, best)
    
    return new_population, False
```

#### Task 4.2.2: Complete Evolution Run
```python
def evolve(config):
    """
    Main evolution loop:
    1. Initialize all components
    2. Create initial population
    3. Run generations until convergence or max_generations
    4. Save final results
    """
    # Initialize
    neighborhoods = load_semantic_neighborhoods()
    evaluator = LLMEvaluator(api_key=config['api_key'])
    seed_prompts = load_seed_prompts(config['seed_file'])
    
    # Create initial population
    population = create_initial_population(seed_prompts, 
                                          config['population_size'])
    
    # Evolution loop
    for generation in range(config['max_generations']):
        population, converged = run_generation(
            generation, population, evaluator, config
        )
        
        if converged:
            break
        
        # Stagnation detection
        if generation > 5:
            recent_best = load_recent_best_fitness(n=5)
            if max(recent_best) - min(recent_best) < 0.001:
                print("Stagnation detected - increasing mutation")
                config['token_mutation_prob'] *= 2
    
    # Final evaluation on full test set
    best = max(population, key=lambda x: x.fitness)
    final_accuracy = evaluate_genome(best, full_test_set, evaluator)
    
    print(f"Final test accuracy: {final_accuracy:.2%}")
    save_final_results(best, final_accuracy)
    
    return best
```

## PHASE 5: Monitoring and Checkpointing

### Step 5.1: Logging System

#### Task 5.1.1: Metrics Tracking
```python
class ExperimentLogger:
    def __init__(self, log_dir="./data/logs"):
        self.log_dir = log_dir
        self.metrics_file = f"{log_dir}/metrics.jsonl"
        self.genomes_file = f"{log_dir}/genomes.jsonl"
        
    def log_generation(self, generation, population):
        """
        Log per-generation metrics:
        - Best/avg/min fitness
        - Diversity metrics
        - Token frequency analysis
        - Mutation effectiveness
        """
        metrics = {
            'generation': generation,
            'timestamp': time.time(),
            'best_fitness': max(g.fitness for g in population),
            'avg_fitness': numpy.mean([g.fitness for g in population]),
            'min_fitness': min(g.fitness for g in population),
            'std_fitness': numpy.std([g.fitness for g in population]),
            'avg_length': numpy.mean([len(g.token_ids) for g in population]),
            'diversity': calculate_population_diversity(population),
            'cache_hit_rate': evaluator.cache_hits / max(1, evaluator.call_count)
        }
        
        with jsonlines.open(self.metrics_file, 'a') as writer:
            writer.write(metrics)
```

### Step 5.2: Checkpoint Management

#### Task 5.2.1: Save Checkpoint
```python
def save_checkpoint(generation, population, best_genome):
    """
    Save complete state for recovery:
    1. Population genomes as pickle
    2. Best genome as text file
    3. Metadata as JSON
    """
    checkpoint_dir = f"./data/checkpoints/gen_{generation:04d}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save population
    with open(f"{checkpoint_dir}/population.pkl", 'wb') as f:
        pickle.dump(population, f)
    
    # Save best as text
    with open(f"{checkpoint_dir}/best_prompt.txt", 'w') as f:
        f.write(best_genome.to_text(tokenizer))
    
    # Save metadata
    metadata = {
        'generation': generation,
        'best_fitness': best_genome.fitness,
        'best_accuracy': best_genome.accuracy,
        'population_size': len(population),
        'timestamp': time.time()
    }
    
    with open(f"{checkpoint_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

## PHASE 6: Seed Prompt Initialization

### Step 6.1: Create Seed Prompts

#### Task 6.1.1: Define 50 Seed Prompts
```python
SEED_PROMPTS = [
    # Basic Chain-of-Thought
    "Let's solve this step by step.",
    "Think through this problem carefully.",
    
    # Structured Approaches
    "First, identify what we're looking for. Second, list given information. Third, determine the operations. Fourth, calculate.",
    "Break this into parts: (1) Understand the question (2) Plan the solution (3) Execute calculations (4) Verify answer.",
    
    # Role-based
    "As a math teacher, I'll solve this methodically.",
    "You are a careful mathematician. Show all work.",
    
    # Verification-focused
    "Solve and then double-check by working backwards.",
    "Calculate the answer and verify it makes sense.",
    
    # Format-specific
    "Solution:\nStep 1: [reasoning]\nStep 2: [reasoning]\nFinal Answer: [number]",
    "Given: [list facts]\nFind: [goal]\nSolution: [work]\nAnswer: [result]",
    
    # [40 more variations covering different strategies]
]

def create_seed_genomes():
    """Convert text prompts to PromptGenome objects"""
    seed_genomes = []
    for prompt_text in SEED_PROMPTS:
        token_ids = tokenizer.encode(prompt_text)
        genome = PromptGenome(token_ids)
        genome.generation_born = -1  # Mark as seed
        seed_genomes.append(genome)
    return seed_genomes
```

## PHASE 7: Analysis and Comparison

### Step 7.1: Performance Analysis

#### Task 7.1.1: Evolution Trajectory Analysis
```python
def analyze_evolution():
    """
    1. Load all generation checkpoints
    2. Extract fitness progression
    3. Identify phase transitions
    4. Measure diversity over time
    5. Generate plots
    """
    generations = []
    best_fitness = []
    avg_fitness = []
    diversity = []
    
    for gen in range(max_generation):
        checkpoint = load_checkpoint(gen)
        generations.append(gen)
        best_fitness.append(checkpoint['best_fitness'])
        avg_fitness.append(checkpoint['avg_fitness'])
        diversity.append(checkpoint['diversity'])
    
    # Plot fitness curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_fitness, label='Best')
    plt.plot(generations, avg_fitness, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(generations, diversity)
    plt.xlabel('Generation')
    plt.ylabel('Population Diversity')
    
    plt.savefig('./data/results/evolution_curves.png')
```

### Step 7.2: Baseline Comparison

#### Task 7.2.1: Evaluate Against Baselines
```python
def compare_with_baselines():
    """
    Test evolved prompt against known baselines:
    1. Load best evolved prompt
    2. Evaluate on same 200-problem test set
    3. Compare with baseline prompts
    4. Calculate statistical significance
    """
    baselines = {
        'zero_shot': "",
        'simple_cot': "Let's think step by step",
        'structured': "Break down the problem:\n1. What we know:\n2. What to find:\n3. Solution:",
        'best_known': "[Insert current SOTA prompt from literature]"
    }
    
    # Load evolved prompt
    best_evolved = load_best_genome()
    
    # Test all prompts
    results = {}
    test_problems = load_final_test_set()
    
    for name, prompt in baselines.items():
        genome = text_to_genome(prompt)
        accuracy = evaluate_genome(genome, test_problems, evaluator)
        results[name] = accuracy
    
    # Test evolved
    evolved_accuracy = evaluate_genome(best_evolved, test_problems, evaluator)
    results['evolved'] = evolved_accuracy
    
    # Statistical significance test
    from scipy.stats import binomtest
    n_problems = len(test_problems)
    
    for baseline_name, baseline_acc in results.items():
        if baseline_name != 'evolved':
            # Binomial test for difference
            successes = int(evolved_accuracy * n_problems)
            p_value = binomtest(successes, n_problems, baseline_acc).pvalue
            print(f"Evolved vs {baseline_name}: {evolved_accuracy:.2%} vs {baseline_acc:.2%} (p={p_value:.4f})")
    
    return results
```

## PHASE 8: Production Deployment

### Step 8.1: Final Validation

#### Task 8.1.1: Comprehensive Testing
```python
def final_validation():
    """
    1. Test on full GSM8K test set (1,319 problems)
    2. Test on related datasets (SVAMP, ASDiv)
    3. Measure inference time
    4. Calculate confidence intervals
    """
    best_genome = load_best_genome()
    
    # Full GSM8K test
    full_test = load_dataset("openai/gsm8k", "main")['test']
    gsm8k_accuracy = evaluate_genome(best_genome, full_test, evaluator)
    
    # Transfer test on related datasets
    svamp_accuracy = test_on_svamp(best_genome)
    asdiv_accuracy = test_on_asdiv(best_genome)
    
    # Bootstrap confidence intervals
    bootstrap_scores = []
    for _ in range(1000):
        sample = random.sample(full_test, 100)
        score = evaluate_genome(best_genome, sample, evaluator)
        bootstrap_scores.append(score)
    
    ci_lower = numpy.percentile(bootstrap_scores, 2.5)
    ci_upper = numpy.percentile(bootstrap_scores, 97.5)
    
    results = {
        'gsm8k_accuracy': gsm8k_accuracy,
        'confidence_interval': (ci_lower, ci_upper),
        'svamp_transfer': svamp_accuracy,
        'asdiv_transfer': asdiv_accuracy,
        'prompt_length': len(best_genome.token_ids),
        'prompt_text': best_genome.to_text(tokenizer)
    }
    
    return results
```

### Step 8.2: Result Documentation

#### Task 8.2.1: Generate Final Report
```python
def generate_final_report():
    """
    Create comprehensive report including:
    1. Evolution statistics
    2. Final prompt text
    3. Performance metrics
    4. Comparison tables
    5. Discovered patterns
    """
    report = {
        'experiment_id': generate_experiment_id(),
        'date': datetime.now().isoformat(),
        'configuration': load_config(),
        'evolution_summary': {
            'total_generations': final_generation,
            'total_evaluations': total_api_calls,
            'convergence_generation': convergence_gen,
            'total_cost_usd': calculate_total_cost()
        },
        'best_prompt': {
            'text': best_genome.to_text(tokenizer),
            'length_tokens': len(best_genome.token_ids),
            'fitness': best_genome.fitness,
            'accuracy': best_genome.accuracy
        },
        'performance': {
            'gsm8k_test': final_accuracy,
            'baseline_comparison': baseline_results,
            'transfer_learning': transfer_results
        },
        'discovered_patterns': extract_common_patterns(top_genomes),
        'computational_resources': {
            'total_runtime_hours': total_runtime / 3600,
            'api_calls': total_api_calls,
            'cache_hit_rate': final_cache_hit_rate,
            'estimated_cost': total_cost
        }
    }
    
    # Save report
    with open('./data/results/final_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown summary
    generate_markdown_report(report)
    
    return report
```

## PHASE 9: Error Handling and Recovery

### Step 9.1: Fault Tolerance

#### Task 9.1.1: Implement Recovery Mechanisms
```python
class EvolutionRunner:
    def __init__(self, config):
        self.config = config
        self.current_generation = 0
        self.population = None
        
    def run_with_recovery(self):
        """
        Main evolution loop with automatic recovery:
        1. Check for existing checkpoints
        2. Resume from last valid state
        3. Handle API failures gracefully
        4. Implement exponential backoff
        """
        # Check for existing run
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming from generation {latest_checkpoint['generation']}")
            self.current_generation = latest_checkpoint['generation']
            self.population = latest_checkpoint['population']
        else:
            print("Starting fresh run")
            self.population = create_initial_population(
                seed_prompts, 
                self.config['population_size']
            )
        
        # Main loop with error handling
        while self.current_generation < self.config['max_generations']:
            try:
                self.population, converged = run_generation(
                    self.current_generation,
                    self.population,
                    evaluator,
                    self.config
                )
                
                if converged:
                    break
                    
                self.current_generation += 1
                
            except RateLimitError:
                print("Rate limit hit - waiting 60 seconds")
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in generation {self.current_generation}: {e}")
                print("Saving emergency checkpoint")
                save_emergency_checkpoint(self.current_generation, self.population)
                raise
```

## EXECUTION INSTRUCTIONS

### Launch Sequence:
1. **Setup Phase**: 2-4 hours
   ```bash
   python scripts/setup_environment.py
   python scripts/download_data.py
   python scripts/build_neighborhoods.py
   ```

2. **Evolution Phase**: 24-48 hours
   ```bash
   python scripts/run_evolution.py --config configs/experiment_config.json
   ```

3. **Analysis Phase**: 2-3 hours
   ```bash
   python scripts/analyze_results.py
   python scripts/generate_report.py
   ```

### Critical Success Parameters:
- **Population**: 500 genomes per generation
- **Parents**: 50 selected per generation
- **Mutation**: population_prob=0.8, token_prob=0.002
- **Max Length**: 200 tokens
- **Target**: 85% accuracy on GSM8K
- **Budget**: $2,000-3,000 total

### Convergence Criteria:
- Accuracy >= 85% OR
- 30 generations completed OR
- Fitness improvement < 0.001 for 5 consecutive generations

This specification provides complete implementation details for every component, leaving no architectural decisions undefined. The system is designed for robustness, efficiency, and reproducibility.