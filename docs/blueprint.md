GSM8K Genetic Algorithm Prompt Optimization Blueprint
Project Overview
Objective: Evolve optimal prompts for GSM8K math problems using genetic algorithms
Target Performance: Beat current best ~95% accuracy using Anthropic's Claude 3

Phase 1: Environment Setup
1.1 Dataset Preparation
# Download GSM8K dataset
gsm8k_train = load_dataset("gsm8k", "main", split="train")  # 7,473 problems
gsm8k_test = load_dataset("gsm8k", "main", split="test")    # 1,319 problems

# Create evaluation sets
eval_set_small = random.sample(gsm8k_test, 50)   # Quick eval during evolution
eval_set_full = random.sample(gsm8k_test, 200)   # Final evaluation
validation_set = random.sample(gsm8k_train, 100)  # Prevent overfitting

1.2 Infrastructure Requirements
LLM API: GPT-4o or Claude for evaluation

Compute: Standard CPU (no GPU needed)

Storage: ~1GB for logs and checkpoints

Token Embedding Model: Download pre-trained embeddings (GloVe or Word2Vec)

1.3 Dependencies
pip install openai anthropic numpy pandas matplotlib
pip install gensim  # For semantic neighborhoods
pip install nltk    # For tokenization
pip install jsonlines tqdm

Phase 2: Core Components Implementation
2.1 Prompt Genome Structure
class PromptGenome:
    def __init__(self, tokens, max_length=200):
        self.tokens = tokens[:max_length]
        self.fitness = None
        self.accuracy = None
        self.generation_born = 0
        self.parents = []
        self.mutation_history = []

    def to_text(self):
        return " ".join(self.tokens)

    def length(self):
        return len(self.tokens)

2.2 Semantic Neighborhood Builder
class SemanticNeighborhoods:
    def __init__(self, model_path='glove-wiki-gigaword-100'):
        self.embeddings = load_embeddings(model_path)
        self.neighborhoods = {}
        self.build_neighborhoods()

    def build_neighborhoods(self, k=50):
        for token, embedding in self.embeddings.items():
            # Find k nearest neighbors
            similarities = cosine_similarity(embedding, all_embeddings)
            neighbors = top_k_indices(similarities, k)
            self.neighborhoods[token] = neighbors

    def get_neighbor(self, token, random_chance=0.1):
        if random.random() < random_chance:
            return random.choice(list(self.embeddings.keys()))
        if token in self.neighborhoods:
            return random.choice(self.neighborhoods[token])
        return token  # Fallback to original

2.3 Genetic Operators
Crossover Function
def crossover(parent1, parent2):
    # Find sentence boundaries for semantic preservation
    boundaries1 = find_sentence_boundaries(parent1.tokens)
    boundaries2 = find_sentence_boundaries(parent2.tokens)

    # Select crossover points near boundaries
    if boundaries1 and boundaries2:
        point1 = random.choice(boundaries1)
        point2 = random.choice(boundaries2)
    else:
        # Fallback to midpoint
        point1 = len(parent1.tokens) // 2
        point2 = len(parent2.tokens) // 2

    # Create offspring
    offspring_tokens = parent1.tokens[:point1] + parent2.tokens[point2:]
    return PromptGenome(offspring_tokens)

Mutation Function
def mutate(genome, neighborhoods, pop_mutation_prob=0.8, token_mutation_prob=0.002):
    if random.random() > pop_mutation_prob:
        return genome  # No mutation

    mutated_tokens = genome.tokens.copy()
    mutations_made = 0

    for i in range(len(mutated_tokens)):
        if random.random() < token_mutation_prob:
            old_token = mutated_tokens[i]
            new_token = neighborhoods.get_neighbor(old_token)
            mutated_tokens[i] = new_token
            mutations_made += 1

    new_genome = PromptGenome(mutated_tokens)
    new_genome.mutation_history.append(f"Mutated {mutations_made} tokens")
    return new_genome

2.4 Evaluation Pipeline
import asyncio
from openai import AsyncOpenAI

class GSM8KEvaluator:
    def __init__(self, api_key, cache_enabled=True):
        self.api_client = AsyncOpenAI(api_key=api_key)
        self.cache = {} if cache_enabled else None

    async def get_solution(self, full_prompt, semaphore):
        # Asynchronously get one solution, respecting the semaphore
        cache_key = hash(full_prompt)
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]

        async with semaphore:
            response = await self.api_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=200,
                temperature=0
            )
            result = response.choices[0].message.content
            if self.cache is not None:
                self.cache[cache_key] = result
            return result

    def calculate_fitness(self, accuracy, length):
        # Slight penalty for excessive length
        length_penalty = max(0, 1 - (length / 500) * 0.1)
        return accuracy * length_penalty

Phase 3: Genetic Algorithm Core
3.1 Population Initialization
def create_initial_population(seed_prompts, population_size=500):
    population = []

    # Create diverse crossovers from seeds
    for _ in range(population_size):
        parent1 = random.choice(seed_prompts)
        parent2 = random.choice(seed_prompts)
        offspring = crossover(parent1, parent2)
        population.append(offspring)

    return population

3.2 Selection Strategy
def select_parents(population, num_parents=50):
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    # Multi-strategy selection
    elite = sorted_pop[:20]  # Top 40%

    # Diversity selection - most different from elite
    diverse = select_diverse(sorted_pop[20:], elite, count=15)

    # Random lottery for exploration
    lottery = random.sample(sorted_pop[35:], min(15, len(sorted_pop[35:])))

    return elite + diverse + lottery

3.3 Main Evolution Loop
async def evaluate_population_concurrently(population, eval_problems, evaluator, concurrency_limit=200):
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []
    
    # Create all tasks: each genome is evaluated on each problem
    for genome in population:
        for problem in eval_problems:
            full_prompt = f"{genome.to_text()}\n\nProblem: {problem['question']}"
            tasks.append(evaluator.get_solution(full_prompt, semaphore))

    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks)

    # Process results and assign fitness
    result_idx = 0
    for genome in population:
        correct = 0
        for problem in eval_problems:
            predicted = extract_number(results[result_idx])
            ground_truth = extract_number(problem['answer'])
            if predicted == ground_truth:
                correct += 1
            result_idx += 1
        
        accuracy = correct / len(eval_problems)
        genome.accuracy = accuracy
        genome.fitness = evaluator.calculate_fitness(accuracy, genome.length())

def evolve(config):
    # Initialize
    neighborhoods = SemanticNeighborhoods()
    evaluator = GSM8KEvaluator(api_key=config.api_key)
    population = create_initial_population(config.seed_prompts)

    # Evolution loop
    for generation in range(config.max_generations):
        print(f"\n=== Generation {generation} ===")

        # Evaluate the entire population in one concurrent batch
        asyncio.run(evaluate_population_concurrently(population, config.eval_problems, evaluator))

        # Log statistics
        best = max(population, key=lambda x: x.fitness)
        avg_fitness = np.mean([g.fitness for g in population])
        print(f"Best fitness: {best.fitness:.4f} (Accuracy: {best.accuracy:.2%})")
        print(f"Average fitness: {avg_fitness:.4f}")

        # Check convergence
        if best.accuracy >= config.target_accuracy:
            print(f"Target accuracy reached!")
            break

        # Selection
        parents = select_parents(population, config.num_parents)

        # Create next generation
        new_population = [crossover(random.choice(parents), random.choice(parents)) for _ in range(config.population_size)]
        population = [mutate(g, neighborhoods) for g in new_population]
        
        save_checkpoint(generation, population, best)

    return best

Phase 4: Seed Prompts
4.1 Initial Prompt Seeds (50 diverse starting points)
seed_prompts = [
    # Basic CoT
    "Let's solve this step by step.",

    # Structured approach
    "First, identify what we need to find. Then list the given information. Next, determine the operations needed. Finally, calculate the answer.",

    # Role-based
    "You are a patient math teacher. Break down this problem into simple steps that a student could follow.",

    # Verification focused
    "Solve this problem and then verify your answer by working backwards.",

    # Format specific
    "Solve this problem. Show your work using numbered steps. Each step should have the calculation and the result.",

    # ... (45 more diverse prompts)
]

Phase 5: Experiment Execution
5.1 Configuration
config = {
    'population_size': 500,
    'num_parents': 50,
    'max_generations': 30,
    'pop_mutation_prob': 0.8,
    'token_mutation_prob': 0.002,
    'max_prompt_length': 200,
    'eval_problems_per_gen': 50,  # Increase in later generations
    'target_accuracy': 0.85,
    'random_seed': 42
}

5.2 Progressive Evaluation Strategy
Generation 1-10: 50 problems per prompt (quick iteration)

Generation 11-20: 100 problems per prompt (better signal)

Generation 21-30: 150 problems per prompt (fine-tuning)

5.3 Monitoring & Logging
class ExperimentLogger:
    def log_generation(self, gen, population):
        # Track metrics
        metrics = {
            'generation': gen,
            'best_accuracy': max(g.accuracy for g in population),
            'avg_accuracy': np.mean([g.accuracy for g in population]),
            'diversity': calculate_diversity(population),
            'avg_length': np.mean([g.length() for g in population])
        }

        # Save best prompts
        top_5 = sorted(population, key=lambda x: x.fitness)[:5]
        for i, genome in enumerate(top_5):
            save_prompt(f"gen_{gen}_rank_{i}.txt", genome)

Phase 6: Comparison & Analysis
6.1 Baseline Comparison
baselines = {
    'zero_shot': "Answer:",
    'simple_cot': "Let's think step by step",
    'structured': "Break down the problem:\n1. What we know:\n2. What we need:\n3. Solution:",
    'best_known': "[Insert current SOTA prompt]"
}

# Evaluate all baselines on same test set
for name, prompt in baselines.items():
    accuracy = evaluate_baseline(prompt, test_set)
    print(f"{name}: {accuracy:.2%}")

6.2 Analysis Scripts
def analyze_evolution():
    # Load all checkpoints
    history = load_experiment_history()

    # Plot fitness over time
    plot_fitness_curve(history)

    # Analyze prompt evolution
    track_phrase_frequency(history)

    # Diversity metrics
    plot_diversity_over_time(history)

    # Length distribution
    analyze_length_distribution(history)

6.3 Final Validation
def final_validation(best_genome):
    # Test on held-out set
    test_accuracy = evaluate_on_full_test(best_genome)

    # Check for overfitting
    train_accuracy = evaluate_on_train_sample(best_genome)

    # Statistical significance
    significance = bootstrap_confidence_interval(best_genome, baselines)

    return {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'overfit_ratio': train_accuracy / test_accuracy,
        'significance': significance
    }

Phase 7: Cost Optimization
7.1 Caching Strategy
Cache all API responses by hash(prompt + problem)

Estimated 60% cache hit rate after generation 10

Reduces cost from $100 to $40 per generation

7.2 Concurrent Evaluation
Use asyncio to run all API requests for an entire population concurrently.

A semaphore controls the maximum number of parallel requests to respect rate limits.

This is far more efficient than simple batching for I/O-bound tasks like API calls.

7.3 Early Stopping
def check_early_stopping(history, patience=5):
    if len(history) < patience:
        return False

    recent_best = [gen['best_accuracy'] for gen in history[-patience:]]
    improvement = max(recent_best) - min(recent_best)

    return improvement < 0.001  # Less than 0.1% improvement

Phase 8: Running the Experiment
8.1 Launch Script
# Run with monitoring
python run_evolution.py \
    --config config.json \
    --seeds seed_prompts.txt \
    --output results/ \
    --checkpoint_every 1 \
    --verbose

8.2 Expected Timeline
Setup: 2-4 hours (downloading models, preparing data)

Evolution: 24-48 hours (30 generations)

Analysis: 2-3 hours (generating reports)

8.3 Expected Costs
Optimistic: $1,500 (high cache hits, quick convergence)

Realistic: $2,000-2,500 (normal run)

Pessimistic: $3,000 (low cache hits, 30 full generations)

Phase 9: Success Metrics
9.1 Primary Metrics
Accuracy: Target 85%+ on GSM8K test set

Consistency: <2% variance across different problem samples

Efficiency: <200 tokens average prompt length

9.2 Secondary Metrics
Convergence Speed: Generations to reach 80% accuracy

Diversity Maintenance: Population diversity >0.3 throughout

Novel Discoveries: Identify prompt patterns not in literature

9.3 Publication Criteria
Beat baseline by >3% with statistical significance (p<0.05)

Discover interpretable prompt patterns

Demonstrate generalization to related math datasets

Appendix: Troubleshooting
Common Issues and Solutions
Issue: Premature convergence (all prompts become similar)

Solution: Increase mutation rate temporarily, add diversity bonus to fitness

Issue: No improvement after 10 generations

Solution: Increase evaluation set size, check for cached bad results

Issue: API rate limits

Solution: Implement exponential backoff, adjust semaphore limit in the concurrent evaluator

Issue: Prompts becoming too long

Solution: Increase length penalty in fitness function

Issue: Nonsensical mutations

Solution: Verify semantic neighborhoods are working, reduce random mutation probability