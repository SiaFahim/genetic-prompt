# Complete Implementation Instructions for GSM8K Genetic Algorithm Project

## PROJECT OVERVIEW
You will implement a genetic algorithm system that evolves optimal prompts for solving GSM8K math problems. The system will start with 50 seed prompts, evolve them through crossover and mutation operations across 500-member populations, and converge to prompts achieving 85% accuracy or better. Reference the `technical_implementation_specification.md` for exact specifications and `blueprint.md` for architectural patterns.

## PHASE 1: ENVIRONMENT AND INFRASTRUCTURE SETUP

### Task 1.1: Create Project Structure
Inside the current directory, establish the complete folder hierarchy as specified in the technical specification document section 1.1.2. Every subdirectory must be created exactly as shown, including `src/genetics`, `src/evaluation`, `src/embeddings`, `src/utils`, `data/seeds`, `data/checkpoints`, `data/results`, `configs`, and `scripts`. This structure is critical for the modular organization of the codebase.

### Task 1.2: Set Up Python Environment
Create a Python 3.10 virtual environment specifically for this project. Name it `gsm8k_ga_env`. Activate this environment and install all dependencies with the exact versions listed in the technical specification section 1.1.1. The version pinning is crucial to ensure reproducible results. After installation, create a `requirements.txt` file that captures these exact versions for future reference.

### Task 1.3: Configure API Access
Set up authentication for the OpenAI API or Anthropic Claude API, depending on which LLM you'll use for evaluation. Store the API key securely using environment variables. Create a `.env` file in the project root with your API credentials. Implement a configuration loader that reads these credentials securely without hardcoding them in the source code.

### Task 1.4: Download GSM8K Dataset
Use the HuggingFace datasets library to download the complete GSM8K dataset. Save it locally to `./data/gsm8k_raw` directory. The dataset contains 7,473 training problems and 1,319 test problems. Each problem has a question field and an answer field that contains step-by-step solutions with calculator annotations marked by double angle brackets and final answers marked after four hash symbols.

### Task 1.5: Prepare Evaluation Subsets
From the downloaded dataset, create three distinct evaluation subsets using deterministic random sampling. First, create a primary evaluation set by sampling 100 problems from the test set using random seed 42. Second, create a validation set by sampling 100 problems from the training set using seed 43. Third, create a final test set by sampling 200 non-overlapping problems from the test set using seed 44. Save each subset as a JSON file in the data directory with clear naming conventions.

### Task 1.6: Implement Answer Extraction
Create a robust answer extraction function that parses the GSM8K answer format. This function must handle the specific format where final answers appear after four hash symbols. It needs to correctly extract integers, decimals, negative numbers, and numbers with comma separators. The function should return a float value or None if no answer is found. This is critical for accurately scoring model outputs.

## PHASE 2: SEMANTIC EMBEDDING SYSTEM

### Task 2.1: Acquire Word Embeddings
Download the GloVe embeddings file, specifically the 100-dimensional version trained on 6 billion tokens. This file is approximately 822MB. Store it in a dedicated embeddings directory within the data folder. These embeddings will be the foundation for semantic-aware mutations.

### Task 2.2: Build Token Vocabulary
Extract the vocabulary from your tokenizer, limiting it to the 10,000 most common tokens for computational efficiency. Create a bidirectional mapping between tokens and their integer IDs. This mapping must be consistent throughout the entire experiment to ensure reproducibility.

### Task 2.3: Construct Semantic Neighborhoods
For each token in your vocabulary, compute its 50 nearest semantic neighbors using cosine similarity on the GloVe embeddings. This is a one-time computation that will take approximately one hour. Store the results as a pickle file that maps each token ID to a list of its 50 nearest neighbor token IDs. This neighborhood structure enables semantic mutations that preserve meaning while exploring variations.

### Task 2.4: Implement Neighborhood Lookup
Create an efficient lookup system that can quickly retrieve semantic neighbors for any token. Include a fallback mechanism for tokens not in the pre-computed neighborhoods. This system should support both deterministic neighbor selection for reproducibility and random selection from the neighbor set for diversity.

## PHASE 3: GENETIC ALGORITHM CORE COMPONENTS

### Task 3.1: Implement PromptGenome Class
Create the PromptGenome class as the fundamental unit of evolution. Each genome instance must store a list of token IDs (limited to 200 tokens), fitness score, accuracy score, generation born, parent IDs for lineage tracking, unique genome ID, mutation count, and evaluation count. Implement methods to convert between token IDs and text representation, generate unique hashes for caching, and track genealogy.

### Task 3.2: Develop Crossover Operator
Implement a sophisticated crossover function that preserves semantic coherence. First, convert parent genomes to text and identify sentence boundaries using natural language processing tools. Select crossover points near these boundaries, with a small random offset of plus or minus 5 tokens to introduce variety. If no clear boundaries exist, fall back to midpoint crossover. The offspring should inherit the first portion from parent one and the second portion from parent two.

### Task 3.3: Create Mutation System
Build a two-level mutation system. At the population level, each genome has an 80% probability of undergoing mutation. At the token level, each token in selected genomes has a 0.2% probability of being mutated. When a token is selected for mutation, use the semantic neighborhoods 90% of the time to replace it with a semantically similar token, and 10% of the time select a completely random token from the vocabulary. Track the number of mutations applied to each genome.

### Task 3.4: Design Population Initialization
Create a function that generates the initial population of 500 genomes from 50 seed prompts. For each offspring, randomly select two parents from the seed set, apply crossover, then apply initial mutation with a slightly higher rate (0.5% per token) to ensure diversity. Mark each genome with generation zero and record parent IDs for genealogy tracking.

## PHASE 4: EVALUATION PIPELINE

### Task 4.1: Build LLM Interface
Create an evaluator class that manages all interactions with the language model API. Implement robust error handling including automatic retry with exponential backoff for rate limits and API errors. The evaluator must maintain a disk-based cache system using MD5 hashes of prompts to avoid redundant API calls. Track both the total number of API calls and cache hit rate for cost monitoring.

### Task 4.2: Implement Fitness Calculation
Develop the core evaluation function that measures genome performance. For each genome, construct full prompts by combining the evolved prompt text with each test problem. Send these to the LLM with temperature set to zero for deterministic outputs. Extract predicted answers and compare with ground truth, allowing for small numerical tolerance (0.001) to handle floating-point precision issues. Calculate accuracy as the percentage of correct answers.

### Task 4.3: Add Length Penalty
Implement a fitness modifier that penalizes excessively long prompts. Prompts under 200 tokens receive no penalty. For each token beyond 200, apply a small penalty that reduces fitness by up to 10% maximum for very long prompts. This encourages evolution toward concise, efficient prompts while allowing flexibility when length provides value.

### Task 4.4: Enable Parallel Evaluation
Create a batch evaluation system using thread pools to evaluate multiple genomes concurrently. Limit concurrent API calls to avoid rate limits while maximizing throughput. Implement graceful failure handling where failed evaluations receive zero fitness but don't crash the entire batch. Use progress bars to monitor evaluation status.

## PHASE 5: SELECTION STRATEGIES

### Task 5.1: Implement Elite Selection
Create a multi-strategy selection system that balances exploitation and exploration. Select the top 20 genomes by fitness score as elite members who are guaranteed to produce offspring. These represent the current best solutions and ensure the population doesn't regress.

### Task 5.2: Add Diversity Selection
Implement diversity-based selection that identifies 15 genomes most different from the elite set. Calculate diversity using Jaccard distance on token sets or edit distance on token sequences. This mechanism prevents premature convergence by maintaining genetic diversity even when fitness plateaus.

### Task 5.3: Include Random Selection
Add a lottery system that randomly selects 15 genomes from the remaining population regardless of fitness. This provides ongoing exploration and prevents the algorithm from getting trapped in local optima. The random selection should exclude the elite and diverse selections to maximize coverage.

### Task 5.4: Combine Selection Methods
Create a parent selection function that combines all three strategies, returning exactly 50 parents for the next generation. This balanced approach ensures both hill-climbing toward better solutions and exploration of novel prompt structures.

## PHASE 6: EVOLUTION LOOP

### Task 6.1: Implement Generation Execution
Create the main generation processing function that orchestrates one complete evolutionary cycle. This function must determine the appropriate number of test problems based on generation number (50 for early generations, 100 for middle, 150 for late generations), evaluate the entire population, calculate and log statistics, check convergence criteria, select parents, generate offspring through crossover and mutation, and save checkpoints.

### Task 6.2: Add Convergence Detection
Implement multiple convergence criteria that can terminate evolution early. Check if the best genome achieves 85% or higher accuracy on the evaluation set. Monitor fitness improvement across five consecutive generations and trigger convergence if improvement is less than 0.1%. Also implement a hard limit of 30 generations to bound computation costs.

### Task 6.3: Handle Stagnation
Create a stagnation detection system that identifies when evolution has plateaued. When the best fitness hasn't improved by more than 0.1% for five generations, temporarily double the mutation rate to escape local optima. After two generations with increased mutation, return to normal rates if improvement resumes.

### Task 6.4: Build Main Evolution Controller
Implement the top-level evolution function that initializes all components, manages the generation loop, handles recovery from interruptions, and produces final results. This controller should load configurations, initialize semantic neighborhoods, create seed populations, run the evolution loop with proper error handling, and perform final validation on the best evolved prompt.

## PHASE 7: MONITORING AND LOGGING

### Task 7.1: Create Metrics Logger
Build a comprehensive logging system that tracks all relevant metrics throughout evolution. Log per-generation statistics including best, average, and minimum fitness, population diversity metrics, average prompt length, mutation effectiveness, and cache hit rates. Store logs in JSON Lines format for easy parsing and analysis.

### Task 7.2: Implement Checkpointing
Create a checkpoint system that saves complete state after each generation. Save the entire population as a pickle file, the best genome as both pickle and human-readable text, and metadata including generation number, fitness statistics, and timestamp. Organize checkpoints in generation-specific directories for easy navigation.

### Task 7.3: Add Progress Visualization
Implement real-time progress monitoring that displays current generation, best fitness so far, average fitness, time elapsed, estimated time remaining, and current cache hit rate. Update the display after each generation to provide feedback during long-running evolution.

### Task 7.4: Enable Resume Capability
Create a recovery system that can resume evolution from the latest checkpoint after interruption. On startup, scan for existing checkpoints, load the most recent valid state, and continue evolution from that point. This ensures no work is lost due to crashes or interruptions.

## PHASE 8: SEED PROMPT CREATION

### Task 8.1: Design Seed Prompts
Create 50 diverse seed prompts that cover different problem-solving strategies. Include basic chain-of-thought prompts, structured step-by-step approaches, role-based prompts that assume teacher or mathematician personas, verification-focused prompts that emphasize checking work, format-specific prompts with clear output structure, and creative variations that might discover novel approaches.

### Task 8.2: Convert Seeds to Genomes
Transform each text prompt into a PromptGenome object. Tokenize the text using your chosen tokenizer, create genome instances with the token IDs, mark them with generation negative one to indicate seed status, and assign unique IDs for tracking. Store the seed genomes in a dedicated file for reproducibility.

### Task 8.3: Validate Seed Diversity
Analyze the seed prompts to ensure sufficient initial diversity. Calculate pairwise similarities between all seeds and verify that no two seeds are more than 90% similar. Check that seeds cover a range of lengths from very short (10 tokens) to moderately long (150 tokens). This diversity is crucial for effective evolution.

## PHASE 9: ANALYSIS AND BENCHMARKING

### Task 9.1: Implement Evolution Analysis
Create analysis functions that process the complete evolution history. Load all checkpoint files and extract fitness progression over generations. Identify phase transitions where the algorithm shifted from exploration to exploitation. Calculate diversity metrics showing how the population converged over time. Generate plots showing fitness curves, diversity trends, and prompt length distribution.

### Task 9.2: Prepare Baseline Comparisons
Set up a comparison framework that evaluates the evolved prompt against established baselines. Include a zero-shot baseline with no prompt, simple chain-of-thought with "Let's think step by step", structured approaches from the literature, and current state-of-the-art prompts achieving 82-85% accuracy. Ensure all prompts are evaluated on identical problem sets for fair comparison.

### Task 9.3: Calculate Statistical Significance
Implement statistical tests to verify that performance improvements are significant. Use binomial tests to compare accuracy rates between evolved and baseline prompts. Calculate confidence intervals using bootstrap sampling with at least 1000 iterations. Report p-values for all comparisons to demonstrate that improvements aren't due to chance.

### Task 9.4: Test Generalization
Evaluate the best evolved prompt on related datasets to test generalization. Download and test on SVAMP dataset for arithmetic word problems. Test on ASDiv dataset for diverse math problems. Compare transfer performance with baseline prompts to assess whether evolution created GSM8K-specific optimizations or general improvements.

## PHASE 10: FINAL VALIDATION AND REPORTING

### Task 10.1: Perform Comprehensive Testing
Conduct final validation of the best evolved prompt on the complete GSM8K test set of 1,319 problems. This is the definitive performance measure. Calculate exact accuracy and confidence intervals. Measure average inference time per problem. Document any failure patterns or problem types where the evolved prompt struggles.

### Task 10.2: Document Discovered Patterns
Analyze the top-performing genomes to identify common patterns that evolution discovered. Extract frequently occurring phrases or structures. Identify novel prompt engineering techniques not present in the seed prompts. Document how prompts evolved over generations, showing the progression from seeds to final forms.

### Task 10.3: Generate Final Report
Create a comprehensive report documenting the entire experiment. Include configuration details and random seeds for reproducibility, evolution statistics including total API calls and convergence generation, the complete text of the best evolved prompt, performance metrics compared to all baselines, computational resources used and total cost, and key insights about effective prompt structures for mathematical reasoning.

### Task 10.4: Package Deliverables
Prepare all deliverables for use and distribution. Save the best prompt in multiple formats (text, JSON, pickle). Create a simple inference script that applies the evolved prompt to new problems. Document how to use the evolved prompt with different LLM APIs. Archive all code, data, and results for reproducibility.

## PHASE 11: PRODUCTION DEPLOYMENT

### Task 11.1: Create Inference Pipeline
Build a production-ready pipeline for using the evolved prompt. Implement a simple interface that accepts math problems and returns solutions. Add batch processing capability for multiple problems. Include error handling for malformed inputs. Optimize for minimal latency while maintaining accuracy.

### Task 11.2: Develop Integration Guide
Write clear documentation for integrating the evolved prompt into existing systems. Provide examples for common LLM frameworks and APIs. Document optimal temperature and sampling parameters. Include troubleshooting guides for common issues. Create code snippets in multiple programming languages.

### Task 11.3: Establish Monitoring
Set up monitoring for production use of the evolved prompt. Track accuracy on real-world problems over time. Monitor for prompt degradation with model updates. Log failure cases for future improvement. Create alerts for significant performance drops.

## CRITICAL IMPLEMENTATION NOTES

### Resource Requirements
Ensure you have sufficient computational resources including at least 16GB RAM for holding populations in memory, 50GB disk space for checkpoints and cache, stable internet connection for API calls, and budget approval for $2,000-3,000 in API costs.

### Quality Assurance
Throughout implementation, maintain code quality by writing comprehensive docstrings for all functions, implementing unit tests for critical components, using type hints for better code clarity, following PEP 8 style guidelines, and conducting code reviews before major milestones.

### Risk Mitigation
Implement safeguards against common failure modes including API rate limit handling with exponential backoff, checkpoint corruption detection and recovery, out-of-memory handling for large populations, network failure resilience with retries, and cost monitoring to prevent budget overruns.

### Performance Optimization
Optimize performance through aggressive caching of API responses, batch API calls where possible, parallel evaluation using thread pools, efficient data structures for genome storage, and incremental metric calculation to avoid redundant computation.

This completes the detailed implementation instructions. Each task should be completed in sequence, with careful attention to the specifications in the referenced technical documentation. The system should achieve 85% or better accuracy on GSM8K within 30 generations and approximately $2,000-3,000 in API costs.