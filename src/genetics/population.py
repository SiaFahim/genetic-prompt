"""
Population initialization and management for genetic algorithm.
Handles creation of initial population from seed prompts.
"""

import random
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from src.genetics.genome import PromptGenome
from src.genetics.crossover import CrossoverOperator
from src.genetics.mutation import MutationOperator
from src.genetics.random_seeds import RandomSeedGenerator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class PopulationInitializer:
    """Handles initialization of genetic algorithm populations."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize population initializer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.crossover_op = CrossoverOperator(config_path)
        self.mutation_op = MutationOperator(config_path)
        self.random_seed_generator = RandomSeedGenerator(config_path)

        # Configuration
        self.population_size = self.config.get('genetic_algorithm.population_size', 500)
        self.random_seed = self.config.get('experiment.random_seed', 42)

        # Random seed configuration
        self.use_random_seed_prompts = self.config.get('genetic_algorithm.use_random_seed_prompts', False)

        # Paths
        self.seeds_dir = Path(self.config.get('paths.seeds_dir', './data/seeds'))
        self.seeds_dir.mkdir(parents=True, exist_ok=True)
    
    def load_seed_prompts(self, seed_file: Optional[str] = None) -> List[str]:
        """
        Load seed prompts from file, generate random seeds, or return default prompts.

        Args:
            seed_file: Optional path to seed file

        Returns:
            List of seed prompt strings
        """
        # Check if random seed prompts are enabled
        if self.use_random_seed_prompts:
            logger.info("Using random seed prompts (pure randomness initialization)")
            return self.random_seed_generator.generate_random_seed_prompts()

        # Load from file if specified
        if seed_file and Path(seed_file).exists():
            try:
                with open(seed_file, 'r') as f:
                    seeds = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(seeds)} seed prompts from {seed_file}")
                return seeds
            except Exception as e:
                logger.warning(f"Failed to load seed file {seed_file}: {e}")

        # Default seed prompts for GSM8K math problems
        default_seeds = [
            "Solve this step by step.",
            "Let's work through this problem carefully.",
            "First, identify what we need to find.",
            "Break down the problem into smaller parts.",
            "Calculate each step systematically.",
            "Let me solve this math problem step by step.",
            "To find the answer, I need to:",
            "Let's start by understanding what the problem is asking.",
            "I'll solve this by working through each part.",
            "Here's how to approach this problem:",
            "Step 1: Read the problem carefully.",
            "Let's organize the given information.",
            "I need to find the total by adding up all parts.",
            "To solve this, I'll use basic arithmetic.",
            "Let me calculate this step by step.",
            "First, let's identify the key numbers.",
            "I'll work through this systematically.",
            "Let's solve this math problem together.",
            "To get the answer, I need to calculate:",
            "Here's my step-by-step solution:",
            "Let me break this down into simple steps.",
            "I'll solve this using logical reasoning.",
            "First, I'll find what we know.",
            "Let's calculate the answer step by step.",
            "To solve this problem, I will:",
            "Let me work through this calculation.",
            "I need to find the solution by:",
            "Here's how I'll approach this:",
            "Let's solve this math word problem.",
            "I'll calculate the answer systematically.",
            "First, let me understand the problem.",
            "To find the solution, I'll:",
            "Let me solve this problem carefully.",
            "I'll work through each calculation.",
            "Here's my mathematical approach:",
            "Let's find the answer step by step.",
            "I need to calculate the total amount.",
            "To solve this, I'll use math operations.",
            "Let me figure out the answer.",
            "I'll solve this problem methodically.",
            "First, I'll identify the operation needed.",
            "Let's work out the solution together.",
            "I need to find the correct answer by:",
            "Here's how to solve this math problem:",
            "Let me calculate the final result.",
            "I'll solve this using arithmetic.",
            "To get the right answer, I'll:",
            "Let me work through this step by step.",
            "I need to solve for the unknown value.",
            "Here's my solution to this problem:"
        ]
        
        logger.info(f"Using {len(default_seeds)} default seed prompts")
        return default_seeds
    
    def create_genome_from_text(self, text: str, generation: int = 0) -> PromptGenome:
        """
        Create a genome from text string.
        
        Args:
            text: Text to convert to genome
            generation: Generation number
            
        Returns:
            PromptGenome instance
        """
        tokens = text.strip().split()
        genome = PromptGenome(tokens)
        genome.generation_born = generation
        genome.created_timestamp = time.time()
        
        return genome
    
    def initialize_population(self, seed_prompts: Optional[List[str]] = None,
                            target_size: Optional[int] = None) -> List[PromptGenome]:
        """
        Initialize population from seed prompts using crossover and mutation.
        
        Args:
            seed_prompts: Optional list of seed prompts
            target_size: Optional target population size
            
        Returns:
            List of initialized genomes
        """
        if target_size is None:
            target_size = self.population_size
        
        if seed_prompts is None:
            seed_prompts = self.load_seed_prompts()
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        logger.info(f"Initializing population of {target_size} genomes from {len(seed_prompts)} seeds")
        
        # Create initial genomes from seeds
        initial_genomes = []
        for i, seed_text in enumerate(seed_prompts):
            genome = self.create_genome_from_text(seed_text, generation=0)
            genome.metadata['seed_index'] = i
            genome.metadata['initialization_method'] = 'seed'
            initial_genomes.append(genome)
        
        population = initial_genomes.copy()
        
        # Generate additional genomes through crossover and mutation
        while len(population) < target_size:
            if len(population) >= 2:
                # Create offspring through crossover
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                
                if parent1 != parent2:  # Avoid self-crossover
                    offspring1, offspring2 = self.crossover_op.crossover(parent1, parent2)
                    
                    # Mutate offspring
                    offspring1 = self.mutation_op.mutate_genome(offspring1)
                    offspring2 = self.mutation_op.mutate_genome(offspring2)
                    
                    # Add metadata
                    offspring1.metadata['initialization_method'] = 'crossover_mutation'
                    offspring2.metadata['initialization_method'] = 'crossover_mutation'
                    
                    population.append(offspring1)
                    if len(population) < target_size:
                        population.append(offspring2)
                else:
                    # If we somehow selected the same parent twice, just mutate one
                    mutated = self.mutation_op.mutate_genome(parent1)
                    mutated.metadata['initialization_method'] = 'mutation_only'
                    population.append(mutated)
            else:
                # If we have fewer than 2 genomes, just mutate existing ones
                if population:
                    parent = random.choice(population)
                    mutated = self.mutation_op.mutate_genome(parent)
                    mutated.metadata['initialization_method'] = 'mutation_only'
                    population.append(mutated)
                else:
                    # This shouldn't happen, but just in case
                    logger.warning("No genomes available for population expansion")
                    break
        
        # Trim to exact target size if we overshot
        population = population[:target_size]
        
        # Assign unique IDs and final metadata
        for i, genome in enumerate(population):
            genome.metadata['population_index'] = i
            genome.metadata['initialization_timestamp'] = time.time()
        
        logger.info(f"Created population of {len(population)} genomes")
        
        # Log initialization statistics
        methods = {}
        for genome in population:
            method = genome.metadata.get('initialization_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        logger.info(f"Initialization methods: {methods}")
        
        return population
    
    def create_diverse_population(self, seed_prompts: Optional[List[str]] = None,
                                target_size: Optional[int] = None,
                                diversity_factor: float = 0.3) -> List[PromptGenome]:
        """
        Create a diverse population with emphasis on genetic diversity.
        
        Args:
            seed_prompts: Optional list of seed prompts
            target_size: Optional target population size
            diversity_factor: Factor controlling diversity emphasis (0-1)
            
        Returns:
            List of diverse genomes
        """
        if target_size is None:
            target_size = self.population_size
        
        if seed_prompts is None:
            seed_prompts = self.load_seed_prompts()
        
        logger.info(f"Creating diverse population with diversity factor {diversity_factor}")
        
        # Start with regular initialization
        population = self.initialize_population(seed_prompts, target_size // 2)
        
        # Add diverse variants
        while len(population) < target_size:
            # Select a random genome as base
            base_genome = random.choice(population)
            
            # Create a highly mutated variant
            diverse_genome = base_genome.copy()
            
            # Apply multiple rounds of mutation for diversity
            mutation_rounds = random.randint(2, 5)
            for _ in range(mutation_rounds):
                diverse_genome = self.mutation_op.mutate_genome(diverse_genome, is_stagnant=True)
            
            diverse_genome.metadata['initialization_method'] = 'diversity_enhanced'
            diverse_genome.metadata['mutation_rounds'] = mutation_rounds
            
            population.append(diverse_genome)
        
        # Trim to exact size
        population = population[:target_size]
        
        logger.info(f"Created diverse population of {len(population)} genomes")
        return population
    
    def validate_population(self, population: List[PromptGenome]) -> Dict[str, Any]:
        """
        Validate population and return statistics.
        
        Args:
            population: Population to validate
            
        Returns:
            Dictionary with validation statistics
        """
        if not population:
            return {'error': 'Empty population'}
        
        # Basic statistics
        lengths = [len(genome.tokens) for genome in population]
        unique_hashes = len(set(genome.get_hash() for genome in population))
        
        # Genealogy statistics
        methods = {}
        generations = {}
        
        for genome in population:
            method = genome.metadata.get('initialization_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
            
            gen = genome.generation_born
            generations[gen] = generations.get(gen, 0) + 1
        
        stats = {
            'population_size': len(population),
            'unique_genomes': unique_hashes,
            'duplicate_rate': 1 - (unique_hashes / len(population)),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'initialization_methods': methods,
            'generation_distribution': generations,
            'all_have_ids': all(hasattr(g, 'genome_id') and g.genome_id for g in population),
            'all_have_tokens': all(g.tokens for g in population)
        }
        
        return stats
