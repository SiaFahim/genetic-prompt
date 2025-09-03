import random
from typing import List

from src.genetics.genome import PromptGenome
from src.genetics.operators import crossover, mutate


def initialize_population(seed_genomes: List[PromptGenome], population_size: int, neighbors, vocab_size: int, id2token) -> List[PromptGenome]:
    population: List[PromptGenome] = []
    for _ in range(population_size):
        p1 = random.choice(seed_genomes)
        p2 = random.choice(seed_genomes)
        child = crossover(p1, p2, id2token=id2token, token2id=None)
        # higher initial mutation rate per spec
        child = mutate(child, neighbors, vocab_size, pop_prob=0.9, token_prob=0.005)
        child.generation_born = 0
        child.parent_ids = [p1.genome_id, p2.genome_id]
        population.append(child)
    return population

