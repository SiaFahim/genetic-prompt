import random
from typing import List

from src.genetics.genome import PromptGenome


def jaccard_distance(a: PromptGenome, b: PromptGenome) -> float:
    sa = set(a.token_ids)
    sb = set(b.token_ids)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    jaccard = inter / union
    return 1.0 - jaccard


def select_elite(population: List[PromptGenome], k: int) -> List[PromptGenome]:
    return sorted(population, key=lambda g: (g.fitness or 0), reverse=True)[:k]


def select_diverse(candidates: List[PromptGenome], reference: List[PromptGenome], n: int) -> List[PromptGenome]:
    scored = []
    for c in candidates:
        if not reference:
            scored.append((c, 0.0))
            continue
        min_sim = 1.0
        for r in reference:
            d = 1.0 - (1.0 - jaccard_distance(c, r))  # convert back to similarity
            min_sim = min(min_sim, d)
        scored.append((c, min_sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in scored[:n]]


def select_parents(population: List[PromptGenome], elite_k: int = 10, diverse_k: int = 15, random_k: int = 25) -> List[PromptGenome]:
    elite = select_elite(population, elite_k)
    remaining = [g for g in population if g not in elite]
    diverse = select_diverse(remaining, elite, diverse_k)
    remaining2 = [g for g in remaining if g not in diverse]
    random_sel = random.sample(remaining2, min(random_k, len(remaining2)))
    return elite + diverse + random_sel

