import json
import os
import time
from typing import List

from src.genetics.genome import PromptGenome


def calc_diversity(population: List[PromptGenome], sample_pairs: int = 50) -> float:
    import random
    if len(population) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for _ in range(sample_pairs):
        a, b = random.sample(population, 2)
        sa, sb = set(a.token_ids), set(b.token_ids)
        union = len(sa | sb) or 1
        jacc = len(sa & sb) / union
        total += 1.0 - jacc
        pairs += 1
    return total / max(1, pairs)


class JSONLLogger:
    def __init__(self, log_dir: str = "data/results/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, "metrics.jsonl")

    def log_generation(self, generation: int, population: List[PromptGenome], evaluator_stats: dict = None, id2token=None):
        best = max(population, key=lambda g: g.fitness or 0.0)
        avg = sum((g.fitness or 0.0) for g in population) / max(1, len(population))
        diversity = calc_diversity(population)
        best_text = None
        if id2token is not None:
            try:
                best_text = best.to_text(id2token)
            except Exception:
                best_text = None
        payload = {
            "generation": generation,
            "timestamp": time.time(),
            "best_fitness": best.fitness,
            "best_accuracy": best.accuracy,
            "avg_fitness": avg,
            "diversity": diversity,
            "population_size": len(population),
            "best_text": best_text,
        }
        if evaluator_stats:
            payload.update(evaluator_stats)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        return best.fitness

