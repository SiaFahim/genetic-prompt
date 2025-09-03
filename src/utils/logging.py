import json
import os
import time
from typing import List

from src.genetics.genome import PromptGenome


class JSONLLogger:
    def __init__(self, log_dir: str = "data/results/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, "metrics.jsonl")

    def log_generation(self, generation: int, population: List[PromptGenome], evaluator_stats: dict = None):
        best = max(population, key=lambda g: g.fitness or 0.0)
        avg = sum((g.fitness or 0.0) for g in population) / max(1, len(population))
        payload = {
            "generation": generation,
            "timestamp": time.time(),
            "best_fitness": best.fitness,
            "best_accuracy": best.accuracy,
            "avg_fitness": avg,
            "population_size": len(population),
        }
        if evaluator_stats:
            payload.update(evaluator_stats)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
        return best.fitness

