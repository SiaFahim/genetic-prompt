import json
import os
import pickle
from typing import List

from src.genetics.genome import PromptGenome


def save_checkpoint(generation: int, population: List[PromptGenome], best: PromptGenome, dir_root: str = "data/checkpoints"):
    d = os.path.join(dir_root, f"gen_{generation:04d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "population.pkl"), "wb") as f:
        pickle.dump(population, f)
    with open(os.path.join(d, "best_prompt.txt"), "w") as f:
        f.write(" ".join(map(str, best.token_ids)))
    meta = {
        "generation": generation,
        "best_fitness": best.fitness,
        "best_accuracy": best.accuracy,
        "population_size": len(population),
    }
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

