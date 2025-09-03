import glob
import json
import os
import pickle
from typing import List, Optional, Tuple

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


def find_latest_checkpoint(dir_root: str = "data/checkpoints") -> Optional[Tuple[int, str]]:
    paths = sorted(glob.glob(os.path.join(dir_root, "gen_*")))
    if not paths:
        return None
    latest = paths[-1]
    gen_str = os.path.basename(latest).split("_")[-1]
    try:
        gen = int(gen_str)
    except ValueError:
        gen = None
    return (gen, latest) if gen is not None else None


def load_checkpoint(path: str) -> Tuple[List[PromptGenome], PromptGenome, dict]:
    with open(os.path.join(path, "population.pkl"), "rb") as f:
        population: List[PromptGenome] = pickle.load(f)
    with open(os.path.join(path, "metadata.json"), "r") as f:
        meta = json.load(f)
    best = max(population, key=lambda g: g.fitness or 0.0)
    return population, best, meta

