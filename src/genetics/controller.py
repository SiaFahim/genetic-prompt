import os
import random
from typing import List

from src.utils.config import load_config
from src.utils.data import load_jsonl
from src.embeddings.semantic_utils import SemanticNeighbors
from src.genetics.genome import PromptGenome
from src.genetics.population import initialize_population
from src.genetics.evolution import run_generation
from src.utils.logging import JSONLLogger
from src.evaluation.evaluator import LLMEvaluator


def evolve(config_path: str = "configs/experiment_config.json"):
    cfg = load_config(config_path)

    # Data
    data_root = cfg.paths.get("data_root", "data")
    primary_path = os.path.join(data_root, "gsm8k_primary_eval.jsonl")
    full_val_path = os.path.join(data_root, "gsm8k_validation.jsonl")
    problems = load_jsonl(primary_path)

    # Vocab and neighborhoods
    token2id_path = os.path.join(data_root, "embeddings", "vocab", "token2id.json")
    id2token_path = os.path.join(data_root, "embeddings", "vocab", "id2token.json")
    import json
    with open(token2id_path, "r") as f:
        token2id = json.load(f)
    with open(id2token_path, "r") as f:
        id2token = {int(k): v for k, v in json.load(f).items()}

    neighbors = SemanticNeighbors(
        neighborhoods_path=os.path.join(data_root, "embeddings", "neighborhoods.pkl"),
        token2id_path=token2id_path,
        deterministic=True,
        seed=cfg.random_seed,
    )

    # Seeds: simple seeds from top tokens
    seeds: List[PromptGenome] = []
    top_tokens = list(token2id.keys())[:50]
    for i in range(50):
        text = " ".join(top_tokens[max(0, i-5): i+5]) or "let's think step by step"
        g = PromptGenome.from_text(text, token2id)
        g.generation_born = -1
        seeds.append(g)

    population = initialize_population(seeds, cfg.raw["population"]["population_size"], neighbors, vocab_size=len(token2id))

    # Evaluator
    api_key = cfg.api_keys.get("openai")
    evaluator = LLMEvaluator(api_key=api_key, model=cfg.model_name)

    # Generation loop (short placeholder: 2 gens)
    logger = JSONLLogger(cfg.paths.get("logs", "data/results/logs"))

    max_gens = min(2, cfg.raw["population"]["max_generations"])  # keep small for smoke
    recent_best: list[float] = []
    for gen in range(max_gens):
        # Adaptive evaluation size based on gen
        if gen < 10:
            eval_size = cfg.raw["evaluation"]["eval_problems_gen1_10"]
        elif gen < 20:
            eval_size = cfg.raw["evaluation"]["eval_problems_gen11_20"]
        else:
            eval_size = cfg.raw["evaluation"]["eval_problems_gen21_30"]

        population, done = run_generation(
            gen,
            population,
            problems[:eval_size],
            evaluator,
            cfg.raw,
            id2token,
            neighbors,
            vocab_size=len(token2id),
            logger=logger,
            recent_best=recent_best,
        )
        # Track best fitness
        best = max(population, key=lambda g: g.fitness or 0.0)
        recent_best.append(best.fitness or 0.0)
        if done:
            break

    return population

