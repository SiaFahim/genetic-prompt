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

    # Resolve project root (directory that contains 'src') to make paths absolute
    here = os.getcwd()
    candidates = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    project_root = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "src")):
            project_root = c
            break
    if project_root is None:
        project_root = here

    # Data (absolute paths)
    data_root = cfg.paths.get("data_root", "data")
    if not os.path.isabs(data_root):
        data_root = os.path.join(project_root, data_root)
    primary_path = os.path.join(data_root, "gsm8k_primary_eval.jsonl")
    full_val_path = os.path.join(data_root, "gsm8k_validation.jsonl")
    problems = load_jsonl(primary_path)

    # Vocab and neighborhoods (absolute paths)
    embeddings_root = os.path.join(project_root, "data", "embeddings")
    token2id_path = os.path.join(embeddings_root, "vocab", "token2id.json")
    id2token_path = os.path.join(embeddings_root, "vocab", "id2token.json")
    import json
    with open(token2id_path, "r") as f:
        token2id = json.load(f)
    with open(id2token_path, "r") as f:
        id2token = {int(k): v for k, v in json.load(f).items()}

    neighbors = SemanticNeighbors(
        neighborhoods_path=os.path.join(embeddings_root, "neighborhoods.pkl"),
        token2id_path=token2id_path,
        deterministic=True,
        seed=cfg.random_seed,
    )

    # Seeds: curated prompts
    from src.genetics.seeds import SEED_PROMPTS
    seeds: List[PromptGenome] = []
    for text in SEED_PROMPTS:
        g = PromptGenome.from_text(text, token2id)
        g.generation_born = -1
        seeds.append(g)

    population = initialize_population(seeds, cfg.raw["population"]["population_size"], neighbors, vocab_size=len(token2id))

    # Evaluator
    api_key = cfg.api_keys.get("openai")
    evaluator = LLMEvaluator(api_key=api_key, model=cfg.model_name)

    # Generation loop (short placeholder: 2 gens)
    logger = JSONLLogger(cfg.paths.get("logs", "data/results/logs"))

    max_gens = cfg.raw["population"]["max_generations"]
    recent_best: list[float] = []
    mutation_multiplier = 1.0
    bump_generations_remaining = 0

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
            mutation_multiplier=mutation_multiplier,
        )
        # Track best fitness
        best = max(population, key=lambda g: g.fitness or 0.0)
        recent_best.append(best.fitness or 0.0)

        # Handle stagnation: if last 5 gens tiny improvement, bump mutation for 2 next gens
        if len(recent_best) >= 5 and max(recent_best[-5:]) - min(recent_best[-5:]) < 0.001:
            if bump_generations_remaining == 0:
                print("Stagnation detected: temporarily doubling mutation rate for next 2 generations")
                mutation_multiplier = 2.0
                bump_generations_remaining = 2
        if bump_generations_remaining > 0:
            bump_generations_remaining -= 1
            if bump_generations_remaining == 0:
                mutation_multiplier = 1.0

        if done:
            break

    return population

