#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Ensure project root is on sys.path for module imports when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.data import load_jsonl
from src.embeddings.semantic_utils import SemanticNeighbors
from src.genetics.genome import PromptGenome
from src.genetics.population import initialize_population
from src.genetics.evolution import run_generation
from src.evaluation.evaluator import LLMEvaluator


def load_vocab_maps(root: str = "data/embeddings/vocab"):
    with open(os.path.join(root, "token2id.json"), "r") as f:
        token2id = json.load(f)
    with open(os.path.join(root, "id2token.json"), "r") as f:
        id2token = json.load(f)
    # keys in id2token are strings; convert to int
    id2token = {int(k): v for k, v in id2token.items()}
    return token2id, id2token


def main():
    p = argparse.ArgumentParser(description="Run GSM8K GA Evolution")
    p.add_argument("--config", default="configs/experiment_config.json")
    p.add_argument("--gens", type=int, default=1)
    p.add_argument("--eval_size", type=int, default=20)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.concurrency is not None:
        cfg.raw["evaluation"]["concurrency_limit"] = args.concurrency
    if args.pop is not None:
        cfg.raw["population"]["population_size"] = args.pop

    # Data
    data_root = cfg.paths.get("data_root", "data")
    primary_path = os.path.join(data_root, "gsm8k_primary_eval.jsonl")
    problems = load_jsonl(primary_path)

    # Vocab and neighborhoods
    token2id, id2token = load_vocab_maps()
    neighbors = SemanticNeighbors(
        neighborhoods_path=os.path.join(data_root, "embeddings", "neighborhoods.pkl"),
        token2id_path=os.path.join(data_root, "embeddings", "vocab", "token2id.json"),
        deterministic=True,
        seed=cfg.random_seed,
    )

    # Seeds: create trivial seed prompts from top tokens
    seeds = []
    top_tokens = list(token2id.keys())[:50]
    for i in range(50):
        text = " ".join(top_tokens[max(0, i-5): i+5]) or "let's think step by step"
        genome = PromptGenome.from_text(text, token2id)
        genome.generation_born = -1
        seeds.append(genome)

    # Population
    population = initialize_population(seeds, cfg.raw["population"]["population_size"], neighbors, vocab_size=len(token2id))

    # Evaluator
    api_key = cfg.api_keys.get("openai")
    evaluator = LLMEvaluator(api_key=api_key, model=cfg.model_name)

    # Run a small number of generations as a smoke test
    for gen in range(args.gens):
        population, done = run_generation(gen, population, problems[:args.eval_size], evaluator, cfg.raw, id2token, neighbors, vocab_size=len(token2id))
        if done:
            break


if __name__ == "__main__":
    main()

