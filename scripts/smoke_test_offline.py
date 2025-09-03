#!/usr/bin/env python3
import asyncio
import json
import os
import random
import sys

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.data import load_jsonl
from src.embeddings.semantic_utils import SemanticNeighbors
from src.genetics.genome import PromptGenome
from src.genetics.population import initialize_population
from src.genetics.evolution import run_generation


class MockEvaluator:
    def __init__(self):
        self.call_count = 0
        self.cache_hits = 0

    async def generate(self, prompt_text: str, semaphore: asyncio.Semaphore, max_tokens: int = 200, temperature: float = 0.0) -> str:
        self.call_count += 1
        return "Final answer is #### 0"

    async def evaluate_genome(self, genome, problems, semaphore, id2token, tolerance: float = 1e-3):
        # mimic LLMEvaluator semantics but without API calls
        from src.utils.answer_extraction import extract_final_answer
        from src.evaluation.metrics import accuracy_with_tolerance, length_penalty
        preds = []
        golds = []
        prompt_head = genome.to_text(id2token)
        for pr in problems:
            # use canned response
            out = await self.generate("", semaphore)
            preds.append(extract_final_answer(out))
            golds.append(pr.get("final_answer"))
        acc = accuracy_with_tolerance(preds, golds, tol=tolerance)
        pen = length_penalty(len(genome.token_ids))
        fitness = acc * pen
        genome.accuracy = acc
        genome.fitness = fitness
        genome.evaluation_count = len(problems)
        return fitness


def main():
    cfg = load_config()

    # Data (small subset)
    data_root = cfg.paths.get("data_root", "data")
    primary = load_jsonl(os.path.join(data_root, "gsm8k_primary_eval.jsonl"))[:10]

    # Vocab maps
    with open(os.path.join(data_root, "embeddings", "vocab", "token2id.json"), "r") as f:
        token2id = json.load(f)
    with open(os.path.join(data_root, "embeddings", "vocab", "id2token.json"), "r") as f:
        id2token = {int(k): v for k, v in json.load(f).items()}

    neighbors = SemanticNeighbors(
        neighborhoods_path=os.path.join(data_root, "embeddings", "neighborhoods.pkl"),
        token2id_path=os.path.join(data_root, "embeddings", "vocab", "token2id.json"),
        deterministic=True,
        seed=cfg.random_seed,
    )

    # Seeds and population (small)
    seeds = []
    top_tokens = list(token2id.keys())[:10]
    for i in range(10):
        text = " ".join(top_tokens[max(0, i-3): i+3]) or "let's think step by step"
        g = PromptGenome.from_text(text, token2id)
        g.generation_born = -1
        seeds.append(g)

    population = initialize_population(seeds, 20, neighbors, vocab_size=len(token2id), id2token=id2token)

    # Mock evaluator
    evaluator = MockEvaluator()

    # Run a single generation
    from src.genetics.evolution import run_generation
    population, done = run_generation(0, population, primary, evaluator, cfg.raw, id2token, neighbors, vocab_size=len(token2id))
    print("Smoke test completed. Best fitness:", max((g.fitness or 0.0) for g in population))


if __name__ == "__main__":
    main()

