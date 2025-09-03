import asyncio
import random
from typing import List, Tuple

from src.evaluation.evaluator import LLMEvaluator
from src.genetics.genome import PromptGenome
from src.genetics.operators import crossover, mutate
from src.genetics.selection import select_parents
from src.evaluation.metrics import length_penalty
from src.utils.checkpointing import save_checkpoint


async def evaluate_population(population: List[PromptGenome], problems: List[dict], evaluator: LLMEvaluator, concurrency_limit: int, id2token):
    sem = asyncio.Semaphore(concurrency_limit)

    async def eval_one(g: PromptGenome):
        try:
            await evaluator.evaluate_genome(g, problems, sem, id2token)
        except Exception as e:
            g.fitness = 0.0

    await asyncio.gather(*(eval_one(g) for g in population))
    return population


def run_generation(gen_num: int, population: List[PromptGenome], problems: List[dict], evaluator: LLMEvaluator, config: dict, id2token, neighbors, vocab_size: int, logger=None, recent_best: List[float]=None, mutation_multiplier: float = 1.0) -> Tuple[List[PromptGenome], bool]:
    # Evaluate
    asyncio.run(evaluate_population(population, problems, evaluator, config["evaluation"]["concurrency_limit"], id2token))

    # Stats
    best = max(population, key=lambda x: x.fitness or 0.0)
    avg = sum((g.fitness or 0.0) for g in population) / max(1, len(population))
    print(f"Generation {gen_num}: best_fitness={best.fitness:.4f} best_acc={best.accuracy:.4f} avg_fitness={avg:.4f}")
    if logger:
        evaluator_stats = {"cache_hit_rate": evaluator.cache_hits / max(1, evaluator.call_count) if evaluator.call_count else 0.0}
        logger.log_generation(gen_num, population, evaluator_stats=evaluator_stats)

    # Convergence
    target = config.get("target_accuracy", 0.85)
    if (best.accuracy or 0.0) >= target:
        print("Converged by accuracy target")
        save_checkpoint(gen_num, population, best, dir_root=config["paths"]["checkpoints"])
        return population, True

    # Stagnation detection (last 5 gens, improvement < 0.1%)
    if recent_best is not None and len(recent_best) >= 5:
        if max(recent_best[-5:]) - min(recent_best[-5:]) < 0.001:
            print("Stagnation detected - consider increasing mutation rates externally")

    # Selection
    sel_cfg = config.get("selection", {})
    parents = select_parents(population, elite_k=sel_cfg.get("elite_count", 10), diverse_k=sel_cfg.get("diverse_count", 15), random_k=sel_cfg.get("random_count", 25))

    # Reproduction
    new_population: List[PromptGenome] = []
    for _ in range(config["population"]["population_size"]):
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        child = crossover(p1, p2, id2token=id2token, token2id=None)
        child = mutate(
            child,
            neighbors,
            vocab_size,
            pop_prob=config["mutation"]["population_prob"],
            token_prob=config["mutation"]["token_prob"] * mutation_multiplier,
        )
        child.generation_born = gen_num + 1
        child.parent_ids = [p1.genome_id, p2.genome_id]
        new_population.append(child)

    # Checkpoint
    save_checkpoint(gen_num, new_population, best, dir_root=config["paths"]["checkpoints"]) 
    return new_population, False

