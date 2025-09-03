import asyncio
import os
from typing import List

from openai import AsyncOpenAI

from src.evaluation.cache_manager import DiskCache
from src.utils.answer_extraction import extract_final_answer
from src.evaluation.metrics import accuracy_with_tolerance, length_penalty


class LLMEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o", cache_dir: str = "data/cache"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache = DiskCache(cache_dir)
        self.call_count = 0
        self.cache_hits = 0

    async def generate(self, prompt_text: str, semaphore: asyncio.Semaphore, max_tokens: int = 200, temperature: float = 0.0) -> str:
        key = DiskCache.key_for(prompt_text)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt_text}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    text = resp.choices[0].message.content
                    self.cache[key] = text
                    self.call_count += 1
                    return text
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("API call failed after retries")

    async def evaluate_genome(self, genome, problems: List[dict], semaphore: asyncio.Semaphore, id2token, tolerance: float = 1e-3) -> float:
        prompt_head = genome.to_text(id2token)
        preds = []
        golds = []
        for pr in problems:
            full_prompt = f"{prompt_head}\n\nQuestion: {pr['question']}\nAnswer:"
            out = await self.generate(full_prompt, semaphore)
            preds.append(extract_final_answer(out))
            golds.append(pr.get("final_answer"))
        acc = accuracy_with_tolerance(preds, golds, tol=tolerance)
        pen = length_penalty(len(genome.token_ids))
        fitness = acc * pen
        genome.accuracy = acc
        genome.fitness = fitness
        genome.evaluation_count = len(problems)
        return fitness

