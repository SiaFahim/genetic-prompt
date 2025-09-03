import random
from typing import List, Tuple

from nltk.tokenize import sent_tokenize

from src.genetics.genome import PromptGenome


def find_sentence_token_boundaries(text: str) -> List[int]:
    # Split into sentences and accumulate token counts as boundaries
    sentences = sent_tokenize(text)
    boundaries = []
    count = 0
    for s in sentences:
        # simple whitespace tokens; GA uses our vocab elsewhere
        toks = s.split()
        count += len(toks)
        boundaries.append(count)
    return [b for b in boundaries if b > 0]


def crossover(p1: PromptGenome, p2: PromptGenome, id2token, token2id, offset: int = 5) -> PromptGenome:
    text1 = p1.to_text(id2token)
    text2 = p2.to_text(id2token)
    b1 = find_sentence_token_boundaries(text1)
    b2 = find_sentence_token_boundaries(text2)

    if b1 and b2:
        c1 = random.choice(b1)
        c2 = random.choice(b2)
        c1 = max(0, min(len(p1.token_ids), c1 + random.randint(-offset, offset)))
        c2 = max(0, min(len(p2.token_ids), c2 + random.randint(-offset, offset)))
    else:
        c1 = len(p1.token_ids) // 2
        c2 = len(p2.token_ids) // 2

    child_ids = p1.token_ids[:c1] + p2.token_ids[c2:]
    return PromptGenome(token_ids=child_ids, max_length=p1.max_length)


def mutate(genome: PromptGenome, neighbors, vocab_size: int, pop_prob: float = 0.8, token_prob: float = 0.002) -> PromptGenome:
    if random.random() > pop_prob:
        return genome
    ids = genome.token_ids[:]
    mutations = 0
    for i in range(len(ids)):
        if random.random() < token_prob:
            old_id = ids[i]
            # 90% semantic neighbor, 10% random
            if random.random() < 0.9:
                new_id = neighbors.sample_neighbor_id(old_id, random_chance=0.0, vocab_size=vocab_size)
            else:
                new_id = random.randint(0, vocab_size - 1)
            if new_id != old_id:
                ids[i] = new_id
                mutations += 1
    if mutations > 0:
        child = PromptGenome(token_ids=ids, max_length=genome.max_length)
        child.mutation_count = genome.mutation_count + mutations
        return child
    return genome

