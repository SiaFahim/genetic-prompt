import json
import pickle
import random
from typing import Dict, List


class SemanticNeighbors:
    def __init__(self, neighborhoods_path: str, token2id_path: str, deterministic: bool = True, seed: int = 42):
        with open(token2id_path, "r") as f:
            self.token2id: Dict[str, int] = json.load(f)
        with open(neighborhoods_path, "rb") as f:
            self.neighborhoods: List[List[int]] = pickle.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.deterministic = deterministic
        self.rnd = random.Random(seed)

    def neighbors_of_id(self, token_id: int) -> List[int]:
        if 0 <= token_id < len(self.neighborhoods):
            return self.neighborhoods[token_id]
        return []

    def sample_neighbor_id(self, token_id: int, random_chance: float = 0.1, vocab_size: int = None) -> int:
        if not self.deterministic and self.rnd.random() < random_chance and vocab_size:
            return self.rnd.randrange(0, vocab_size)
        nbs = self.neighbors_of_id(token_id)
        if not nbs:
            return token_id
        return nbs[0] if self.deterministic else self.rnd.choice(nbs)

