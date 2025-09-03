import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class PromptGenome:
    token_ids: List[int]
    max_length: int = 200
    fitness: Optional[float] = None
    accuracy: Optional[float] = None
    generation_born: int = 0
    parent_ids: List[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mutation_count: int = 0
    evaluation_count: int = 0

    def __post_init__(self):
        if len(self.token_ids) > self.max_length:
            self.token_ids = self.token_ids[: self.max_length]

    def to_text(self, id2token: Dict[int, str]) -> str:
        return " ".join(id2token.get(tid, "") for tid in self.token_ids).strip()

    @classmethod
    def from_text(cls, text: str, token2id: Dict[str, int], **kwargs):
        tokens = [tok for tok in text.split() if tok]
        ids = [token2id[t] for t in tokens if t in token2id]
        return cls(token_ids=ids, **kwargs)

    def get_hash(self) -> str:
        # hash bytes of token_ids for caching
        return hashlib.md5(bytes(self.token_ids)).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "token_ids": self.token_ids,
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "generation_born": self.generation_born,
            "parent_ids": self.parent_ids,
            "genome_id": self.genome_id,
            "mutation_count": self.mutation_count,
            "evaluation_count": self.evaluation_count,
        }

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(
            token_ids=d["token_ids"],
            fitness=d.get("fitness"),
            accuracy=d.get("accuracy"),
            generation_born=d.get("generation_born", 0),
            parent_ids=d.get("parent_ids", []),
            genome_id=d.get("genome_id") or str(uuid.uuid4()),
            mutation_count=d.get("mutation_count", 0),
            evaluation_count=d.get("evaluation_count", 0),
        )

