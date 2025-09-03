import json
import os
import re
from collections import Counter
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]  # lowercase


def iter_jsonl(paths: Iterable[str]):
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def build_vocab(jsonl_paths: List[str], out_dir: str, max_tokens: int = 20000, min_freq: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    counter = Counter()

    for row in iter_jsonl(jsonl_paths):
        counter.update(tokenize(row.get("question", "")))
        counter.update(tokenize(row.get("answer", "")))

    # Sort by frequency then lexicographically for determinism
    sorted_tokens = sorted([t for t, c in counter.items() if c >= min_freq], key=lambda t: (-counter[t], t))
    trimmed = sorted_tokens[:max_tokens]

    token2id = {t: i for i, t in enumerate(trimmed)}
    id2token = {i: t for t, i in token2id.items()}

    with open(os.path.join(out_dir, "token2id.json"), "w") as f:
        json.dump(token2id, f)
    with open(os.path.join(out_dir, "id2token.json"), "w") as f:
        json.dump(id2token, f)

    print(f"Vocab built: {len(trimmed)} tokens -> {out_dir}")

