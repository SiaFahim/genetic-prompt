import json
import os
import numpy as np
from typing import Dict, List


def load_glove_embeddings(txt_path: str, token2id: Dict[str, int]) -> np.ndarray:
    dim = None
    vocab_size = len(token2id)
    E = None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word, vec = parts[0], parts[1:]
            if word not in token2id:
                continue
            if dim is None:
                dim = len(vec)
                E = np.zeros((vocab_size, dim), dtype=np.float32)
            E[token2id[word]] = np.asarray(vec, dtype=np.float32)
    # For tokens not found, keep zeros; we can handle fallback later
    if E is None:
        raise ValueError("No embeddings matched the vocab. Check tokenization and GloVe file path.")
    # Normalize for cosine similarity
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    E = E / norms
    return E


def compute_neighborhoods(embeddings: np.ndarray, k: int = 50) -> List[List[int]]:
    # Cosine similarity matrix via dot product (already normalized)
    # For large vocabs, this is O(V^2). Consider block processing for very large V.
    sims = embeddings @ embeddings.T
    # For each row, get top-k neighbor indices (excluding self)
    V = embeddings.shape[0]
    neighborhoods: List[List[int]] = []
    for i in range(V):
        row = sims[i]
        # argsort descending, skip self
        idx = np.argpartition(-row, range(1, k+1))[:k+1]  # includes self likely
        # refine exact order for these candidates
        idx = idx[np.argsort(-row[idx])]
        idx = [j for j in idx if j != i][:k]
        neighborhoods.append(idx)
    return neighborhoods


def save_neighborhoods(path: str, neighborhoods: List[List[int]]):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(neighborhoods, f)


def build_neighborhoods(glove_txt: str, token2id_path: str, out_path: str, k: int = 50):
    with open(token2id_path, "r") as f:
        token2id = json.load(f)
    E = load_glove_embeddings(glove_txt, token2id)
    nbs = compute_neighborhoods(E, k=k)
    save_neighborhoods(out_path, nbs)
    print(f"Saved neighborhoods (k={k}) to {out_path}")

