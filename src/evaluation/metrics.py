from typing import List, Optional


def accuracy_with_tolerance(preds: List[Optional[float]], golds: List[Optional[float]], tol: float = 1e-3) -> float:
    assert len(preds) == len(golds)
    correct = 0
    total = len(preds)
    for p, g in zip(preds, golds):
        if p is None or g is None:
            continue
        if abs(p - g) <= tol:
            correct += 1
    return correct / max(1, total)


def length_penalty(length_tokens: int, max_length: int = 200) -> float:
    if length_tokens <= max_length:
        return 1.0
    excess = length_tokens - max_length
    return max(0.0, 1.0 - (excess / 300.0) * 0.1)  # cap ~10%

