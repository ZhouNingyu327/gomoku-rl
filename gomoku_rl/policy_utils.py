"""Numerically safe policy normalization for MCTS and self-play."""

from __future__ import annotations

import numpy as np


def sanitize_policy(probs: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """
    Return a valid probability vector over legal moves (sums to 1, no NaN/Inf).
    Falls back to uniform over legal moves when input is degenerate.
    """
    p = np.asarray(probs, dtype=np.float64).copy()
    legal = np.asarray(legal, dtype=bool)
    p[~legal] = 0.0
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0:
        p = legal.astype(np.float64)
        total = p.sum()
        if total <= 0:
            return np.ones(len(p), dtype=np.float64) / len(p)
    return p / total
