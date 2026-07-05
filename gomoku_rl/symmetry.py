"""Eight-fold dihedral symmetries for 15x15 Gomoku boards and move indices."""

from __future__ import annotations

import numpy as np

# Each symmetry: (transform_id, inverse_id)
# transform maps flat index i -> j; inverse maps j -> i for policy/value alignment.
_SYMMETRIES: list[tuple[np.ndarray, np.ndarray]] = []


def _build_symmetries(size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build 8 board transforms and inverse index permutations."""
    syms: list[tuple[np.ndarray, np.ndarray]] = []
    board = np.arange(size * size, dtype=np.int64).reshape(size, size)

    def add(transform_fn):
        t_board = transform_fn(board)
        fwd = t_board.reshape(-1).copy()
        inv = np.empty_like(fwd)
        inv[fwd] = np.arange(size * size, dtype=np.int64)
        syms.append((fwd, inv))

    add(lambda b: b)  # identity
    add(lambda b: np.rot90(b, 1))
    add(lambda b: np.rot90(b, 2))
    add(lambda b: np.rot90(b, 3))
    add(lambda b: np.fliplr(b))
    add(lambda b: np.fliplr(np.rot90(b, 1)))
    add(lambda b: np.fliplr(np.rot90(b, 2)))
    add(lambda b: np.fliplr(np.rot90(b, 3)))
    return syms


def get_symmetries(size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    global _SYMMETRIES
    if not _SYMMETRIES or len(_SYMMETRIES[0][0]) != size * size:
        _SYMMETRIES = _build_symmetries(size)
    return _SYMMETRIES


def transform_board(planes: np.ndarray, sym_id: int, size: int) -> np.ndarray:
    """Apply symmetry `sym_id` to (C, H, W) planes."""
    syms = get_symmetries(size)
    fwd, _ = syms[sym_id]
    out = planes.copy()
    for c in range(planes.shape[0]):
        flat = planes[c].reshape(-1)[fwd].reshape(size, size)
        out[c] = flat
    return out


def transform_policy(policy: np.ndarray, sym_id: int, size: int) -> np.ndarray:
    """Permute flat policy (225,) under symmetry."""
    syms = get_symmetries(size)
    fwd, _ = syms[sym_id]
    return policy[fwd]


def inverse_transform_policy(policy: np.ndarray, sym_id: int, size: int) -> np.ndarray:
    """Map augmented policy back to canonical move ordering."""
    syms = get_symmetries(size)
    _, inv = syms[sym_id]
    return policy[inv]


def average_policies(policies: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean over symmetry-augmented policy logits/probs."""
    return np.mean(np.stack(policies, axis=0), axis=0)


def random_symmetry_id(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 8))
