"""
Prioritized experience replay for AlphaZero training samples.

Each sample: (planes, policy_target, value_target, player_color)
Only complete, verified game traces are admitted (no partial episodes).
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import TrainConfig


@dataclass
class ReplaySample:
    """Single training example from a finished self-play game."""

    planes: np.ndarray       # (C, H, W) float32
    policy: np.ndarray       # (225,) float32 — MCTS visit distribution
    value: float             # z in [-1, 1] from this player's perspective
    priority: float = 1.0


class PrioritizedReplayBuffer:
    """
    Thread-safe sum-tree-free prioritized buffer using segment sampling.

    Priority p_i = (|TD error| + eps) ^ alpha  (here: initial priority from
    game outcome magnitude; updated after each training pass).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.capacity = cfg.buffer_capacity
        self._data: List[ReplaySample] = []
        self._priorities: List[float] = []
        self._lock = threading.Lock()
        self._beta = cfg.priority_beta

    def __len__(self) -> int:
        return len(self._data)

    @property
    def beta(self) -> float:
        return self._beta

    def anneal_beta(self) -> None:
        self._beta = min(1.0, self._beta + self.cfg.priority_beta_increment)

    def add_game(
        self,
        planes_list: List[np.ndarray],
        policies_list: List[np.ndarray],
        players_list: List[int],
        outcome: float,
        verified: bool = True,
    ) -> int:
        """
        Insert a complete game trajectory.

        `outcome`: final result from BLACK's perspective (+1 win, -1 loss, 0 draw).
        Returns number of positions stored, or 0 if rejected.
        """
        if not verified or len(planes_list) == 0:
            return 0

        n = 0
        with self._lock:
            for planes, policy, player in zip(planes_list, policies_list, players_list):
                # Value target from current player's perspective
                if player == 1:  # BLACK
                    z = outcome
                else:
                    z = -outcome
                priority = (abs(z) + 0.01) ** self.cfg.priority_alpha
                sample = ReplaySample(
                    planes=planes.astype(np.float32),
                    policy=policy.astype(np.float32),
                    value=float(z),
                    priority=priority,
                )
                self._append(sample, priority)
                n += 1
        return n

    def _append(self, sample: ReplaySample, priority: float) -> None:
        if len(self._data) >= self.capacity:
            self._data.pop(0)
            self._priorities.pop(0)
        self._data.append(sample)
        self._priorities.append(priority)

    def sample(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Sample a prioritized minibatch.

        Returns:
          planes   (B, C, H, W)
          policies (B, 225)
          values   (B, 1)
          indices  (B,) int — for priority update
          weights  (B,) float — importance-sampling corrections
        """
        bs = batch_size or self.cfg.batch_size
        with self._lock:
            n = len(self._data)
            if n == 0:
                raise RuntimeError("Buffer empty.")
            bs = min(bs, n)

            probs = np.array(self._priorities, dtype=np.float64)
            probs = probs / probs.sum()
            indices = np.random.choice(n, size=bs, replace=n < bs, p=probs)

            # Importance sampling weights: w_i = (N * P(i))^(-beta)
            weights = (n * probs[indices]) ** (-self._beta)
            weights = weights / weights.max()

            planes = np.stack([self._data[i].planes for i in indices])
            policies = np.stack([self._data[i].policy for i in indices])
            values = np.array([[self._data[i].value] for i in indices], dtype=np.float32)

        return planes, policies, values, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        with self._lock:
            for idx, err in zip(indices, td_errors):
                p = (float(abs(err)) + 1e-4) ** self.cfg.priority_alpha
                self._priorities[int(idx)] = p
                self._data[int(idx)].priority = p
