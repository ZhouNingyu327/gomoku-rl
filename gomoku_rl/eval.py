"""
Automated Elo-style evaluation for model promotion gating.

Uses win-rate threshold against a frozen baseline checkpoint before promoting
the candidate network to production self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import torch

from .config import TrainConfig
from .env import BLACK, GomokuEnv
from .mcts import MCTS
from .network import DualHeadNet


@dataclass
class EvalResult:
    candidate_wins: int
    baseline_wins: int
    draws: int
    win_rate: float
    promoted: bool


class EloEvaluator:
    """
    Pairwise evaluation: candidate (new) vs baseline (checkpoint).

    Approximate Elo update (informative only):
      E = 1 / (1 + 10^((R_b - R_a)/400))
      R_a' = R_a + K * (S - E)
    """

    def __init__(self, cfg: TrainConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    @torch.inference_mode()
    def evaluate(
        self,
        candidate: DualHeadNet,
        baseline: DualHeadNet,
        num_games: int | None = None,
    ) -> EvalResult:
        n = num_games or self.cfg.eval_games
        rng = np.random.default_rng(self.cfg.seed)

        # Temporarily reduce sims for faster eval
        eval_cfg = replace(
            self.cfg,
            num_simulations=self.cfg.eval_mcts_simulations,
            mcts_threads=1,
            use_symmetry_augment=False,
        )

        mcts_c = MCTS(eval_cfg, candidate, self.device, rng)
        mcts_b = MCTS(eval_cfg, baseline, self.device, rng)

        c_wins = b_wins = draws = 0

        for g in range(n):
            env = GomokuEnv(board_size=self.cfg.board_size)
            # Alternate colors
            candidate_is_black = g % 2 == 0
            move_idx = 0

            while env.result.name == "ONGOING":
                temp = 1e-3  # greedy during eval
                if candidate_is_black == (env.current_player == BLACK):
                    action = mcts_c.get_action(env, temperature=temp, add_root_noise=False)
                else:
                    action = mcts_b.get_action(env, temperature=temp, add_root_noise=False)
                env.step(action)
                move_idx += 1
                if move_idx > self.cfg.board_size * self.cfg.board_size:
                    break

            outcome = env.outcome_value(BLACK)
            if outcome > 0:
                winner = "candidate" if candidate_is_black else "baseline"
            elif outcome < 0:
                winner = "baseline" if candidate_is_black else "candidate"
            else:
                winner = "draw"

            if winner == "candidate":
                c_wins += 1
            elif winner == "baseline":
                b_wins += 1
            else:
                draws += 1

        decisive = c_wins + b_wins
        win_rate = c_wins / decisive if decisive > 0 else 0.5
        promoted = win_rate >= self.cfg.promotion_win_rate

        return EvalResult(
            candidate_wins=c_wins,
            baseline_wins=b_wins,
            draws=draws,
            win_rate=win_rate,
            promoted=promoted,
        )

    @staticmethod
    def elo_update(ra: float, rb: float, score: float, k: float = 32.0) -> Tuple[float, float]:
        """Return updated (Ra, Rb) given score in {1, 0.5, 0} for player a."""
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        ra_new = ra + k * (score - ea)
        rb_new = rb + k * ((1 - score) - eb)
        return ra_new, rb_new
