"""
Monte Carlo Tree Search with PUCT and Dirichlet root noise.

PUCT selection (child a at state s):
  U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

where Q is mean action value, P is prior from the policy network, N visit counts.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import TrainConfig
from .env import GomokuEnv
from .network import DualHeadNet
from . import symmetry as sym


@dataclass
class MCTSNode:
    """Single node in the search tree."""

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """
    Batched, multi-threaded MCTS engine.

    Symmetry augmentation: during leaf evaluation, a random dihedral transform
    is applied to the observation; policy logits are inverse-mapped and averaged
    across all 8 symmetries when `use_symmetry_augment` is enabled.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        network: DualHeadNet,
        device: torch.device,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.net = network
        self.device = device
        self.rng = rng or np.random.default_rng()
        self._lock = threading.Lock()
        self.net.eval()

    # ------------------------------------------------------------------ public
    def run(
        self,
        env: GomokuEnv,
        add_root_noise: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, MCTSNode]:
        """
        Execute `num_simulations` PUCT simulations from `env` state.

        Returns:
          action_probs: (225,) normalized visit count distribution
          root:         root MCTSNode (for analysis / training targets)
        """
        root = MCTSNode(prior=1.0)
        legal = env.legal_moves_mask()

        # Expand root once to seed priors
        priors, _ = self._evaluate(env, legal)
        if add_root_noise:
            priors = self._add_dirichlet_noise(priors, legal)
        self._expand(root, priors, legal)

        sims = self.cfg.num_simulations
        threads = max(1, self.cfg.mcts_threads)
        chunk = math.ceil(sims / threads)

        def worker(n: int) -> None:
            for _ in range(n):
                self._simulate(env.clone(), root)

        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(worker, chunk) for _ in range(threads)]
            for f in futures:
                f.result()

        return self._action_probs(root, legal, temperature), root

    def get_action(
        self,
        env: GomokuEnv,
        temperature: float = 1.0,
        add_root_noise: bool = True,
    ) -> int:
        probs, _ = self.run(env, add_root_noise=add_root_noise, temperature=temperature)
        if temperature < 1e-3:
            return int(np.argmax(probs))
        return int(self.rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------ core
    def _simulate(self, env: GomokuEnv, root: MCTSNode) -> None:
        node = root
        search_path: List[Tuple[MCTSNode, int, GomokuEnv]] = [(node, -1, env)]
        current_env = env

        # Selection
        while node.is_expanded and node.children:
            action = self._select_child(node)
            step = current_env.step(action)
            node = node.children[action]
            search_path.append((node, action, current_env))
            if step.terminated:
                leaf_value = self._terminal_value(current_env, root_player=env.current_player)
                self._backpropagate(search_path, leaf_value)
                return

        # Leaf evaluation
        legal = current_env.legal_moves_mask()
        if not np.any(legal):
            self._backpropagate(search_path, 0.0)
            return

        priors, value = self._evaluate(current_env, legal)
        with self._lock:
            if not node.is_expanded:
                self._expand(node, priors, legal)

        # Value from leaf player's perspective -> backprop from root player's view
        leaf_player = current_env.current_player
        root_player = env.current_player
        v = float(value) if leaf_player == root_player else -float(value)
        self._backpropagate(search_path, v)

    def _select_child(self, node: MCTSNode) -> int:
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)
        best_score = -float("inf")
        best_action = -1

        for action, child in node.children.items():
            # PUCT: Q + U
            q = child.q_value
            u = (
                self.cfg.c_puct
                * child.prior
                * sqrt_total
                / (1.0 + child.visit_count)
            )
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expand(self, node: MCTSNode, priors: np.ndarray, legal: np.ndarray) -> None:
        for idx in np.flatnonzero(legal):
            a = int(idx)
            node.children[a] = MCTSNode(prior=float(priors[a]))
        node.is_expanded = True

    def _backpropagate(
        self,
        path: List[Tuple[MCTSNode, int, GomokuEnv]],
        value: float,
    ) -> None:
        v = value
        for node, action, env in reversed(path):
            with self._lock:
                node.visit_count += 1
                node.value_sum += v
            v = -v  # zero-sum alternation

    # ------------------------------------------------------------------ network
    @torch.inference_mode()
    def _evaluate(self, env: GomokuEnv, legal: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.cfg.mcts_use_symmetry:
            policies: List[np.ndarray] = []
            values: List[float] = []
            obs = env.get_observation()
            size = self.cfg.board_size
            for sid in range(8):
                aug_obs = sym.transform_board(obs, sid, size)
                p, v = self._net_forward(aug_obs, legal, sym_id=sid)
                policies.append(p)
                values.append(v)
            policy = sym.average_policies(policies)
            policy = policy / (policy.sum() + 1e-8)
            value = float(np.mean(values))
            return policy, value

        obs = env.get_observation()
        policy, value = self._net_forward(obs, legal, sym_id=0)
        return policy, float(value)

    def _net_forward(
        self,
        obs: np.ndarray,
        legal: np.ndarray,
        sym_id: int = 0,
    ) -> Tuple[np.ndarray, float]:
        size = self.cfg.board_size
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(legal).unsqueeze(0).to(self.device)
        if sym_id != 0:
            # legal mask must follow the same symmetry
            fwd, _ = sym.get_symmetries(size)[sym_id]
            legal_aug = legal[fwd]
            mask = torch.from_numpy(legal_aug).unsqueeze(0).to(self.device)

        policy_t, value_t = self.net.predict(x, mask)
        policy = policy_t.squeeze(0).cpu().numpy()
        if sym_id != 0:
            policy = sym.inverse_transform_policy(policy, sym_id, size)
        return policy, value_t.item()

    # ------------------------------------------------------------------ helpers
    def _add_dirichlet_noise(self, priors: np.ndarray, legal: np.ndarray) -> np.ndarray:
        legal_idx = np.flatnonzero(legal)
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(legal_idx))
        p = priors.copy()
        eps = self.cfg.dirichlet_epsilon
        p[legal_idx] = (1 - eps) * p[legal_idx] + eps * noise
        p = p / p.sum()
        return p

    @staticmethod
    def _action_probs(
        root: MCTSNode,
        legal: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        visits = np.zeros(len(legal), dtype=np.float64)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        visits[~legal] = 0.0
        if visits.sum() == 0:
            visits[legal] = 1.0
        if temperature < 1e-3:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
            return probs
        visits = visits ** (1.0 / temperature)
        return visits / visits.sum()

    @staticmethod
    def _terminal_value(env: GomokuEnv, root_player: int) -> float:
        return env.outcome_value(root_player)


def batch_mcts_policies(
    mcts: MCTS,
    envs: List[GomokuEnv],
    temperatures: List[float],
    add_root_noise: bool = True,
) -> List[np.ndarray]:
    """Run MCTS on a list of environments (sequential; GPU batching in network)."""
    return [
        mcts.run(e, add_root_noise=add_root_noise, temperature=t)[0]
        for e, t in zip(envs, temperatures)
    ]
