"""
Asynchronous off-policy AlphaZero training pipeline.

Architecture
------------
  Actor processes  : self-play with MCTS -> push complete games to replay buffer
  Learner thread   : sample buffer -> policy CE + value MSE -> optimizer step
  Evaluator        : periodic candidate vs baseline gate (win rate > threshold)

Decoupling allows GPU-saturated learning while CPUs/GPUs generate games in parallel.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

# Prevent OpenBLAS / MKL thread explosion across actor + MCTS workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .buffer import PrioritizedReplayBuffer
from .config import TrainConfig
from .env import BLACK, GomokuEnv, GameResult
from .eval import EloEvaluator
from .mcts import MCTS
from .network import DualHeadNet, build_network
from . import symmetry as sym

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ self-play
def _play_game(
    cfg: TrainConfig,
    network_state: dict,
    device_str: str,
    seed: int,
    net: Optional[DualHeadNet] = None,
) -> Optional[dict]:
    """
    Worker: run one self-play game, return verified trajectory or None.

    Returns dict with keys: planes, policies, players, outcome (black perspective).
    """
    rng = np.random.default_rng(seed)
    device = torch.device(device_str)
    owns_net = net is None
    if net is None:
        net = DualHeadNet(cfg).to(device)
        net.load_state_dict(network_state)
    net.eval()

    mcts = MCTS(cfg, net, device, rng)
    env = GomokuEnv(board_size=cfg.board_size)

    planes_list: List[np.ndarray] = []
    policies_list: List[np.ndarray] = []
    players_list: List[int] = []

    move_num = 0
    while env.result == GameResult.ONGOING:
        planes_list.append(env.get_observation())
        players_list.append(env.current_player)

        temp = 1.0 if move_num < cfg.temperature_moves else cfg.temp_threshold
        action_probs, _ = mcts.run(
            env,
            add_root_noise=(move_num == 0),
            temperature=temp,
        )
        policies_list.append(action_probs)

        if temp < cfg.temp_threshold:
            action = int(np.argmax(action_probs))
        else:
            legal = env.legal_moves_mask()
            p = action_probs.copy()
            p[~legal] = 0
            p = p / (p.sum() + 1e-8)
            action = int(rng.choice(len(p), p=p))

        env.step(action)
        move_num += 1

    if env.result == GameResult.ONGOING:
        return None  # incomplete — filtered out

    outcome = env.outcome_value(BLACK)
    result = {
        "planes": planes_list,
        "policies": policies_list,
        "players": players_list,
        "outcome": outcome,
        "verified": True,
    }
    if owns_net:
        del net
    return result


def _actor_loop_shared_net(
    actor_id: int,
    cfg: TrainConfig,
    game_queue: queue.Queue,
    weight_queue: queue.Queue,
    stop_event: threading.Event,
    shared_net: DualHeadNet,
    net_lock: threading.Lock,
    device: torch.device,
) -> None:
    """Colab actor: reuse one GPU network, sync weights via lock."""
    rng = np.random.default_rng(cfg.seed + actor_id)
    game_count = 0
    device_str = str(device)

    while not stop_event.is_set():
        try:
            while True:
                state = weight_queue.get_nowait()
                with net_lock:
                    shared_net.load_state_dict(state)
        except queue.Empty:
            pass

        seed = int(rng.integers(0, 2**31 - 1))
        try:
            with net_lock:
                shared_net.eval()
                traj = _play_game(cfg, {}, device_str, seed, net=shared_net)
            if traj is not None:
                game_queue.put(traj)
                game_count += 1
        except Exception as exc:
            logger.exception("Shared actor %d failed: %s", actor_id, exc)

        if game_count >= cfg.games_per_actor:
            break


def _actor_loop_thread(
    actor_id: int,
    cfg: TrainConfig,
    game_queue: queue.Queue,
    weight_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Thread-based actor (lower memory than spawn on CPU-only machines)."""
    device_str = cfg.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed + actor_id)
    network_state = None
    game_count = 0

    while not stop_event.is_set():
        try:
            while True:
                network_state = weight_queue.get_nowait()
        except queue.Empty:
            pass

        if network_state is None:
            time.sleep(0.1)
            continue

        seed = int(rng.integers(0, 2**31 - 1))
        try:
            traj = _play_game(cfg, network_state, device_str, seed)
            if traj is not None:
                game_queue.put(traj)
                game_count += 1
        except Exception as exc:
            logger.exception("Actor %d failed: %s", actor_id, exc)

        if game_count >= cfg.games_per_actor:
            break


def _actor_loop(
    actor_id: int,
    cfg: TrainConfig,
    game_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Continuously generate self-play games and push to queue."""
    device_str = cfg.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed + actor_id)
    network_state = None
    game_count = 0

    while not stop_event.is_set():
        # Sync weights from learner
        try:
            while True:
                network_state = weight_queue.get_nowait()
        except queue.Empty:
            pass

        if network_state is None:
            time.sleep(0.1)
            continue

        seed = int(rng.integers(0, 2**31 - 1))
        try:
            traj = _play_game(cfg, network_state, device_str, seed)
            if traj is not None:
                game_queue.put(traj)
                game_count += 1
        except Exception as exc:
            logger.exception("Actor %d failed: %s", actor_id, exc)

        if game_count >= cfg.games_per_actor:
            break


# ------------------------------------------------------------------ TrainPipeline
@dataclass
class TrainPipeline:
    """
    Orchestrates actor processes, replay buffer, learner, and evaluation.

    Usage:
        pipeline = TrainPipeline(cfg)
        pipeline.run()
    """

    cfg: TrainConfig
    _device: torch.device = field(init=False)
    _net: DualHeadNet = field(init=False)
    _baseline: DualHeadNet = field(init=False)
    _buffer: PrioritizedReplayBuffer = field(init=False)
    _optimizer: torch.optim.Optimizer = field(init=False)
    _scheduler: MultiStepLR = field(init=False)
    _step: int = field(init=False, default=0)
    _stop_event: mp.Event = field(init=False)

    def __post_init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True,
        )
        if not isinstance(self.cfg.checkpoint_dir, Path):
            self.cfg.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.set_num_threads(1)

        self._net = build_network(self.cfg, self._device)
        self._baseline = build_network(self.cfg, self._device)
        self._baseline.load_state_dict(self._unwrap(self._net).state_dict())
        self._buffer = PrioritizedReplayBuffer(self.cfg)
        self._optimizer = AdamW(
            self._net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self._scheduler = MultiStepLR(
            self._optimizer,
            milestones=list(self.cfg.lr_milestones),
            gamma=self.cfg.lr_gamma,
        )
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ public
    def run(self) -> None:
        use_threads = self._device.type == "cpu" or self.cfg.force_thread_actors
        game_queue: queue.Queue = queue.Queue(maxsize=256)
        weight_queue: queue.Queue = queue.Queue(maxsize=4)
        thread_stop = threading.Event()
        net_lock = threading.Lock()
        shared_actor_net = (
            self._unwrap(self._net) if self.cfg.shared_actor_net and use_threads else None
        )

        state = self._net_state_cpu()
        for _ in range(self.cfg.num_actors):
            weight_queue.put(state)

        actors: list[threading.Thread | mp.Process] = []
        if use_threads:
            for i in range(self.cfg.num_actors):
                if shared_actor_net is not None:
                    target = _actor_loop_shared_net
                    args = (
                        i,
                        self.cfg,
                        game_queue,
                        weight_queue,
                        thread_stop,
                        shared_actor_net,
                        net_lock,
                        self._device,
                    )
                else:
                    target = _actor_loop_thread
                    args = (i, self.cfg, game_queue, weight_queue, thread_stop)
                t = threading.Thread(target=target, args=args, daemon=True)
                t.start()
                actors.append(t)
        else:
            ctx = mp.get_context("spawn")
            mp_game: mp.Queue = ctx.Queue(maxsize=256)
            mp_weight: mp.Queue = ctx.Queue(maxsize=4)
            for _ in range(self.cfg.num_actors):
                mp_weight.put(state)
            game_queue = mp_game  # type: ignore[assignment]
            weight_queue = mp_weight  # type: ignore[assignment]
            for i in range(self.cfg.num_actors):
                p = ctx.Process(
                    target=_actor_loop,
                    args=(i, self.cfg, mp_game, mp_weight, self._stop_event),
                    daemon=True,
                )
                p.start()
                actors.append(p)

        collector = threading.Thread(target=self._collect_games, args=(game_queue,), daemon=True)
        collector.start()

        logger.info(
            "Training on %s | actors=%d (%s%s) | buffer=%d | sims=%d",
            self._device,
            self.cfg.num_actors,
            "threads" if use_threads else "processes",
            ", shared-net" if shared_actor_net is not None else "",
            self.cfg.buffer_capacity,
            self.cfg.num_simulations,
        )

        try:
            while self._step < self.cfg.train_steps:
                if len(self._buffer) < self.cfg.min_buffer_size:
                    time.sleep(0.2)
                    if self._step % self.cfg.sync_interval == 0:
                        try:
                            weight_queue.put(self._net_state_cpu(), block=False)
                        except queue.Full:
                            pass
                    continue

                loss = self._train_step()
                if self._step % 100 == 0:
                    logger.info(
                        "step=%d loss=%.4f buffer=%d",
                        self._step,
                        loss,
                        len(self._buffer),
                    )

                if self._step % self.cfg.sync_interval == 0:
                    try:
                        weight_queue.put(self._net_state_cpu(), block=False)
                    except queue.Full:
                        pass

                if self._step % self.cfg.save_interval == 0 and self._step > 0:
                    self._save_checkpoint("latest.pt")

                if self._step % self.cfg.eval_interval == 0 and self._step > 0:
                    self._evaluate_and_maybe_promote()

                self._step += 1
        finally:
            self._stop_event.set()
            thread_stop.set()
            for a in actors:
                a.join(timeout=5)
            self._save_checkpoint("final.pt")
            logger.info("Training finished at step %d.", self._step)

    # ------------------------------------------------------------------ internal
    def _collect_games(self, game_queue: queue.Queue) -> None:
        while not self._stop_event.is_set():
            try:
                traj = game_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            n = self._buffer.add_game(
                traj["planes"],
                traj["policies"],
                traj["players"],
                traj["outcome"],
                verified=traj["verified"],
            )
            if n > 0:
                logger.info("Buffered %d positions (total=%d)", n, len(self._buffer))

    def _train_step(self) -> float:
        self._net.train()
        planes, target_pi, target_v, indices, weights = self._buffer.sample()

        # Data augmentation: random symmetry per sample
        if self.cfg.use_symmetry_augment:
            planes, target_pi = self._augment_batch(planes, target_pi)

        x = torch.from_numpy(planes).to(self._device)
        target_pi_t = torch.from_numpy(target_pi).to(self._device)
        target_v_t = torch.from_numpy(target_v).to(self._device)
        w_t = torch.from_numpy(weights).to(self._device)

        logits, value = self._net(x)
        log_probs = F.log_softmax(logits, dim=-1)

        # Policy: cross-entropy with MCTS target distribution
        policy_loss = -(target_pi_t * log_probs).sum(dim=-1)
        policy_loss = (policy_loss * w_t).mean()

        # Value: MSE to game outcome z
        value_loss = F.mse_loss(value, target_v_t, reduction="none").squeeze(-1)
        value_loss = (value_loss * w_t).mean()

        loss = (
            self.cfg.policy_loss_weight * policy_loss
            + self.cfg.value_loss_weight * value_loss
        )

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.cfg.grad_clip)
        self._optimizer.step()
        self._scheduler.step()
        self._buffer.anneal_beta()

        # Priority update from value TD error
        with torch.no_grad():
            td = (value - target_v_t).squeeze(-1).cpu().numpy()
        self._buffer.update_priorities(indices, td)

        return float(loss.item())

    def _augment_batch(
        self,
        planes: np.ndarray,
        policies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.cfg.board_size
        for i in range(len(planes)):
            sid = sym.random_symmetry_id(np.random.default_rng())
            planes[i] = sym.transform_board(planes[i], sid, size)
            policies[i] = sym.transform_policy(policies[i], sid, size)
        return planes, policies

    def _evaluate_and_maybe_promote(self) -> None:
        evaluator = EloEvaluator(self.cfg, self._device)
        self._net.eval()
        result = evaluator.evaluate(self._net, self._baseline)
        logger.info(
            "Eval: candidate %d - %d baseline (draws=%d) win_rate=%.1f%% promoted=%s",
            result.candidate_wins,
            result.baseline_wins,
            result.draws,
            result.win_rate * 100,
            result.promoted,
        )
        if result.promoted:
            self._unwrap(self._baseline).load_state_dict(self._unwrap(self._net).state_dict())
            self._save_checkpoint("best.pt")
            logger.info("Baseline promoted.")

    def _unwrap(self, net: DualHeadNet) -> DualHeadNet:
        return net.module if hasattr(net, "module") else net  # DataParallel

    def _net_state_cpu(self) -> dict:
        return {k: v.cpu() for k, v in self._unwrap(self._net).state_dict().items()}

    def _save_checkpoint(self, name: str) -> None:
        path = self.cfg.checkpoint_dir / name
        torch.save(
            {
                "step": self._step,
                "model": self._unwrap(self._net).state_dict(),
                "baseline": self._unwrap(self._baseline).state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
        if self.cfg.save_to_drive:
            self._mirror_to_drive(name)

    def _mirror_to_drive(self, name: str) -> None:
        import shutil

        drive_dir = Path(self.cfg.drive_checkpoint_dir)
        try:
            drive_dir.mkdir(parents=True, exist_ok=True)
            src = self.cfg.checkpoint_dir / name
            dst = drive_dir / name
            shutil.copy2(src, dst)
            logger.info("Mirrored checkpoint to Drive: %s", dst)
        except Exception as exc:
            logger.warning("Drive mirror failed: %s", exc)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._unwrap(self._net).load_state_dict(ckpt["model"])
        self._unwrap(self._baseline).load_state_dict(ckpt.get("baseline", ckpt["model"]))
        if "optimizer" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
        logger.info("Loaded checkpoint from %s (step=%d)", path, self._step)
