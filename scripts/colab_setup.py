#!/usr/bin/env python3
"""
One-shot Colab setup & training launcher.

All stability fixes are included in the main codebase:
  - NaN-safe MCTS policy (sanitize_policy, log-space temperature)
  - inplace=False activations (BatchNorm + backward safe)
  - CPU actors + GPU learner (no CUDA autograd conflicts)
  - Colab preset: 120 sims, 4 actors

Usage in Colab (after clone):
    !python scripts/colab_setup.py
    # or with options:
    !python scripts/colab_setup.py --train-steps 5000 --resume checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pip_install() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Colab one-shot Gomoku training")
    p.add_argument("--train-steps", type=int, default=50_000)
    p.add_argument("--num-simulations", type=int, default=120)
    p.add_argument("--num-actors", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-buffer-size", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-drive", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    _pip_install()

    from gomoku_rl.colab import train_colab

    train_colab(
        mount_drive=not args.no_drive,
        save_to_drive=not args.no_drive,
        resume=args.resume,
        train_steps=args.train_steps,
        num_simulations=args.num_simulations,
        num_actors=args.num_actors,
        batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        actor_device="cpu",
    )


if __name__ == "__main__":
    main()
