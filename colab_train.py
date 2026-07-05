#!/usr/bin/env python3
"""Colab / cloud GPU training entry point."""

from __future__ import annotations

import argparse
import logging

from gomoku_rl.colab import train_colab


def main() -> None:
    p = argparse.ArgumentParser(description="Train Gomoku on Colab GPU")
    p.add_argument("--train-steps", type=int, default=50_000)
    p.add_argument("--num-simulations", type=int, default=400)
    p.add_argument("--num-actors", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-buffer-size", type=int, default=2000)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=5000)
    p.add_argument("--no-drive", action="store_true", help="Skip Google Drive mount/save")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

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
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
