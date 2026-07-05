"""Production AlphaZero-style Gomoku (Renju) training entry point."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import TrainConfig
from .train import TrainPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Gomoku AI with AlphaZero-style RL")
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-actors", type=int, default=4)
    p.add_argument("--num-simulations", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=500_000)
    p.add_argument("--min-buffer-size", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=2000)
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        board_size=args.board_size,
        device=args.device,
        num_actors=args.num_actors,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        min_buffer_size=args.min_buffer_size,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        checkpoint_dir=Path(args.checkpoint_dir),
        compile_model=args.compile,
    )

    pipeline = TrainPipeline(cfg)
    if args.resume:
        pipeline.load_checkpoint(Path(args.resume))

    logging.info("Starting Gomoku Renju training pipeline.")
    pipeline.run()


if __name__ == "__main__":
    main()
