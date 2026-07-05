"""
Google Colab helpers: Drive mount, preset config, training launcher.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import TrainConfig
from .train import TrainPipeline

logger = logging.getLogger(__name__)


def mount_google_drive() -> None:
    """Mount Google Drive in Colab (no-op outside Colab)."""
    try:
        from google.colab import drive  # type: ignore[import-untyped]

        drive.mount("/content/drive")
        logger.info("Google Drive mounted at /content/drive")
    except ImportError:
        logger.warning("google.colab not available — skipping Drive mount.")


def setup_colab(
    mount_drive: bool = True,
    save_to_drive: bool = True,
    **cfg_overrides: object,
) -> TrainConfig:
    """
    Build Colab training config and optionally mount Drive.

    Checkpoints:
      - local:  /content/gomoku_rl/checkpoints/
      - drive:  /content/drive/MyDrive/gomoku_rl/checkpoints/  (if enabled)
    """
    if mount_drive:
        mount_google_drive()

    cfg = TrainConfig.colab_preset(
        save_to_drive=save_to_drive,
        **cfg_overrides,
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def train_colab(
    mount_drive: bool = True,
    save_to_drive: bool = True,
    resume: Optional[str | Path] = None,
    **cfg_overrides: object,
) -> TrainPipeline:
    """One-call Colab training entry point."""
    cfg = setup_colab(
        mount_drive=mount_drive,
        save_to_drive=save_to_drive,
        **cfg_overrides,
    )
    pipeline = TrainPipeline(cfg)
    if resume:
        pipeline.load_checkpoint(Path(resume))
    logging.basicConfig(level=logging.INFO)
    logger.info("Colab training | GPU=%s | steps=%d", cfg.device, cfg.train_steps)
    pipeline.run()
    return pipeline
