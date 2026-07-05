"""Load a trained checkpoint for inference / human play."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import TrainConfig
from .network import DualHeadNet


def resolve_checkpoint(checkpoint_dir: Path | str = "checkpoints") -> Optional[Path]:
    """Pick best available checkpoint: best > final > latest."""
    root = Path(checkpoint_dir)
    for name in ("best.pt", "latest.pt", "final.pt"):
        path = root / name
        if path.exists():
            return path
    return None


def load_model(
    checkpoint: Path | str,
    device: Optional[torch.device] = None,
    cfg: Optional[TrainConfig] = None,
) -> Tuple[DualHeadNet, TrainConfig, torch.device]:
    """Restore DualHeadNet weights from a training checkpoint."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = cfg or ckpt.get("config") or TrainConfig()
    if not isinstance(cfg, TrainConfig):
        cfg = TrainConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DualHeadNet(cfg).to(device)
    state = ckpt.get("model") or ckpt.get("baseline")
    if state is None:
        raise KeyError(f"No model weights in checkpoint: {path}")
    net.load_state_dict(state)
    net.eval()
    return net, cfg, device
