"""
Dual-head ResNet policy/value network for Gomoku.

Architecture mirrors AlphaZero with BatchNorm + Swish activations for
TorchScript / TensorRT compatibility (eval mode, fused BN at export).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig


def _activation(use_swish: bool) -> nn.Module:
    # inplace=False avoids autograd errors with BatchNorm during backward
    return nn.SiLU(inplace=False) if use_swish else nn.ReLU(inplace=False)


class ResidualBlock(nn.Module):
    """Pre-activation style residual block: BN -> Act -> Conv -> BN -> Act -> Conv."""

    def __init__(self, channels: int, use_swish: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = _activation(use_swish)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class DualHeadNet(nn.Module):
    """
    Input  (B, C, H, W)  with C=5 planes, H=W=15.
    Policy (B, H*W)      log-probabilities over intersections (masked externally).
    Value  (B, 1)        tanh-bounded scalar in [-1, 1].
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.num_channels
        use_swish = cfg.use_swish

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.num_planes, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            _activation(use_swish),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(c, use_swish) for _ in range(cfg.num_res_blocks)]
        )

        # Policy head: (B, c, H, W) -> (B, H*W)
        self.policy_conv = nn.Conv2d(c, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cfg.board_size * cfg.board_size, cfg.board_area)

        # Value head: global pool -> MLP -> tanh
        self.value_conv = nn.Conv2d(c, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(cfg.board_size * cfg.board_size, c)
        self.value_fc2 = nn.Linear(c, 1)
        self.act = _activation(use_swish)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: float tensor (B, C, H, W) normalized {0,1}.
        Returns:
          policy_logits: (B, board_area)
          value:         (B, 1)
        """
        h = self.stem(x)
        h = self.blocks(h)

        # Policy head
        p = self.act(self.policy_bn(self.policy_conv(h)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head: v = tanh(W2 * act(W1 * flatten(conv(h))))
        v = self.act(self.value_bn(self.value_conv(h)))
        v = v.reshape(v.size(0), -1)
        v = self.act(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    @torch.inference_mode()
    def predict(
        self,
        planes: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper applying legal move mask to policy softmax.

        planes:     (B, C, H, W)
        legal_mask: (B, H*W) bool — True = legal
        Returns policy_probs (B, H*W), value (B, 1)
        """
        logits, value = self.forward(planes)
        logits = logits.masked_fill(~legal_mask, -1e9)
        policy = F.softmax(logits, dim=-1)
        policy = torch.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
        return policy, value

    def export_torchscript(self, example: torch.Tensor, path: str) -> None:
        """Trace for deployment (TensorRT / C++ runtime)."""
        self.eval()
        traced = torch.jit.trace(self, example)
        traced.save(path)


def build_network(cfg: TrainConfig, device: torch.device) -> DualHeadNet:
    net = DualHeadNet(cfg).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        n = cfg.num_gpus or torch.cuda.device_count()
        device_ids = list(range(min(n, torch.cuda.device_count())))
        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)  # type: ignore[assignment]
    if cfg.compile_model and hasattr(torch, "compile"):
        net = torch.compile(net)  # type: ignore[assignment]
    return net
