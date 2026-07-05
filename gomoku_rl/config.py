"""Training and inference hyper-parameters for AlphaZero-style Gomoku."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Central configuration; all tensor shapes derive from board_size."""

    board_size: int = 15
    num_planes: int = 5  # self, opp, last, second_last, color_to_move

    # ResNet backbone
    num_res_blocks: int = 10
    num_channels: int = 128
    use_swish: bool = True

    # MCTS
    c_puct: float = 1.5
    num_simulations: int = 800
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    mcts_threads: int = 1
    use_symmetry_augment: bool = True
    mcts_use_symmetry: bool = False  # 8x slower; enable on GPU for stronger search

    # Self-play
    num_actors: int = 4
    games_per_actor: int = 100_000
    temperature_moves: int = 15  # sample from visit counts for first N moves
    temp_threshold: float = 1e-3

    # Replay buffer (prioritized)
    buffer_capacity: int = 500_000
    batch_size: int = 512
    min_buffer_size: int = 1000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 1e-4

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_milestones: tuple = (100_000, 200_000, 300_000)
    lr_gamma: float = 0.1
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip: float = 1.0

    # Training loop
    train_steps: int = 500_000
    save_interval: int = 5_000
    eval_interval: int = 10_000
    sync_interval: int = 100  # learner -> actors weight sync
    train_interval: float = 0.01  # seconds between optimizer steps when buffer ready

    # Model promotion (Elo gate)
    eval_games: int = 40
    promotion_win_rate: float = 0.55
    eval_mcts_simulations: int = 400

    # Paths & device
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: str = "cuda"
    num_gpus: int = 0  # 0 = auto (all visible CUDA devices for DataParallel)
    seed: int = 42
    compile_model: bool = False  # torch.compile / TorchScript when supported

    # Colab / Jupyter
    force_thread_actors: bool = False  # required in Colab (no fork/spawn in notebook)
    shared_actor_net: bool = False  # reuse one GPU net across actor threads
    colab_mode: bool = False

    # Google Drive checkpoint mirror (Colab)
    save_to_drive: bool = False
    drive_checkpoint_dir: str = "/content/drive/MyDrive/gomoku_rl/checkpoints"

    @property
    def board_area(self) -> int:
        return self.board_size * self.board_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """(C, H, W) for DualHeadNet."""
        return (self.num_planes, self.board_size, self.board_size)

    @classmethod
    def colab_preset(cls, **overrides: object) -> "TrainConfig":
        """GPU-friendly defaults for Google Colab (T4 / L4)."""
        base = cls(
            device="cuda",
            num_res_blocks=10,
            num_channels=128,
            num_simulations=400,
            mcts_threads=2,
            mcts_use_symmetry=False,
            num_actors=2,
            batch_size=256,
            min_buffer_size=2000,
            train_steps=50_000,
            save_interval=1000,
            eval_interval=5000,
            eval_games=20,
            eval_mcts_simulations=200,
            force_thread_actors=True,
            shared_actor_net=True,
            colab_mode=True,
            checkpoint_dir=Path("/content/gomoku_rl/checkpoints"),
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base
