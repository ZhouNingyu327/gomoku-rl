# Gomoku RL — AlphaZero-style 五子棋 (Renju)

Production-grade **15×15 Gomoku** AI with **Renju forbidden-move rules** for Black, trained via **MCTS + dual-head ResNet** (AlphaZero-style).

## Features

- **GomokuEnv** — Renju 禁手 (三三 / 四四 / 长连) for Black
- **DualHeadNet** — ResNet policy + value heads
- **MCTS** — PUCT, Dirichlet root noise, 8-fold symmetry
- **TrainPipeline** — async self-play + prioritized replay + Elo promotion gate
- **Colab support** — GPU presets, Google Drive checkpoint sync
- **GUI** — local human vs AI (`play.py`)

## Quick Start (Local)

```bash
git clone https://github.com/ZhouNingyu327/gomoku-rl.git
cd gomoku-rl
pip install -e .

# Train (GPU)
python -m gomoku_rl.main --device cuda --num-actors 4 --num-simulations 120

# Play against AI
python play.py --checkpoint checkpoints/best.pt --sims 200
```

## Google Colab Training

### 一键训练（推荐）

```bash
# Colab 中 clone 后运行一个命令即可（含全部稳定性修复）
!python scripts/colab_setup.py --train-steps 5000

# 续训
!python scripts/colab_setup.py --train-steps 50000 --resume /content/gomoku_rl/checkpoints/final.pt
```

或打开 [`notebooks/train_gomoku_colab.ipynb`](notebooks/train_gomoku_colab.ipynb)，**只需运行 1 个代码 cell**。

### 内置修复（无需手动 patch）

| 修复 | 说明 |
|------|------|
| `sanitize_policy` | MCTS 策略 NaN 安全 |
| `temperature <= 1e-3` | 低温贪心，避免 overflow |
| `inplace=False` | BatchNorm backward 安全 |
| `actor_device=cpu` | 自对弈 CPU，训练 GPU，无 CUDA 冲突 |
| `120 sims × 4 actors` | 速度与吞吐平衡 |

### Colab preset 默认值

| Parameter | Value |
|-----------|-------|
| Learner | GPU (CUDA) |
| Actors | 4 线程，**CPU** 自对弈 |
| MCTS sims | 120 |
| Batch size | 256 |
| Checkpoints | 本地 + Google Drive |

Download `best.pt` from Drive and use locally:

```bash
python play.py --checkpoint path/to/best.pt
```

## Project Layout

```
gomoku_rl/
  env.py        # Gomoku + Renju simulator
  network.py    # DualHeadNet ResNet
  mcts.py       # MCTS + PUCT
  train.py      # TrainPipeline
  colab.py      # Colab / Drive helpers
  policy_utils.py  # NaN-safe policy normalization
  gui.py        # Tkinter human vs AI
  config.py     # TrainConfig + colab_preset()
notebooks/
  train_gomoku_colab.ipynb
scripts/
  colab_setup.py   # 一键 Colab 训练（推荐）
colab_train.py  # Colab CLI 入口
play.py         # Local GUI launcher
```

## Upload to GitHub

```bash
cd gomoku-rl
git init
git add .
git commit -m "Initial commit: AlphaZero Gomoku with Colab training"
git branch -M main
git remote add origin https://github.com/ZhouNingyu327/gomoku-rl.git
git push -u origin main
```

## Renju Rules (Black only)

| Forbidden | Description |
|-----------|-------------|
| 长连 Overline | 6+ consecutive stones |
| 四四 Double-four | Two four-threats in one move |
| 三三 Double-three | Two open-threes in one move |

White has no restrictions.

## License

MIT
