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
git clone https://github.com/YOUR_USERNAME/gomoku-rl.git
cd gomoku-rl
pip install -e .

# Train (GPU)
python -m gomoku_rl.main --device cuda --num-actors 2 --num-simulations 400

# Play against AI
python play.py --checkpoint checkpoints/best.pt --sims 200
```

## Google Colab Training

### Option A — Notebook (recommended)

1. Upload repo to GitHub (see below)
2. Open [`notebooks/train_gomoku_colab.ipynb`](notebooks/train_gomoku_colab.ipynb) in Colab
3. Set runtime to **GPU** (T4 / L4)
4. Run all cells — checkpoints save to Drive automatically

### Option B — Script

```python
# In a Colab cell after cloning the repo:
!pip install -e /content/gomoku-rl

from gomoku_rl.colab import train_colab
train_colab(
    mount_drive=True,
    save_to_drive=True,
    train_steps=50_000,
    num_simulations=400,
)
```

Or from terminal in Colab:

```bash
python colab_train.py --train-steps 50000 --num-simulations 400
```

### Colab preset defaults

| Parameter | Value |
|-----------|-------|
| GPU | CUDA (auto) |
| MCTS sims | 400 |
| Actors | 2 (threaded, shared GPU net) |
| Batch size | 256 |
| Train steps | 50,000 |
| Checkpoints | `/content/gomoku_rl/checkpoints/` + Google Drive mirror |

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
  gui.py        # Tkinter human vs AI
  config.py     # TrainConfig + colab_preset()
notebooks/
  train_gomoku_colab.ipynb
colab_train.py  # Colab CLI entry
play.py         # Local GUI launcher
```

## Upload to GitHub

```bash
cd gomoku-rl
git init
git add .
git commit -m "Initial commit: AlphaZero Gomoku with Colab training"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gomoku-rl.git
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
