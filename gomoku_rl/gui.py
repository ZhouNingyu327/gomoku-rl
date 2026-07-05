"""
Tkinter GUI — play Gomoku (Renju) against the trained AI.

Usage:
    python -m gomoku_rl.gui
    python -m gomoku_rl.gui --checkpoint checkpoints/best.pt --sims 200
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
import torch

from .config import TrainConfig
from .env import BLACK, WHITE, GomokuEnv, GameResult
from .inference import load_model, resolve_checkpoint
from .mcts import MCTS


CELL = 36
MARGIN = 28
STONE_R = 14


class GomokuGUI:
    def __init__(
        self,
        cfg: TrainConfig,
        net,
        device: torch.device,
        human_color: int = BLACK,
        mcts_sims: int = 200,
    ) -> None:
        self.cfg = cfg
        self.net = net
        self.device = device
        self.human_color = human_color
        self.mcts_sims = mcts_sims

        self.env = GomokuEnv(board_size=cfg.board_size)
        self.rng = np.random.default_rng()
        self.ai_thinking = False

        size = cfg.board_size
        canvas_size = MARGIN * 2 + CELL * (size - 1)

        self.root = tk.Tk()
        self.root.title("五子棋 Renju — Human vs AI")
        self.root.resizable(False, False)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        color_name = "黑棋 (先手, 禁手)" if human_color == BLACK else "白棋"
        self.status_var = tk.StringVar(
            value=f"你是{color_name} | 点击交叉点落子"
        )
        ttk.Label(top, textvariable=self.status_var, font=("Microsoft YaHei UI", 11)).pack(
            side=tk.LEFT
        )

        ttk.Button(top, text="新对局", command=self.new_game).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="悔棋", command=self.undo_move).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            bg="#DCB35C",
            highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.on_click)

        self._history: list[tuple[int, int]] = []
        self._draw_board()
        self._refresh_stones()

        if self.env.current_player != self.human_color:
            self.root.after(400, self._ai_move)

    # ------------------------------------------------------------------ drawing
    def _draw_board(self) -> None:
        self.canvas.delete("grid")
        n = self.cfg.board_size
        for i in range(n):
            x0 = MARGIN + i * CELL
            y0 = MARGIN
            x1 = MARGIN + i * CELL
            y1 = MARGIN + (n - 1) * CELL
            self.canvas.create_line(x0, y0, x1, y1, fill="#4A3728", tags="grid")
            self.canvas.create_line(y0, x0, y1, x1, fill="#4A3728", tags="grid")

        # Star points (hoshi) on 15x15
        if n == 15:
            stars = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
            for r, c in stars:
                x, y = self._rc_to_xy(r, c)
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3, fill="#4A3728", outline="", tags="grid"
                )

    def _rc_to_xy(self, r: int, c: int) -> tuple[int, int]:
        return MARGIN + c * CELL, MARGIN + r * CELL

    def _xy_to_rc(self, x: int, y: int) -> tuple[int, int]:
        c = round((x - MARGIN) / CELL)
        r = round((y - MARGIN) / CELL)
        n = self.cfg.board_size
        r = max(0, min(n - 1, r))
        c = max(0, min(n - 1, c))
        return r, c

    def _refresh_stones(self) -> None:
        self.canvas.delete("stone")
        board = self.env.board
        for r in range(self.cfg.board_size):
            for c in range(self.cfg.board_size):
                v = board[r, c]
                if v == 0:
                    continue
                x, y = self._rc_to_xy(r, c)
                fill = "#111111" if v == BLACK else "#F5F5F5"
                outline = "#333333" if v == BLACK else "#AAAAAA"
                self.canvas.create_oval(
                    x - STONE_R,
                    y - STONE_R,
                    x + STONE_R,
                    y + STONE_R,
                    fill=fill,
                    outline=outline,
                    width=2,
                    tags="stone",
                )

        if self._history:
            r, c = self._history[-1]
            x, y = self._rc_to_xy(r, c)
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4, fill="#E74C3C", outline="", tags="stone"
            )

    # ------------------------------------------------------------------ game logic
    def new_game(self) -> None:
        if self.ai_thinking:
            return
        self.env.reset()
        self._history.clear()
        self._refresh_stones()
        self.status_var.set("新对局开始 — 轮到你" if self.env.current_player == self.human_color else "AI 思考中…")
        if self.env.current_player != self.human_color:
            self.root.after(300, self._ai_move)

    def undo_move(self) -> None:
        if self.ai_thinking or not self._history:
            return
        # Undo AI + human (two moves) when human just played; one if only human moved first
        pops = 2 if len(self._history) >= 2 else 1
        self.env.reset()
        self._history = self._history[:-pops]
        for r, c in self._history:
            self.env.step(self.env.rc_to_action(r, c))
        self._refresh_stones()
        self.status_var.set("已悔棋 — 轮到你")

    def on_click(self, event: tk.Event) -> None:
        if self.ai_thinking or self.env.result != GameResult.ONGOING:
            return
        if self.env.current_player != self.human_color:
            return

        r, c = self._xy_to_rc(event.x, event.y)
        action = self.env.rc_to_action(r, c)
        legal = self.env.legal_moves_mask()
        if not legal[action]:
            if self.human_color == BLACK and self.env.board[r, c] == 0:
                if self.env._is_renju_foul(r, c):
                    messagebox.showwarning("禁手", "此着为禁手（三三 / 四四 / 长连），请另选位置。")
            else:
                messagebox.showinfo("无效", "该位置不能落子。")
            return

        self._apply_move(r, c)
        if self.env.result == GameResult.ONGOING:
            self._ai_move()

    def _apply_move(self, r: int, c: int) -> None:
        step = self.env.step(self.env.rc_to_action(r, c))
        self._history.append((r, c))
        self._refresh_stones()

        if step.info.get("foul"):
            messagebox.showinfo("禁手判负", "黑棋禁手，白棋胜。")
            self.status_var.set("禁手 — 你输了")
            return

        if self.env.result == GameResult.BLACK_WIN:
            msg = "你赢了！" if self.human_color == BLACK else "AI 赢了"
            self.status_var.set(f"黑棋胜 — {msg}")
            messagebox.showinfo("终局", f"黑棋五连 — {msg}")
        elif self.env.result == GameResult.WHITE_WIN:
            msg = "你赢了！" if self.human_color == WHITE else "AI 赢了"
            self.status_var.set(f"白棋胜 — {msg}")
            messagebox.showinfo("终局", f"白棋五连 — {msg}")
        elif self.env.result == GameResult.DRAW:
            self.status_var.set("和棋")
            messagebox.showinfo("终局", "棋盘已满 — 和棋")
        else:
            who = "你" if self.env.current_player == self.human_color else "AI"
            self.status_var.set(f"轮到了 {who}")

    def _ai_move(self) -> None:
        if self.env.result != GameResult.ONGOING:
            return
        if self.env.current_player == self.human_color:
            return

        self.ai_thinking = True
        self.status_var.set("AI 思考中…")

        def worker() -> None:
            play_cfg = TrainConfig(
                board_size=self.cfg.board_size,
                num_res_blocks=self.cfg.num_res_blocks,
                num_channels=self.cfg.num_channels,
                num_simulations=self.mcts_sims,
                mcts_threads=1,
                mcts_use_symmetry=False,
                c_puct=self.cfg.c_puct,
            )
            mcts = MCTS(play_cfg, self.net, self.device, self.rng)
            action = mcts.get_action(self.env, temperature=1e-3, add_root_noise=False)
            r, c = self.env.action_to_rc(action)

            def finish() -> None:
                self.ai_thinking = False
                if self.env.result == GameResult.ONGOING and self.env.current_player != self.human_color:
                    self._apply_move(r, c)
                    if self.env.result == GameResult.ONGOING and self.env.current_player != self.human_color:
                        self._ai_move()

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Gomoku against trained AI")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--human-color", choices=("black", "white"), default="black")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per AI move")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint(args.checkpoint_dir)
    if ckpt is None:
        raise SystemExit(
            "No checkpoint found. Train first:\n"
            "  python -m gomoku_rl.main --device cpu --num-simulations 100 --train-steps 5000"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, cfg, device = load_model(ckpt, device)
    human = BLACK if args.human_color == "black" else WHITE

    print(f"Loaded {ckpt} on {device} | MCTS sims={args.sims}")
    GomokuGUI(cfg, net, device, human_color=human, mcts_sims=args.sims).run()


if __name__ == "__main__":
    main()
