"""
Gomoku (15x15) environment with professional Renju forbidden-move rules for Black.

Tensor conventions
------------------
board : int8 (H, W) with {0=empty, 1=black, 2=white}
planes: float32 (C, H, W) fed to the network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))  # E, S, SE, NE


class GameResult(IntEnum):
    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class GomokuEnv:
    """
    Optimized Gomoku simulator.

    Renju (禁手) for Black only:
      - Overline: >= 6 consecutive black stones in any direction.
      - Four-Four: move creates two or more "four" threats simultaneously.
      - Three-Three: move creates two or more open-three threats simultaneously.
    White has no forbidden moves.
    """

    board_size: int = 15
    renju: bool = True
    _board: np.ndarray = field(init=False, repr=False)
    _current: int = field(init=False, default=BLACK)
    _move_history: List[Tuple[int, int]] = field(init=False, default_factory=list)
    _result: GameResult = field(init=False, default=GameResult.ONGOING)

    def __post_init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> np.ndarray:
        self._board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self._current = BLACK
        self._move_history.clear()
        self._result = GameResult.ONGOING
        return self.get_observation()

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def current_player(self) -> int:
        return self._current

    @property
    def result(self) -> GameResult:
        return self._result

    @property
    def move_count(self) -> int:
        return len(self._move_history)

    def clone(self) -> "GomokuEnv":
        env = GomokuEnv(board_size=self.board_size, renju=self.renju)
        env._board = self._board.copy()
        env._current = self._current
        env._move_history = list(self._move_history)
        env._result = self._result
        return env

    # ------------------------------------------------------------------ obs
    def get_observation(self) -> np.ndarray:
        """
        Planes (5, H, W):
          0: stones of player to move
          1: opponent stones
          2: last move marker (1 at last intersection)
          3: second-to-last move marker
          4: color plane (1.0 if black to move else 0.0)
        """
        me = self._current
        opp = WHITE if me == BLACK else BLACK
        planes = np.zeros((5, self.board_size, self.board_size), dtype=np.float32)
        planes[0] = (self._board == me).astype(np.float32)
        planes[1] = (self._board == opp).astype(np.float32)
        if self._move_history:
            r, c = self._move_history[-1]
            planes[2, r, c] = 1.0
        if len(self._move_history) >= 2:
            r, c = self._move_history[-2]
            planes[3, r, c] = 1.0
        planes[4].fill(1.0 if me == BLACK else 0.0)
        return planes

    # ------------------------------------------------------------------ moves
    def legal_moves_mask(self) -> np.ndarray:
        """Boolean mask (225,) — False where move is illegal."""
        mask = (self._board.reshape(-1) == EMPTY)
        if self._result != GameResult.ONGOING:
            return np.zeros(self.board_size * self.board_size, dtype=bool)
        if self.renju and self._current == BLACK:
            for idx in np.flatnonzero(mask):
                r, c = divmod(int(idx), self.board_size)
                if self._is_renju_foul(r, c):
                    mask[idx] = False
        return mask

    def action_to_rc(self, action: int) -> Tuple[int, int]:
        return divmod(action, self.board_size)

    def rc_to_action(self, r: int, c: int) -> int:
        return r * self.board_size + c

    def step(self, action: int) -> StepResult:
        if self._result != GameResult.ONGOING:
            raise RuntimeError("Game already finished.")

        r, c = self.action_to_rc(action)
        if self._board[r, c] != EMPTY:
            raise ValueError(f"Illegal occupied move ({r}, {c}).")

        reward = 0.0
        terminated = False
        info: dict = {}

        # Renju foul: immediate loss for black
        if self.renju and self._current == BLACK and self._is_renju_foul(r, c):
            self._board[r, c] = BLACK  # record foul stone for visualization
            self._move_history.append((r, c))
            self._result = GameResult.WHITE_WIN
            reward = -1.0
            terminated = True
            info["foul"] = True
            return StepResult(self.get_observation(), reward, terminated, False, info)

        self._board[r, c] = self._current
        self._move_history.append((r, c))

        if self._check_win(r, c, self._current):
            self._result = GameResult.BLACK_WIN if self._current == BLACK else GameResult.WHITE_WIN
            reward = 1.0
            terminated = True
        elif not np.any(self._board == EMPTY):
            self._result = GameResult.DRAW
            terminated = True
        else:
            self._current = WHITE if self._current == BLACK else BLACK

        return StepResult(self.get_observation(), reward, terminated, False, info)

    # ------------------------------------------------------------------ win
    def _check_win(self, r: int, c: int, color: int) -> bool:
        """Five-or-more in a row through (r, c)."""
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self._board[nr, nc] != color:
                        break
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= 5:
                return True
        return False

    # ------------------------------------------------------------------ Renju
    def _is_renju_foul(self, r: int, c: int) -> bool:
        """Detect overline, double-four, double-three for hypothetical black move."""
        if self._board[r, c] != EMPTY:
            return False

        self._board[r, c] = BLACK
        try:
            if self._renju_overline(r, c):
                return True
            fours = self._count_fours(r, c)
            if fours >= 2:
                return True
            open_threes = self._count_open_threes(r, c)
            if open_threes >= 2:
                return True
            return False
        finally:
            self._board[r, c] = EMPTY

    def _line_segment(self, r: int, c: int, dr: int, dc: int, radius: int = 5) -> str:
        """
        Encode line through (r,c) as a string for pattern matching.
        'X' black, 'O' white/other, '.' empty, '#' off-board (treated as blocked).
        """
        chars: List[str] = []
        for k in range(-radius, radius + 1):
            nr, nc = r + k * dr, c + k * dc
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                chars.append("#")
            elif self._board[nr, nc] == BLACK:
                chars.append("X")
            elif self._board[nr, nc] == WHITE:
                chars.append("O")
            else:
                chars.append(".")
        return "".join(chars)

    def _renju_overline(self, r: int, c: int) -> bool:
        for dr, dc in DIRS:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self._board[nr, nc] != BLACK:
                        break
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= 6:
                return True
        return False

    def _count_fours(self, r: int, c: int) -> int:
        """
        Count distinct four-threat lines created through (r,c).
        Includes open four (.XXXX.) and closed four (#XXXX. / .XXXX#).
        """
        total = 0
        for dr, dc in DIRS:
            line = self._line_segment(r, c, dr, dc, radius=5)
            center = 5  # index of (r,c) in 11-char string
            total += self._fours_in_line(line, center)
        return total

    @staticmethod
    def _fours_in_line(line: str, center: int) -> int:
        """Count four-patterns on one axis where center index participates."""
        count = 0
        n = len(line)
        # Scan windows of length 5 that include `center` and contain exactly 4 X
        for start in range(max(0, center - 4), min(n - 4, center) + 1):
            window = line[start : start + 5]
            if window.count("X") != 4:
                continue
            if "O" in window:
                continue
            # Must be a threatening four: at least one adjacent empty outside window
            left = start - 1
            right = start + 5
            left_open = left >= 0 and line[left] == "."
            right_open = right < n and line[right] == "."
            if left_open or right_open:
                # Center stone must belong to the four-run
                rel = center - start
                if 0 <= rel < 5 and window[rel] == "X":
                    count += 1
        return min(count, 1)  # at most one four per direction

    def _count_open_threes(self, r: int, c: int) -> int:
        total = 0
        for dr, dc in DIRS:
            line = self._line_segment(r, c, dr, dc, radius=4)
            center = 4
            total += self._open_threes_in_line(line, center)
        return total

    @staticmethod
    def _open_threes_in_line(line: str, center: int) -> int:
        """
        Open three: .XXX. pattern (both ends empty) that becomes open four next move.
        Also handles embedded patterns like .XX.X. with center filling gap.
        """
        count = 0
        n = len(line)
        patterns = (
            ".XXX.",   # straight open three
            ".XX.X.",  # jump three (center is third X)
            ".X.XX.",  # jump three (center is second X)
        )
        for pat in patterns:
            plen = len(pat)
            for start in range(max(0, center - plen + 1), min(n - plen + 1, center + 1)):
                if line[start : start + plen] != pat:
                    continue
                rel = center - start
                if pat[rel] != "X":
                    continue
                count += 1
                break  # one open-three per direction
        return min(count, 1)

    def outcome_value(self, player: int) -> float:
        """Final value from `player` perspective in [-1, 1]."""
        if self._result == GameResult.DRAW:
            return 0.0
        if self._result == GameResult.ONGOING:
            return 0.0
        winner = BLACK if self._result == GameResult.BLACK_WIN else WHITE
        return 1.0 if winner == player else -1.0
