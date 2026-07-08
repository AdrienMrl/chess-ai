"""Ties eval + search together.

Prefers the NNUE eval when available and working; always falls back to the
classical eval otherwise (missing weights, missing numpy, or any runtime
exception), so training/integrating the NN can never break the engine.
"""
from ai import eval as classical_eval
from ai.search import search_root

try:
    from ai.nnue import nnue_eval
except Exception:
    nnue_eval = None


class Engine:
    def __init__(self, time_budget, use_nnue):
        """use_nnue: "auto" (use if available), "on" (force, error if unavailable),
        "off" (force classical)."""
        if use_nnue not in ("auto", "on", "off"):
            raise ValueError(f"use_nnue must be 'auto', 'on', or 'off', got {use_nnue!r}")
        self.time_budget = time_budget
        self.tt = {}
        self.use_nnue = use_nnue
        self._nnue_ready = False
        if use_nnue in ("auto", "on") and nnue_eval is not None:
            nnue_eval.load()
            self._nnue_ready = nnue_eval.available()
        if use_nnue == "on" and not self._nnue_ready:
            raise RuntimeError("NNUE eval forced on (--eval nnue) but weights are not available")

    def eval_fn(self, board):
        if self.use_nnue != "off" and self._nnue_ready:
            try:
                return nnue_eval.evaluate(board)
            except Exception:
                return classical_eval.evaluate(board)
        return classical_eval.evaluate(board)

    def pick_move(self, board):
        """Returns (move, score, depth_reached). score is centipawns from the
        side-to-move's perspective."""
        return search_root(board, self.time_budget, self.eval_fn, tt=self.tt)
