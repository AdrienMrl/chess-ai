"""NNUE-style evaluation: a small 768->256->1 MLP over one-hot piece-placement
features, trained offline by ai/nnue/train.py against Stockfish centipawn
labels (see ai/data/positions.jsonl).

Fixed contract expected by ai/engine.py:

    nnue_eval.load()          # called once; must never raise
    nnue_eval.available()     # -> bool, True only if a working net was loaded
    nnue_eval.evaluate(board) # -> int, centipawns from WHITE's perspective

Numpy is imported lazily, inside try/except ImportError, so this module
always imports cleanly even on a system python3 with no numpy installed;
in that case load() simply fails to find numpy and available() reports False.
"""
import os

PIECE_ORDER = "PNBRQKpnbrqk"
NUM_FEATURES = 768  # 12 piece types * 64 squares

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "weights.npz")

_state = {
    "attempted": False,
    "available": False,
    "W1": None,
    "b1": None,
    "W2": None,
    "b2": None,
}


def encode_board(board):
    """Encode a chess_game.Board into a 768-dim float32 one-hot feature vector.

    Index for a piece of type `p` (one of "PNBRQKpnbrqk") sitting on square
    `s` (0..63, a1=0, h8=63) is `PIECE_ORDER.index(p) * 64 + s`. This encoding
    is turn-agnostic (piece placement only); side-to-move is not part of the
    input, see ai/nnue/train.py for how labels are made White-relative to
    compensate.
    """
    import numpy as np

    x = np.zeros(NUM_FEATURES, dtype=np.float32)
    squares = board.squares
    for s in range(64):
        c = squares[s]
        if c == ".":
            continue
        idx = PIECE_ORDER.index(c)
        x[idx * 64 + s] = 1.0
    return x


def load():
    """Load weights from ai/data/weights.npz. Never raises. Safe to call
    multiple times (only actually loads once)."""
    if _state["attempted"]:
        return
    _state["attempted"] = True
    try:
        import numpy as np

        with np.load(_WEIGHTS_PATH) as data:
            W1 = data["W1"]
            b1 = data["b1"]
            W2 = data["W2"]
            b2 = data["b2"]
        # Basic sanity checks on shapes so a corrupt/mismatched file doesn't
        # silently produce garbage at inference time.
        if W1.shape[0] != NUM_FEATURES:
            raise ValueError("weights.npz has unexpected shapes")
        _state["W1"] = W1
        _state["b1"] = b1
        _state["W2"] = W2
        _state["b2"] = b2
        _state["available"] = True
    except Exception:
        _state["available"] = False
        _state["W1"] = None
        _state["b1"] = None
        _state["W2"] = None
        _state["b2"] = None


def available():
    """True only if a working net was loaded successfully."""
    if not _state["attempted"]:
        load()
    return _state["available"]


def evaluate(board):
    """Centipawn score from WHITE's perspective. Raises if no net is loaded
    (callers must check available() first, per contract)."""
    if not available():
        raise RuntimeError("nnue_eval.evaluate() called but no net is loaded")

    import numpy as np

    x = encode_board(board)
    W1, b1, W2, b2 = _state["W1"], _state["b1"], _state["W2"], _state["b2"]
    h = x @ W1 + b1
    h = np.maximum(h, 0.0)  # ReLU
    out = h @ W2 + b2
    score = float(np.asarray(out).reshape(-1)[0])
    if not np.isfinite(score):
        raise RuntimeError("nnue_eval.evaluate() produced a non-finite score")
    return int(round(score))
