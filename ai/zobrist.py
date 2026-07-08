"""Zobrist hashing for transposition-table keys.

chess_game.Board has no built-in incremental hash, so this recomputes the
hash from board state each call (O(64)). Simple and correct; an incremental
version (updated from the Move before _make) is a future optimization if
profiling shows this is a bottleneck.
"""
import random

_rng = random.Random(20260708)

PIECE_CHARS = "PNBRQKpnbrqk"
PIECE_KEYS = {p: [_rng.getrandbits(64) for _ in range(64)] for p in PIECE_CHARS}
SIDE_KEY = _rng.getrandbits(64)
CASTLE_KEYS = {c: _rng.getrandbits(64) for c in "KQkq"}
EP_FILE_KEYS = [_rng.getrandbits(64) for _ in range(8)]


def hash_board(board):
    h = 0
    squares = board.squares
    for s in range(64):
        c = squares[s]
        if c != ".":
            h ^= PIECE_KEYS[c][s]
    if board.turn == 1:
        h ^= SIDE_KEY
    for c in board.castling:
        h ^= CASTLE_KEYS[c]
    if board.ep is not None:
        h ^= EP_FILE_KEYS[board.ep % 8]
    return h
