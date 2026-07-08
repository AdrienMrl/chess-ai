"""Classical evaluation function: material + PST + mobility + king safety + tempo.

evaluate(board) returns centipawns from White's perspective (positive = White
better). This is the guaranteed-working eval; ai/nnue/nnue_eval.py is an
optional, additive upgrade that falls back to this on any failure.
"""
from chess_game import WHITE, sq, file_of, rank_of

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

TEMPO_BONUS = 10
MOBILITY_WEIGHT = 4
BISHOP_PAIR_BONUS = 30
KING_SHIELD_PENALTY = 15


def _rows(*rows_rank8_to_rank1):
    """Build a 64-entry PST from 8 rows given rank8 first (as usually printed)."""
    table = [0] * 64
    for i, row in enumerate(rows_rank8_to_rank1):
        rank = 7 - i
        for file in range(8):
            table[rank * 8 + file] = row[file]
    return table


PAWN_PST = _rows(
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
)
KNIGHT_PST = _rows(
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
)
BISHOP_PST = _rows(
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
)
ROOK_PST = _rows(
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
)
QUEEN_PST = _rows(
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
)
KING_PST = _rows(
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20],
)

PST = {"P": PAWN_PST, "N": KNIGHT_PST, "B": BISHOP_PST, "R": ROOK_PST, "Q": QUEEN_PST, "K": KING_PST}


def _mirror(s):
    return sq(file_of(s), 7 - rank_of(s))


def _pseudo_moves_for(board, color):
    """Pseudo-legal move count for `color`, regardless of whose turn it is."""
    original_turn = board.turn
    board.turn = color
    try:
        return board.pseudo_legal_moves()
    finally:
        board.turn = original_turn


def _king_safety(board, color):
    king_sq = board.king_square(color)
    if king_sq is None:
        return 0
    kf, kr = file_of(king_sq), rank_of(king_sq)
    shield_rank = kr + 1 if color == WHITE else kr - 1
    if not (0 <= shield_rank < 8):
        return 0
    pawn_char = "P" if color == WHITE else "p"
    penalty = 0
    for df in (-1, 0, 1):
        f = kf + df
        if 0 <= f < 8 and board.squares[sq(f, shield_rank)] != pawn_char:
            penalty -= KING_SHIELD_PENALTY
    return penalty


def evaluate(board):
    """Centipawn score from White's perspective. Positive = White better."""
    score = 0
    white_bishops = 0
    black_bishops = 0

    for s in range(64):
        c = board.squares[s]
        if c == ".":
            continue
        p = c.upper()
        value = PIECE_VALUES[p]
        pst = PST[p]
        if c.isupper():
            score += value + pst[s]
            if p == "B":
                white_bishops += 1
        else:
            score -= value + pst[_mirror(s)]
            if p == "B":
                black_bishops += 1

    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS

    score += _king_safety(board, 0) - _king_safety(board, 1)

    white_mobility = len(_pseudo_moves_for(board, 0))
    black_mobility = len(_pseudo_moves_for(board, 1))
    score += MOBILITY_WEIGHT * (white_mobility - black_mobility)

    score += TEMPO_BONUS if board.turn == WHITE else -TEMPO_BONUS

    return score
