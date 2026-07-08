"""Classical evaluation function: material + PST + mobility + king safety + tempo.

evaluate(board) returns centipawns from White's perspective (positive = White
better). This is the guaranteed-working eval; ai/nnue/nnue_eval.py is an
optional, additive upgrade that falls back to this on any failure.
"""
from chess_game import (
    WHITE,
    BLACK,
    sq,
    file_of,
    rank_of,
    name_sq,
    KNIGHT_DELTAS,
    KING_DELTAS,
    BISHOP_DIRS,
    ROOK_DIRS,
)

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


def _pawn_mobility_count(board, f0, r0, color):
    direction = 1 if color == WHITE else -1
    start_rank = 1 if color == WHITE else 6
    promo_rank = 7 if color == WHITE else 0
    count = 0

    r1 = r0 + direction
    if 0 <= r1 < 8 and board.squares[sq(f0, r1)] == ".":
        count += 4 if r1 == promo_rank else 1
        if r0 == start_rank:
            r2 = r0 + 2 * direction
            if board.squares[sq(f0, r2)] == ".":
                count += 1

    for df in (-1, 1):
        f = f0 + df
        r = r0 + direction
        if 0 <= f < 8 and 0 <= r < 8:
            t = sq(f, r)
            tc = board.color_of(board.squares[t])
            if tc is not None and tc != color:
                count += 4 if r == promo_rank else 1
            elif board.ep is not None and t == board.ep:
                count += 1
    return count


def _castle_mobility_count(board, s, color):
    if board.in_check(color):
        return 0
    enemy = 1 - color
    count = 0
    if color == WHITE and s == name_sq("e1"):
        if "K" in board.castling and board.squares[name_sq("f1")] == "." and board.squares[name_sq("g1")] == ".":
            if not board.is_attacked(name_sq("f1"), enemy) and not board.is_attacked(name_sq("g1"), enemy):
                count += 1
        if "Q" in board.castling and board.squares[name_sq("d1")] == "." and board.squares[name_sq("c1")] == "." and board.squares[name_sq("b1")] == ".":
            if not board.is_attacked(name_sq("d1"), enemy) and not board.is_attacked(name_sq("c1"), enemy):
                count += 1
    elif color == BLACK and s == name_sq("e8"):
        if "k" in board.castling and board.squares[name_sq("f8")] == "." and board.squares[name_sq("g8")] == ".":
            if not board.is_attacked(name_sq("f8"), enemy) and not board.is_attacked(name_sq("g8"), enemy):
                count += 1
        if "q" in board.castling and board.squares[name_sq("d8")] == "." and board.squares[name_sq("c8")] == "." and board.squares[name_sq("b8")] == ".":
            if not board.is_attacked(name_sq("d8"), enemy) and not board.is_attacked(name_sq("c8"), enemy):
                count += 1
    return count


def _mobility_count(board, color):
    """Pseudo-legal move count for `color`, regardless of whose turn it is.

    Mirrors chess_game.Board.pseudo_legal_moves()'s move counts exactly (same
    per-square deltas, same 4-way promotion counting, same castle checks) but
    only counts destinations instead of allocating a Move object per one —
    this runs on nearly every node via evaluate(), so avoiding the allocation
    and list-building matters.
    """
    count = 0
    for s in range(64):
        c = board.squares[s]
        if c == "." or board.color_of(c) != color:
            continue
        p = c.upper()
        f0, r0 = file_of(s), rank_of(s)
        if p == "P":
            count += _pawn_mobility_count(board, f0, r0, color)
        elif p == "N":
            for df, dr in KNIGHT_DELTAS:
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    if board.color_of(board.squares[sq(f, r)]) != color:
                        count += 1
        elif p == "K":
            for df, dr in KING_DELTAS:
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    if board.color_of(board.squares[sq(f, r)]) != color:
                        count += 1
            count += _castle_mobility_count(board, s, color)
        else:
            dirs = []
            if p in ("B", "Q"):
                dirs += BISHOP_DIRS
            if p in ("R", "Q"):
                dirs += ROOK_DIRS
            for df, dr in dirs:
                f, r = f0 + df, r0 + dr
                while 0 <= f < 8 and 0 <= r < 8:
                    tc = board.color_of(board.squares[sq(f, r)])
                    if tc == color:
                        break
                    count += 1
                    if tc is not None:
                        break
                    f += df
                    r += dr
    return count


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

    white_mobility = _mobility_count(board, 0)
    black_mobility = _mobility_count(board, 1)
    score += MOBILITY_WEIGHT * (white_mobility - black_mobility)

    score += TEMPO_BONUS if board.turn == WHITE else -TEMPO_BONUS

    return score
