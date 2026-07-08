#!/usr/bin/env python3
"""
Standalone extra positional evaluation terms for the chess engine in engine.py.

This module does NOT import or modify engine.py. It only depends on the
chess_game.Board representation (board.squares: 64-char list, 'P'/'N'/'B'/'R'/
'Q'/'K' for White, lowercase for Black, '.' for empty; sq(file, rank),
file_of(s), rank_of(s) with file/rank in 0..7, a1=0).

All functions return a centipawn delta from White's perspective (positive is
good for White). They are meant to be summed into the main evaluation
function by hand later.

Performance note: everything funnels through a single O(64) scan
(`_gather`) that collects pawn files, rook/knight/bishop squares, and king
squares for both colors in one pass. The four scoring functions then work
off that small gathered data instead of each re-scanning the board.
"""

from chess_game import sq, file_of, rank_of, KNIGHT_DELTAS, BISHOP_DIRS

WHITE, BLACK = 0, 1

# ---- tunable constants ----
DOUBLED_PAWN_PENALTY = -18
ISOLATED_PAWN_PENALTY = -18
# Passed pawn bonus indexed by rank-from-promotion distance (0 = about to
# promote's rank-1, i.e. rank 6 for white / rank 1 for black .. up to rank 1
# for white / rank 6 for black). Indexed by "ranks advanced" (0..6), where
# 0 = pawn still on its own 2nd rank and 6 = pawn on the 7th rank (one step
# from queening).
PASSED_PAWN_BONUS = [0, 5, 10, 20, 35, 60, 90]

ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMI_OPEN_FILE_BONUS = 10
ROOK_SEVENTH_RANK_BONUS = 15

MOBILITY_WEIGHT = 4  # centipawns per reachable square, knights/bishops only

KING_SHIELD_PAWN_BONUS = 8

# Open/semi-open file penalty near the king: applied per-file (king's own
# file plus the two adjacent files) when there are no friendly pawns on that
# file, scaled by how many enemy heavy pieces (rooks/queens) remain -- an
# open lane matters a lot with a queen+rooks still on, barely at all with
# none left.
KING_FILE_OPEN_PENALTY = -14      # king's own file
KING_ADJ_FILE_OPEN_PENALTY = -8   # each adjacent file
HEAVY_PIECE_CAP = 3  # 2 rooks + 1 queen == full scale; more doesn't add more

# Local (standalone) piece values, used only to compute a cheap material
# "phase" estimate (1.0 = full opening material on board, 0.0 = bare
# endgame). This mirrors the tapered MG/EG blend engine.py already uses for
# the king PST, recomputed locally here so this module stays independent of
# engine.py. It replaces the old hardcoded "3 exact home squares" check --
# now ANY king square is evaluated for shield/open-file safety as long as
# meaningful enemy material remains, which is what a king on g7 (not one of
# the 3 old home squares) needs.
_PIECE_VAL_LOCAL = {"N": 320, "B": 330, "R": 500, "Q": 900}
_FULL_NONPAWN_MATERIAL = 2 * (
    2 * _PIECE_VAL_LOCAL["N"] + 2 * _PIECE_VAL_LOCAL["B"]
    + 2 * _PIECE_VAL_LOCAL["R"] + _PIECE_VAL_LOCAL["Q"]
)
# Below this phase, treat it as a true endgame where king safety is moot
# (the king should be centralizing/active instead -- engine.py's KING_EG_PST
# already rewards that) and skip king-safety scoring entirely.
KING_SAFETY_ENDGAME_PHASE_CUTOFF = 0.05


def _gather(board):
    """Single O(64) pass collecting everything the scorers need.

    Returns a dict with:
      pawn_files: {WHITE: [count per file 0..7], BLACK: [...]}
      pawn_squares: {WHITE: [squares], BLACK: [squares]}
      rooks: {WHITE: [squares], BLACK: [squares]}
      knights: {WHITE: [squares], BLACK: [squares]}
      bishops: {WHITE: [squares], BLACK: [squares]}
      king_sq: {WHITE: sq or None, BLACK: sq or None}
    """
    pawn_files = {WHITE: [0] * 8, BLACK: [0] * 8}
    pawn_squares = {WHITE: [], BLACK: []}
    rooks = {WHITE: [], BLACK: []}
    knights = {WHITE: [], BLACK: []}
    bishops = {WHITE: [], BLACK: []}
    queens = {WHITE: [], BLACK: []}
    king_sq = {WHITE: None, BLACK: None}

    squares = board.squares
    for s in range(64):
        c = squares[s]
        if c == ".":
            continue
        color = WHITE if c.isupper() else BLACK
        p = c.upper()
        if p == "P":
            pawn_files[color][file_of(s)] += 1
            pawn_squares[color].append(s)
        elif p == "R":
            rooks[color].append(s)
        elif p == "N":
            knights[color].append(s)
        elif p == "B":
            bishops[color].append(s)
        elif p == "Q":
            queens[color].append(s)
        elif p == "K":
            king_sq[color] = s

    return {
        "pawn_files": pawn_files,
        "pawn_squares": pawn_squares,
        "rooks": rooks,
        "knights": knights,
        "bishops": bishops,
        "queens": queens,
        "king_sq": king_sq,
    }


def pawn_structure_score(board, data=None):
    """Doubled / isolated / passed pawn scoring. White score minus Black score."""
    if data is None:
        data = _gather(board)

    pawn_files = data["pawn_files"]
    pawn_squares = data["pawn_squares"]

    score = 0

    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        files = pawn_files[color]
        other_files = pawn_files[BLACK if color == WHITE else WHITE]

        # Doubled pawns: penalty per extra pawn beyond the first on a file.
        for f in range(8):
            if files[f] > 1:
                score += sign * DOUBLED_PAWN_PENALTY * (files[f] - 1)

        # Isolated pawns: no friendly pawns on adjacent files.
        for f in range(8):
            if files[f] == 0:
                continue
            left = files[f - 1] if f > 0 else 0
            right = files[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                score += sign * ISOLATED_PAWN_PENALTY * files[f]

        # Passed pawns: no enemy pawns on same/adjacent files ahead of it
        # (toward promotion direction).
        for s in pawn_squares[color]:
            f = file_of(s)
            r = rank_of(s)
            blocked = False
            check_files = [f]
            if f > 0:
                check_files.append(f - 1)
            if f < 7:
                check_files.append(f + 1)
            # We only have per-file counts, not per-rank positions, for the
            # opponent's pawns here, so do a cheap direct check against the
            # opponent pawn squares list (small list, fast).
            for os_ in pawn_squares[BLACK if color == WHITE else WHITE]:
                if file_of(os_) not in check_files:
                    continue
                orank = rank_of(os_)
                if color == WHITE and orank > r:
                    blocked = True
                    break
                if color == BLACK and orank < r:
                    blocked = True
                    break
            if blocked:
                continue
            ranks_advanced = r if color == WHITE else (7 - r)
            # ranks_advanced: 1 (on starting rank 1/6) .. 6 (on 7th rank).
            idx = max(0, min(6, ranks_advanced))
            score += sign * PASSED_PAWN_BONUS[idx]

    return score


def rook_activity_score(board, data=None):
    """Open/semi-open file bonus and 7th-rank bonus for rooks."""
    if data is None:
        data = _gather(board)

    pawn_files = data["pawn_files"]
    rooks = data["rooks"]

    score = 0
    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        own_files = pawn_files[color]
        enemy_files = pawn_files[BLACK if color == WHITE else WHITE]
        seventh_rank = 6 if color == WHITE else 1

        for s in rooks[color]:
            f = file_of(s)
            if own_files[f] == 0 and enemy_files[f] == 0:
                score += sign * ROOK_OPEN_FILE_BONUS
            elif own_files[f] == 0:
                score += sign * ROOK_SEMI_OPEN_FILE_BONUS
            if rank_of(s) == seventh_rank:
                score += sign * ROOK_SEVENTH_RANK_BONUS

    return score


def mobility_score(board, data=None):
    """Cheap pseudo-mobility for knights and bishops only.

    Counts empty-or-enemy destination squares along each piece's normal
    move pattern (knight jumps; bishop diagonals stopped at first blocker).
    Does not check for pins/check legality -- this is a fast heuristic, not
    a legal move generator.
    """
    if data is None:
        data = _gather(board)

    squares = board.squares
    knights = data["knights"]
    bishops = data["bishops"]

    score = 0
    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        count = 0

        for s in knights[color]:
            f0, r0 = file_of(s), rank_of(s)
            for df, dr in KNIGHT_DELTAS:
                f, r = f0 + df, r0 + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    target = squares[sq(f, r)]
                    if target == "." or (target.isupper() != (color == WHITE)):
                        count += 1

        for s in bishops[color]:
            f0, r0 = file_of(s), rank_of(s)
            for df, dr in BISHOP_DIRS:
                f, r = f0 + df, r0 + dr
                while 0 <= f < 8 and 0 <= r < 8:
                    target = squares[sq(f, r)]
                    if target == ".":
                        count += 1
                    else:
                        if target.isupper() != (color == WHITE):
                            count += 1
                        break
                    f += df
                    r += dr

        score += sign * count * MOBILITY_WEIGHT

    return score


def king_safety_score(board, data=None):
    """Pawn-shield bonus + open-file penalty for both kings.

    Unlike the old version, this no longer requires the king to sit on one
    of 3 exact "home" squares (e1/g1/c1, e8/g8/c8) -- it evaluates whatever
    square the king is actually on, gated by how much material is left on
    the board (a cheap local phase estimate) rather than by square identity.
    That directly covers the missed real-game case of a king on g7 (having
    walked there after castling and being pushed around) still needing its
    pawn cover evaluated.
    """
    if data is None:
        data = _gather(board)

    squares = board.squares
    king_sq = data["king_sq"]
    pawn_files = data["pawn_files"]
    rooks = data["rooks"]
    knights = data["knights"]
    bishops = data["bishops"]
    queens = data["queens"]

    # Cheap material-based phase estimate (1.0 = full opening material,
    # 0.0 = bare endgame). Reused here purely to decide how much king safety
    # still matters -- computed locally so this module stays standalone.
    non_pawn_material = 0
    for color in (WHITE, BLACK):
        non_pawn_material += len(knights[color]) * _PIECE_VAL_LOCAL["N"]
        non_pawn_material += len(bishops[color]) * _PIECE_VAL_LOCAL["B"]
        non_pawn_material += len(rooks[color]) * _PIECE_VAL_LOCAL["R"]
        non_pawn_material += len(queens[color]) * _PIECE_VAL_LOCAL["Q"]
    phase = min(1.0, non_pawn_material / _FULL_NONPAWN_MATERIAL)

    if phase < KING_SAFETY_ENDGAME_PHASE_CUTOFF:
        # True endgame: negligible material left, king safety is moot (the
        # king wants to be active/centralized instead, which is already
        # handled by engine.py's tapered KING_EG_PST).
        return 0

    score = 0
    for color in (WHITE, BLACK):
        sign = 1 if color == WHITE else -1
        ks = king_sq[color]
        if ks is None:
            continue

        f0 = file_of(ks)
        r0 = rank_of(ks)
        own_files = pawn_files[color]
        enemy_color = BLACK if color == WHITE else WHITE
        enemy_heavy = len(rooks[enemy_color]) + len(queens[enemy_color])
        heavy_scale = min(enemy_heavy, HEAVY_PIECE_CAP) / HEAVY_PIECE_CAP

        color_score = 0

        # (a) Pawn shield bonus: pawns directly in front of the king, on its
        # own file and the two adjacent files.
        shield_rank = r0 + 1 if color == WHITE else r0 - 1
        if 0 <= shield_rank < 8:
            pawn_char = "P" if color == WHITE else "p"
            for f in (f0 - 1, f0, f0 + 1):
                if 0 <= f < 8 and squares[sq(f, shield_rank)] == pawn_char:
                    color_score += KING_SHIELD_PAWN_BONUS

        # (b) Open/semi-open file penalty: no own pawns on the king's file
        # or either adjacent file is an open lane for enemy rooks/queens --
        # exactly what unpunished h7-h5/g7-g5 shield pushes created in the
        # observed failure games. Scaled by remaining enemy heavy pieces so
        # it barely matters once rooks/queens are traded off.
        for f, base_penalty in (
            (f0, KING_FILE_OPEN_PENALTY),
            (f0 - 1, KING_ADJ_FILE_OPEN_PENALTY),
            (f0 + 1, KING_ADJ_FILE_OPEN_PENALTY),
        ):
            if 0 <= f < 8 and own_files[f] == 0:
                color_score += base_penalty * heavy_scale

        score += sign * color_score * phase

    return score


def extra_eval(board):
    """Sum of all extra positional evaluation terms, White-perspective centipawns."""
    data = _gather(board)
    return (
        pawn_structure_score(board, data)
        + rook_activity_score(board, data)
        + mobility_score(board, data)
        + king_safety_score(board, data)
    )


if __name__ == "__main__":
    from chess_game import Board

    board = Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"):
        board.push_uci(uci)

    print("FEN:", board.fen())
    print("pawn_structure_score:", pawn_structure_score(board))
    print("rook_activity_score: ", rook_activity_score(board))
    print("mobility_score:      ", mobility_score(board))
    print("king_safety_score:   ", king_safety_score(board))
    print("extra_eval (total):  ", extra_eval(board))
