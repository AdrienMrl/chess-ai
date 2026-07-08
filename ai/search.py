"""Negamax + alpha-beta search with a transposition table, quiescence search,
move ordering (MVV-LVA + killers + history), and time-boxed iterative deepening.

Move generation uses chess_game.Board's own _make/_unmake pair directly (no
subprocess, no FEN formatting) so the hot loop stays cheap.
"""
import time

from ai.zobrist import hash_board

INF = 10 ** 9
MATE = 100_000
MAX_PLY = 64
MAX_DEPTH = 32

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

PIECE_ORDER_VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20_000}

NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2


class TimeUp(Exception):
    pass


class TTEntry:
    __slots__ = ("depth", "score", "flag", "best_move")

    def __init__(self, depth, score, flag, best_move):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


def is_capture(board, move):
    return move.flag == "ep" or board.squares[move.to] != "."


def _mvv_lva_score(board, move):
    if move.flag == "ep":
        victim_val = PIECE_ORDER_VALUE["P"]
    else:
        victim = board.squares[move.to]
        victim_val = PIECE_ORDER_VALUE[victim.upper()] if victim != "." else 0
    attacker_val = PIECE_ORDER_VALUE[board.squares[move.frm].upper()]
    return victim_val * 10 - attacker_val


def order_moves(board, moves, tt_move, killer_pair, history):
    def score(m):
        if tt_move is not None and m == tt_move:
            return 1_000_000
        if is_capture(board, m):
            return 100_000 + _mvv_lva_score(board, m)
        if killer_pair[0] is not None and m == killer_pair[0]:
            return 90_000
        if killer_pair[1] is not None and m == killer_pair[1]:
            return 89_000
        return history.get((m.frm, m.to), 0)

    return sorted(moves, key=score, reverse=True)


def _has_non_pawn_material(board, color):
    """True if `color` has any piece other than pawns/king.

    Used to withhold null-move pruning in king+pawn-only positions, where
    zugzwang (every move worsens your position) makes the "opponent gets a
    free move and still can't beat beta" assumption unsound.
    """
    is_white = color == 0
    for c in board.squares:
        if c == "." or c.upper() in ("P", "K"):
            continue
        if c.isupper() == is_white:
            return True
    return False


def _make_null_move(board):
    undo = (board.turn, board.ep)
    board.turn = 1 - board.turn
    board.ep = None
    return undo


def _unmake_null_move(board, undo):
    board.turn, board.ep = undo


def _relative_eval(board, eval_fn):
    """eval_fn is White-relative; negamax wants side-to-move-relative."""
    v = eval_fn(board)
    return v if board.turn == 0 else -v


def quiescence(board, alpha, beta, ply, deadline, eval_fn, qdepth=6):
    if time.monotonic() > deadline:
        raise TimeUp

    stand_pat = _relative_eval(board, eval_fn)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0 or ply >= MAX_PLY - 1:
        return alpha

    pseudo_captures = [m for m in board.pseudo_legal_moves() if is_capture(board, m)]
    pseudo_captures.sort(key=lambda m: _mvv_lva_score(board, m), reverse=True)

    color = board.turn
    for m in pseudo_captures:
        undo = board._make(m)
        try:
            if board.is_attacked(board.king_square(color), 1 - color):
                continue
            score = -quiescence(board, -beta, -alpha, ply + 1, deadline, eval_fn, qdepth - 1)
        finally:
            board._unmake(undo)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax(board, depth, alpha, beta, ply, deadline, killers, history, eval_fn, tt):
    if time.monotonic() > deadline:
        raise TimeUp

    h = hash_board(board)
    tt_entry = tt.get(h)
    tt_move = None
    if tt_entry is not None:
        tt_move = tt_entry.best_move
        if tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.score
            elif tt_entry.flag == TT_LOWER:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_UPPER:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

    if depth <= 0:
        return quiescence(board, alpha, beta, ply, deadline, eval_fn)

    in_check = board.in_check()
    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not in_check
        and beta < MATE - MAX_PLY
        and _has_non_pawn_material(board, board.turn)
    ):
        null_undo = _make_null_move(board)
        try:
            null_score = -negamax(
                board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1,
                ply + 1, deadline, killers, history, eval_fn, tt,
            )
        finally:
            _unmake_null_move(board, null_undo)
        if null_score >= beta:
            return beta

    moves = board.legal_moves()
    if not moves:
        return -(MATE - ply) if in_check else 0

    ordered = order_moves(board, moves, tt_move, killers[ply], history)

    best_score = -INF
    best_move = None
    orig_alpha = alpha
    for m in ordered:
        undo = board._make(m)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline, killers, history, eval_fn, tt)
        finally:
            board._unmake(undo)

        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if not is_capture(board, m):
                if killers[ply][0] is None or killers[ply][0] != m:
                    killers[ply][1] = killers[ply][0]
                    killers[ply][0] = m
                history[(m.frm, m.to)] = history.get((m.frm, m.to), 0) + depth * depth
            break

    flag = TT_EXACT
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    tt[h] = TTEntry(depth, best_score, flag, best_move)
    return best_score


def _search_at_depth(board, depth, deadline, killers, history, eval_fn, tt):
    moves = board.legal_moves()
    if not moves:
        return 0, None

    h = hash_board(board)
    tt_entry = tt.get(h)
    tt_move = tt_entry.best_move if tt_entry else None
    ordered = order_moves(board, moves, tt_move, killers[0], history)

    alpha, beta = -INF, INF
    best_score = -INF
    best_move = ordered[0]
    for m in ordered:
        undo = board._make(m)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, 1, deadline, killers, history, eval_fn, tt)
        finally:
            board._unmake(undo)
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score

    tt[h] = TTEntry(depth, best_score, TT_EXACT, best_move)
    return best_score, best_move


def search_root(board, time_budget_s, eval_fn, tt=None):
    """Iterative-deepening search from the current position.

    Returns (best_move, best_score, depth_reached). best_score is in
    centipawns from the side-to-move's perspective. tt (a dict) can be
    passed in to persist across moves; a fresh one is created if omitted.
    """
    if tt is None:
        tt = {}
    deadline = time.monotonic() + time_budget_s
    killers = [[None, None] for _ in range(MAX_PLY)]
    history = {}

    best_move = None
    best_score = 0
    depth_reached = 0
    depth = 1
    while depth <= MAX_DEPTH:
        try:
            score, move = _search_at_depth(board, depth, deadline, killers, history, eval_fn, tt)
        except TimeUp:
            break
        if move is not None:
            best_move, best_score = move, score
        depth_reached = depth
        if move is None:
            break  # no legal moves at all
        depth += 1
        if (deadline - time.monotonic()) < time_budget_s * 0.5:
            break

    if best_move is None:
        # Depth 1 never finished in time (or the loop never ran) — guarantee a
        # legal move anyway rather than returning None against a hard deadline.
        legal = board.legal_moves()
        if legal:
            best_move = legal[0]

    return best_move, best_score, depth_reached
