#!/usr/bin/env python3
"""
From-scratch chess AI: negamax + alpha-beta, iterative deepening, Zobrist-hashed
transposition table, quiescence search, tapered material+PST evaluation, and
move ordering (TT move, MVV-LVA captures, killers, history heuristic).

No chess/AI libraries used. Board rules/legality come from chess_game.Board
(the project's own rules referee), used in-process for speed.

Public entry point: choose_move(board, time_limit=1.0) -> uci string
"""
import random
import time
from collections import defaultdict

from chess_game import (
    WHITE, BLACK, sq, file_of, rank_of,
    ZOBRIST_PIECE, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP_FILE,
)
from eval_extra import extra_eval

INF = 10 ** 9
MATE = 100_000

PIECE_VAL = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}


def _rows_to_sq_order(rows_rank8_to_rank1):
    """Convert 8 rows given top-to-bottom (rank8..rank1) into a 64-list indexed
    by sq(file,rank)=rank*8+file (rank1..rank8)."""
    reversed_rows = list(reversed(rows_rank8_to_rank1))
    out = []
    for row in reversed_rows:
        out.extend(row)
    return out


PAWN_PST = _rows_to_sq_order([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

KNIGHT_PST = _rows_to_sq_order([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
])

BISHOP_PST = _rows_to_sq_order([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
])

ROOK_PST = _rows_to_sq_order([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
])

QUEEN_PST = _rows_to_sq_order([
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
])

KING_MG_PST = _rows_to_sq_order([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20],
])

KING_EG_PST = _rows_to_sq_order([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
])

PST = {'P': PAWN_PST, 'N': KNIGHT_PST, 'B': BISHOP_PST, 'R': ROOK_PST, 'Q': QUEEN_PST}

OPENING_NONPAWN_MATERIAL = 2 * (PIECE_VAL['N'] + PIECE_VAL['B'] + PIECE_VAL['R']) + 2 * PIECE_VAL['Q']


def mirror(s):
    return s ^ 56


def evaluate_white(board):
    """Static eval, positive = good for White."""
    score = 0
    non_pawn_material = 0
    bishops_w = bishops_b = 0
    for s in range(64):
        c = board.squares[s]
        if c == '.':
            continue
        p = c.upper()
        white = c.isupper()
        val = PIECE_VAL[p]
        if p == 'K':
            continue
        pst_table = PST[p]
        bonus = pst_table[s] if white else pst_table[mirror(s)]
        if white:
            score += val + bonus
        else:
            score -= val + bonus
        if p != 'P':
            non_pawn_material += val
        if p == 'B':
            if white:
                bishops_w += 1
            else:
                bishops_b += 1

    if bishops_w >= 2:
        score += 30
    if bishops_b >= 2:
        score -= 30

    phase = min(1.0, non_pawn_material / OPENING_NONPAWN_MATERIAL)
    wk = board.king_square(WHITE)
    bk = board.king_square(BLACK)
    if wk is not None:
        mg = KING_MG_PST[wk]
        eg = KING_EG_PST[wk]
        score += phase * mg + (1 - phase) * eg
    if bk is not None:
        mbk = mirror(bk)
        mg = KING_MG_PST[mbk]
        eg = KING_EG_PST[mbk]
        score -= phase * mg + (1 - phase) * eg

    score += extra_eval(board)

    return score


def evaluate_stm(board):
    """Eval from side-to-move perspective (for negamax)."""
    v = evaluate_white(board)
    return v if board.turn == WHITE else -v


# ---- Zobrist hashing ----
# Tables live in chess_game.py so Board can maintain an incrementally-updated
# board.zobrist attribute (kept in sync on every _make/_unmake) instead of
# rescanning all 64 squares on every search node. zobrist_hash() below is kept
# only as a full-rescan ground truth / fallback, not used on the hot path.


def zobrist_hash(board):
    h = 0
    for s in range(64):
        c = board.squares[s]
        if c != '.':
            h ^= ZOBRIST_PIECE[c][s]
    if board.turn == BLACK:
        h ^= ZOBRIST_SIDE
    for r in board.castling:
        h ^= ZOBRIST_CASTLE[r]
    if board.ep is not None:
        h ^= ZOBRIST_EP_FILE[file_of(board.ep)]
    return h


def tt_store_score(score, ply):
    """Mate scores are ply-from-root-dependent; TT entries are keyed only by
    position, so store them as ply-from-this-node to stay valid across
    transpositions reached at a different search ply."""
    if score >= MATE - 1000:
        return score + ply
    if score <= -MATE + 1000:
        return score - ply
    return score


def tt_retrieve_score(score, ply):
    if score >= MATE - 1000:
        return score - ply
    if score <= -MATE + 1000:
        return score + ply
    return score


def has_non_pawn_material(board, color):
    """Cheap check to avoid null-move pruning in zugzwang-prone K+P endgames.
    Early-exits on the first non-pawn/king piece found, so it's O(1) in
    practice for any position with developed pieces."""
    squares = board.squares
    want_upper = (color == WHITE)
    for s in range(64):
        c = squares[s]
        if c == '.':
            continue
        if c.isupper() == want_upper and c.upper() not in ('P', 'K'):
            return True
    return False


def is_capture(board, m):
    return board.squares[m.to] != '.' or m.flag == 'ep'


def legal_captures(board):
    """Cheaper than [m for m in board.legal_moves() if is_capture(board,m)]:
    filters to captures BEFORE paying the make/is_attacked/unmake legality
    check, instead of after. Quiescence nodes vastly outnumber full-depth
    nodes, so this matters a lot for throughput."""
    color = board.turn
    enemy = 1 - color
    out = []
    for m in board.pseudo_legal_moves():
        if not is_capture(board, m):
            continue
        undo = board._make(m)
        legal = not board.is_attacked(board.king_square(color), enemy)
        board._unmake(undo)
        if legal:
            out.append(m)
    return out


def mvv_lva_score(board, m):
    victim = board.squares[m.to]
    victim_val = PIECE_VAL[victim.upper()] if victim != '.' else PIECE_VAL['P']
    attacker_val = PIECE_VAL[board.squares[m.frm].upper()]
    return victim_val * 16 - attacker_val


class Engine:
    def __init__(self):
        self.tt = {}
        self.killers = defaultdict(list)
        self.history = defaultdict(int)
        self.deadline = None
        self.nodes = 0

    def _time_up(self):
        return time.time() >= self.deadline

    def order_moves(self, board, moves, tt_move_uci, ply):
        def key(m):
            u = m.uci()
            if tt_move_uci and u == tt_move_uci:
                return (0, 0)
            if is_capture(board, m):
                return (1, -mvv_lva_score(board, m))
            kl = self.killers.get(ply, [])
            if u in kl:
                return (2, kl.index(u))
            h = self.history.get((m.frm, m.to), 0)
            return (3, -h)
        return sorted(moves, key=key)

    def quiescence(self, board, alpha, beta, qdepth):
        self.nodes += 1
        if self.nodes % 256 == 0 and self._time_up():
            raise TimeoutError
        # Quiescence only searches captures, so it can't tell a normal quiet
        # position apart from checkmate/stalemate. That's fine when not in
        # check (stand-pat is a sound floor), but if check extensions ran out
        # mid-forcing-sequence and we land here still in check, a checkmate
        # must not be scored as a quiet material stand-pat.
        if board.in_check() and not board.legal_moves():
            return -(MATE - 1)
        stand_pat = evaluate_stm(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if qdepth <= 0:
            return alpha
        moves = legal_captures(board)
        moves.sort(key=lambda m: -mvv_lva_score(board, m))
        for m in moves:
            undo = board._make(m)
            try:
                score = -self.quiescence(board, -beta, -alpha, qdepth - 1)
            finally:
                board._unmake(undo)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    MAX_CHECK_EXTENSIONS = 6

    def negamax(self, board, depth, alpha, beta, ply, ext=0):
        self.nodes += 1
        if self.nodes % 256 == 0 and self._time_up():
            raise TimeoutError

        # Must precede the TT probe: the Zobrist key doesn't encode the
        # halfmove clock, so a position cached as a normal score at a low
        # halfmove count could otherwise be wrongly reused as non-draw once
        # the clock reaches the forced-draw threshold.
        if board.halfmove >= 100:
            return 0

        alpha_orig = alpha
        key = board.zobrist
        entry = self.tt.get(key)
        tt_move_uci = None
        if entry is not None:
            tt_move_uci = entry.get('move')
            if entry['depth'] >= depth:
                tt_score = tt_retrieve_score(entry['score'], ply)
                if entry['flag'] == 'exact':
                    return tt_score
                elif entry['flag'] == 'lower':
                    alpha = max(alpha, tt_score)
                elif entry['flag'] == 'upper':
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        moves = board.legal_moves()
        if not moves:
            if board.in_check():
                return -MATE + ply
            return 0

        in_check = board.in_check()

        # Check extension: being in check is tactically forcing, so don't let
        # it fall into quiescence (which only considers captures and would
        # miss non-capture check evasions) — search it one ply deeper
        # instead, capped so a pathological line can't blow the stack.
        if in_check and ext < self.MAX_CHECK_EXTENSIONS:
            depth += 1
            ext += 1

        if depth <= 0:
            return self.quiescence(board, alpha, beta, 8)

        if (
            depth >= 3
            and not in_check
            and has_non_pawn_material(board, board.turn)
        ):
            R = 2
            saved_turn = board.turn
            saved_ep = board.ep
            board.turn = 1 - board.turn
            board.ep = None
            try:
                null_score = -self.negamax(
                    board, max(0, depth - 1 - R), -beta, -beta + 1, ply + 1, ext
                )
            finally:
                board.turn = saved_turn
                board.ep = saved_ep
            if null_score >= beta:
                return beta

        ordered = self.order_moves(board, moves, tt_move_uci, ply)
        best_score = -INF
        best_uci = None
        for m in ordered:
            undo = board._make(m)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, ext)
            finally:
                board._unmake(undo)
            if score > best_score:
                best_score = score
                best_uci = m.uci()
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_capture(board, m):
                    kl = self.killers[ply]
                    if m.uci() not in kl:
                        kl.insert(0, m.uci())
                        del kl[2:]
                    self.history[(m.frm, m.to)] += depth * depth
                break

        flag = 'exact'
        if best_score <= alpha_orig:
            flag = 'upper'
        elif best_score >= beta:
            flag = 'lower'
        # Depth-preferred replacement: don't let a shallower re-search (e.g.
        # from a null-move probe) clobber a deeper, more reliable entry.
        if entry is None or depth >= entry['depth']:
            self.tt[key] = {
                'depth': depth,
                'score': tt_store_score(best_score, ply),
                'flag': flag,
                'move': best_uci,
            }
        return best_score

    def search_root(self, board, depth):
        moves = board.legal_moves()
        key = board.zobrist
        entry = self.tt.get(key)
        tt_move_uci = entry.get('move') if entry else None
        ordered = self.order_moves(board, moves, tt_move_uci, 0)
        alpha, beta = -INF, INF
        best_score = -INF
        best_uci = ordered[0].uci()
        for m in ordered:
            undo = board._make(m)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board._unmake(undo)
            if score > best_score:
                best_score = score
                best_uci = m.uci()
            if score > alpha:
                alpha = score
        return best_score, best_uci

    # Safety margin subtracted from the requested budget so we always return
    # a move comfortably before an external hard limit (e.g. a 10s/move cap
    # in a match), even accounting for one extra ply of search overrun
    # between time checks.
    SAFETY_MARGIN = 0.9

    def choose_move(self, board, time_limit=1.0):
        start = time.time()
        budget = max(0.05, time_limit - self.SAFETY_MARGIN)
        self.deadline = start + budget
        self.nodes = 0
        self.killers = defaultdict(list)

        moves = board.legal_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0].uci()

        best_uci = moves[0].uci()
        depth = 1
        try:
            while True:
                if time.time() >= self.deadline:
                    break
                score, move_uci = self.search_root(board, depth)
                best_uci = move_uci
                if time.time() >= self.deadline:
                    break
                if abs(score) >= MATE - 1000:
                    break
                depth += 1
                if depth > 64:
                    break
        except TimeoutError:
            pass
        return best_uci


_default_engine = Engine()


def choose_move(board, time_limit=1.0):
    return _default_engine.choose_move(board, time_limit=time_limit)


def choose_move_from_fen(fen, time_limit=1.0):
    """Convenience entry point matching the referee's FEN-driven protocol:
    given just a FEN string, return our chosen move in UCI."""
    from chess_game import Board
    b = Board()
    b.load_fen(fen)
    return choose_move(b, time_limit=time_limit)


if __name__ == "__main__":
    from chess_game import Board
    b = Board()
    eng = Engine()
    for i in range(6):
        mv = eng.choose_move(b, time_limit=1.0)
        print(i, b.turn, mv, "nodes:", eng.nodes)
        if mv is None:
            break
        b.push_uci(mv)
    print(b.fen())
