#!/usr/bin/env python3
"""
Simple, dependency-free chess game with a command interface for testing chess AIs.

Drive it by writing commands to stdin (one per line). Every command prints a
single-line response prefixed with "ok " or "err ", except a few that print
multi-line blocks (board). Designed to be easy to control from another program.

Commands
--------
  move <uci>        Make a move in coordinate notation, e.g. e2e4, e7e8q (promotion).
  legal             List all legal moves for the side to move (space separated UCI).
  legal <sq>        List legal moves originating from a square, e.g. legal e2.
  board             Print the board (ASCII).
  fen               Print the current position as FEN.
  load <fen>        Set the position from a FEN string.
  turn              Print side to move: "white" or "black".
  status            Print game state: ongoing | check | checkmate | stalemate |
                    draw-fifty | draw-repetition | draw-material.
  moves             Print the move history (UCI, space separated).
  undo              Undo the last move.
  reset             Reset to the starting position.
  perft <n>         Count leaf nodes at depth n (move-generator self-test).
  help              Show this help.
  quit / exit       Exit.

Example (from a shell):
  printf 'move e2e4\\nmove e7e5\\nboard\\nfen\\nquit\\n' | python3 chess_game.py
"""

import sys
import copy

WHITE, BLACK = 0, 1
PIECES = "PNBRQK"

# 0x88-free 8x8 board. Squares indexed 0..63, a1=0 ... h8=63.
# board[sq] is a char: 'P','N',... for white, lowercase for black, '.' empty.

FILES = "abcdefgh"
RANKS = "12345678"


def sq(file, rank):
    return rank * 8 + file


def file_of(s):
    return s % 8


def rank_of(s):
    return s // 8


def sq_name(s):
    return FILES[file_of(s)] + RANKS[rank_of(s)]


def name_sq(name):
    return sq(FILES.index(name[0]), RANKS.index(name[1]))


KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Move:
    __slots__ = ("frm", "to", "promo", "flag")

    # flag: '' normal, 'ep' en passant, 'ck'/'cq' castle, '2' double pawn push
    def __init__(self, frm, to, promo="", flag=""):
        self.frm = frm
        self.to = to
        self.promo = promo
        self.flag = flag

    def uci(self):
        return sq_name(self.frm) + sq_name(self.to) + self.promo

    def __eq__(self, o):
        return (self.frm, self.to, self.promo) == (o.frm, o.to, o.promo)


class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        self.squares = list(
            "RNBQKBNR"
            "PPPPPPPP"
            "........"
            "........"
            "........"
            "........"
            "pppppppp"
            "rnbqkbnr"
        )
        self.turn = WHITE
        # castling rights: set of 'K','Q','k','q'
        self.castling = set("KQkq")
        self.ep = None  # en passant target square (int) or None
        self.halfmove = 0  # halfmove clock for 50-move rule
        self.fullmove = 1
        self.history = []  # list of (move, undo_state)
        self.uci_history = []

    # ---- FEN ----
    def fen(self):
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                c = self.squares[sq(file, rank)]
                if c == ".":
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += c
            if empty:
                row += str(empty)
            rows.append(row)
        placement = "/".join(rows)
        turn = "w" if self.turn == WHITE else "b"
        castle = "".join(c for c in "KQkq" if c in self.castling) or "-"
        ep = sq_name(self.ep) if self.ep is not None else "-"
        return f"{placement} {turn} {castle} {ep} {self.halfmove} {self.fullmove}"

    def load_fen(self, fen):
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError("FEN needs at least 4 fields")
        placement, turn, castle, ep = parts[0], parts[1], parts[2], parts[3]
        half = parts[4] if len(parts) > 4 else "0"
        full = parts[5] if len(parts) > 5 else "1"

        squares = ["."] * 64
        rank = 7
        file = 0
        for c in placement:
            if c == "/":
                rank -= 1
                file = 0
            elif c.isdigit():
                file += int(c)
            else:
                if c.upper() not in PIECES:
                    raise ValueError(f"bad piece '{c}'")
                squares[sq(file, rank)] = c
                file += 1
        self.squares = squares
        self.turn = WHITE if turn == "w" else BLACK
        self.castling = set(c for c in castle if c in "KQkq")
        self.ep = None if ep == "-" else name_sq(ep)
        self.halfmove = int(half)
        self.fullmove = int(full)
        self.history = []
        self.uci_history = []

    # ---- helpers ----
    def piece_at(self, s):
        return self.squares[s]

    def color_of(self, c):
        if c == ".":
            return None
        return WHITE if c.isupper() else BLACK

    def king_square(self, color):
        target = "K" if color == WHITE else "k"
        for s in range(64):
            if self.squares[s] == target:
                return s
        return None

    def is_attacked(self, s, by_color):
        """Is square s attacked by any piece of by_color?"""
        f0, r0 = file_of(s), rank_of(s)

        # Pawn attacks: a pawn of by_color attacks s if it sits where it could capture s.
        if by_color == WHITE:
            for df in (-1, 1):
                f, r = f0 + df, r0 - 1
                if 0 <= f < 8 and 0 <= r < 8 and self.squares[sq(f, r)] == "P":
                    return True
        else:
            for df in (-1, 1):
                f, r = f0 + df, r0 + 1
                if 0 <= f < 8 and 0 <= r < 8 and self.squares[sq(f, r)] == "p":
                    return True

        # Knights
        kn = "N" if by_color == WHITE else "n"
        for df, dr in KNIGHT_DELTAS:
            f, r = f0 + df, r0 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.squares[sq(f, r)] == kn:
                return True

        # King
        kg = "K" if by_color == WHITE else "k"
        for df, dr in KING_DELTAS:
            f, r = f0 + df, r0 + dr
            if 0 <= f < 8 and 0 <= r < 8 and self.squares[sq(f, r)] == kg:
                return True

        # Sliding: bishops/queens on diagonals
        bq = ("B", "Q") if by_color == WHITE else ("b", "q")
        for df, dr in BISHOP_DIRS:
            f, r = f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                c = self.squares[sq(f, r)]
                if c != ".":
                    if c in bq:
                        return True
                    break
                f += df
                r += dr

        # Sliding: rooks/queens on ranks/files
        rq = ("R", "Q") if by_color == WHITE else ("r", "q")
        for df, dr in ROOK_DIRS:
            f, r = f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                c = self.squares[sq(f, r)]
                if c != ".":
                    if c in rq:
                        return True
                    break
                f += df
                r += dr

        return False

    def in_check(self, color=None):
        if color is None:
            color = self.turn
        ks = self.king_square(color)
        if ks is None:
            return False
        return self.is_attacked(ks, 1 - color)

    # ---- move generation ----
    def pseudo_legal_moves(self):
        moves = []
        color = self.turn
        for s in range(64):
            c = self.squares[s]
            if c == "." or self.color_of(c) != color:
                continue
            p = c.upper()
            f0, r0 = file_of(s), rank_of(s)
            if p == "P":
                self._pawn_moves(s, f0, r0, color, moves)
            elif p == "N":
                for df, dr in KNIGHT_DELTAS:
                    f, r = f0 + df, r0 + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        t = sq(f, r)
                        if self.color_of(self.squares[t]) != color:
                            moves.append(Move(s, t))
            elif p == "K":
                for df, dr in KING_DELTAS:
                    f, r = f0 + df, r0 + dr
                    if 0 <= f < 8 and 0 <= r < 8:
                        t = sq(f, r)
                        if self.color_of(self.squares[t]) != color:
                            moves.append(Move(s, t))
                self._castle_moves(s, color, moves)
            else:
                dirs = []
                if p in ("B", "Q"):
                    dirs += BISHOP_DIRS
                if p in ("R", "Q"):
                    dirs += ROOK_DIRS
                for df, dr in dirs:
                    f, r = f0 + df, r0 + dr
                    while 0 <= f < 8 and 0 <= r < 8:
                        t = sq(f, r)
                        tc = self.color_of(self.squares[t])
                        if tc == color:
                            break
                        moves.append(Move(s, t))
                        if tc is not None:
                            break
                        f += df
                        r += dr
        return moves

    def _pawn_moves(self, s, f0, r0, color, moves):
        direction = 1 if color == WHITE else -1
        start_rank = 1 if color == WHITE else 6
        promo_rank = 7 if color == WHITE else 0

        # forward
        r1 = r0 + direction
        if 0 <= r1 < 8 and self.squares[sq(f0, r1)] == ".":
            if r1 == promo_rank:
                for pr in "qrbn":
                    moves.append(Move(s, sq(f0, r1), pr))
            else:
                moves.append(Move(s, sq(f0, r1)))
            # double push
            if r0 == start_rank:
                r2 = r0 + 2 * direction
                if self.squares[sq(f0, r2)] == ".":
                    moves.append(Move(s, sq(f0, r2), flag="2"))

        # captures
        for df in (-1, 1):
            f = f0 + df
            r = r0 + direction
            if 0 <= f < 8 and 0 <= r < 8:
                t = sq(f, r)
                tc = self.color_of(self.squares[t])
                if tc is not None and tc != color:
                    if r == promo_rank:
                        for pr in "qrbn":
                            moves.append(Move(s, t, pr))
                    else:
                        moves.append(Move(s, t))
                elif self.ep is not None and t == self.ep:
                    moves.append(Move(s, t, flag="ep"))

    def _castle_moves(self, s, color, moves):
        if self.in_check(color):
            return
        enemy = 1 - color
        if color == WHITE and s == name_sq("e1"):
            if "K" in self.castling and self.squares[name_sq("f1")] == "." and self.squares[name_sq("g1")] == ".":
                if not self.is_attacked(name_sq("f1"), enemy) and not self.is_attacked(name_sq("g1"), enemy):
                    moves.append(Move(s, name_sq("g1"), flag="ck"))
            if "Q" in self.castling and self.squares[name_sq("d1")] == "." and self.squares[name_sq("c1")] == "." and self.squares[name_sq("b1")] == ".":
                if not self.is_attacked(name_sq("d1"), enemy) and not self.is_attacked(name_sq("c1"), enemy):
                    moves.append(Move(s, name_sq("c1"), flag="cq"))
        elif color == BLACK and s == name_sq("e8"):
            if "k" in self.castling and self.squares[name_sq("f8")] == "." and self.squares[name_sq("g8")] == ".":
                if not self.is_attacked(name_sq("f8"), enemy) and not self.is_attacked(name_sq("g8"), enemy):
                    moves.append(Move(s, name_sq("g8"), flag="ck"))
            if "q" in self.castling and self.squares[name_sq("d8")] == "." and self.squares[name_sq("c8")] == "." and self.squares[name_sq("b8")] == ".":
                if not self.is_attacked(name_sq("d8"), enemy) and not self.is_attacked(name_sq("c8"), enemy):
                    moves.append(Move(s, name_sq("c8"), flag="cq"))

    def legal_moves(self):
        legal = []
        color = self.turn
        for m in self.pseudo_legal_moves():
            undo = self._make(m)
            if not self.is_attacked(self.king_square(color), 1 - color):
                legal.append(m)
            self._unmake(undo)
        return legal

    # ---- make / unmake (internal, no legality check) ----
    def _make(self, m):
        undo = (
            self.squares[:],
            self.turn,
            set(self.castling),
            self.ep,
            self.halfmove,
            self.fullmove,
        )
        piece = self.squares[m.frm]
        color = self.turn
        captured = self.squares[m.to]

        self.squares[m.frm] = "."
        # place piece (with promotion)
        if m.promo:
            self.squares[m.to] = m.promo.upper() if color == WHITE else m.promo.lower()
        else:
            self.squares[m.to] = piece

        # en passant capture removes the pawn behind
        if m.flag == "ep":
            cap_sq = m.to + (-8 if color == WHITE else 8)
            self.squares[cap_sq] = "."
            captured = "P" if color == BLACK else "p"

        # castle: move the rook
        if m.flag == "ck":
            if color == WHITE:
                self.squares[name_sq("f1")] = "R"
                self.squares[name_sq("h1")] = "."
            else:
                self.squares[name_sq("f8")] = "r"
                self.squares[name_sq("h8")] = "."
        elif m.flag == "cq":
            if color == WHITE:
                self.squares[name_sq("d1")] = "R"
                self.squares[name_sq("a1")] = "."
            else:
                self.squares[name_sq("d8")] = "r"
                self.squares[name_sq("a8")] = "."

        # update castling rights
        if piece == "K":
            self.castling.discard("K")
            self.castling.discard("Q")
        elif piece == "k":
            self.castling.discard("k")
            self.castling.discard("q")
        for square, right in (
            (name_sq("a1"), "Q"),
            (name_sq("h1"), "K"),
            (name_sq("a8"), "q"),
            (name_sq("h8"), "k"),
        ):
            if m.frm == square or m.to == square:
                self.castling.discard(right)

        # en passant target
        if m.flag == "2":
            self.ep = m.frm + (8 if color == WHITE else -8)
        else:
            self.ep = None

        # halfmove clock
        if piece.upper() == "P" or captured != ".":
            self.halfmove = 0
        else:
            self.halfmove += 1

        if color == BLACK:
            self.fullmove += 1
        self.turn = 1 - color
        return undo

    def _unmake(self, undo):
        (
            self.squares,
            self.turn,
            self.castling,
            self.ep,
            self.halfmove,
            self.fullmove,
        ) = undo

    # ---- public move interface ----
    def find_legal(self, uci):
        uci = uci.strip().lower()
        for m in self.legal_moves():
            if m.uci() == uci:
                return m
        # allow omitting promotion default (assume queen) if unambiguous
        if len(uci) == 4:
            for m in self.legal_moves():
                if m.uci() == uci + "q":
                    return m
        return None

    def push_uci(self, uci):
        m = self.find_legal(uci)
        if m is None:
            return False
        self.history.append(self._make(m))
        self.uci_history.append(m.uci())
        return True

    def undo(self):
        if not self.history:
            return False
        self._unmake(self.history.pop())
        self.uci_history.pop()
        return True

    # ---- status ----
    def insufficient_material(self):
        pieces = [c for c in self.squares if c != "."]
        # only kings
        non_king = [c for c in pieces if c.upper() != "K"]
        if not non_king:
            return True
        # king + single minor vs king
        if len(non_king) == 1 and non_king[0].upper() in ("N", "B"):
            return True
        # king+bishop vs king+bishop, same color bishops
        if len(non_king) == 2 and all(c.upper() == "B" for c in non_king):
            bishop_sqs = [i for i, c in enumerate(self.squares) if c.upper() == "B"]
            colors = {(file_of(s) + rank_of(s)) % 2 for s in bishop_sqs}
            if len(colors) == 1:
                return True
        return False

    def status(self):
        legal = self.legal_moves()
        if not legal:
            if self.in_check():
                return "checkmate"
            return "stalemate"
        if self.halfmove >= 100:
            return "draw-fifty"
        if self.insufficient_material():
            return "draw-material"
        if self.in_check():
            return "check"
        return "ongoing"

    # ---- rendering ----
    GLYPHS = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗",
        "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝",
        "n": "♞", "p": "♟", ".": "·",
    }

    def ascii(self, unicode=True):
        def cell(c):
            if not unicode:
                return c
            return self.GLYPHS.get(c, c)

        lines = []
        for rank in range(7, -1, -1):
            row = [f"{rank + 1} │"]
            for file in range(8):
                row.append(cell(self.squares[sq(file, rank)]))
            lines.append(" ".join(row))
        lines.append("  └" + "─" * 17)
        lines.append("    " + " ".join(FILES))
        return "\n".join(lines)


def perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for m in board.legal_moves():
        undo = board._make(m)
        total += perft(board, depth - 1)
        board._unmake(undo)
    return total


def run_repl(inp=sys.stdin, out=sys.stdout):
    board = Board()

    def emit(line):
        out.write(line + "\n")
        out.flush()

    for raw in inp:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        try:
            if cmd in ("quit", "exit"):
                emit("ok bye")
                break
            elif cmd == "help":
                for l in __doc__.strip().splitlines():
                    emit("  " + l)
            elif cmd == "move":
                if not arg:
                    emit("err move requires a UCI move, e.g. move e2e4")
                elif board.push_uci(arg):
                    st = board.status()
                    emit(f"ok {arg} {board.fen()} | {st}")
                else:
                    emit(f"err illegal move: {arg}")
            elif cmd == "legal":
                moves = board.legal_moves()
                if arg:
                    try:
                        origin = name_sq(arg[:2])
                    except (ValueError, IndexError):
                        emit(f"err bad square: {arg}")
                        continue
                    moves = [m for m in moves if m.frm == origin]
                emit("ok " + " ".join(m.uci() for m in moves))
            elif cmd == "board":
                emit("ok")
                emit(board.ascii())
            elif cmd == "fen":
                emit("ok " + board.fen())
            elif cmd == "load":
                fen = line[len("load"):].strip()
                board.load_fen(fen)
                emit("ok " + board.fen())
            elif cmd == "turn":
                emit("ok " + ("white" if board.turn == WHITE else "black"))
            elif cmd == "status":
                emit("ok " + board.status())
            elif cmd == "moves":
                emit("ok " + " ".join(board.uci_history))
            elif cmd == "undo":
                emit("ok undone" if board.undo() else "err nothing to undo")
            elif cmd == "reset":
                board.reset()
                emit("ok " + board.fen())
            elif cmd == "perft":
                n = int(arg) if arg else 1
                emit(f"ok perft({n})={perft(board, n)}")
            else:
                emit(f"err unknown command: {cmd} (try 'help')")
        except Exception as e:
            emit(f"err {type(e).__name__}: {e}")


if __name__ == "__main__":
    run_repl()
