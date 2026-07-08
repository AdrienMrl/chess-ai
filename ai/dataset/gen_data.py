"""Generate NNUE-style training data: Stockfish-driven (varied-strength) self-play,
labeled with full-strength Stockfish eval.

Move selection uses Stockfish itself (UCI_LimitStrength, Elo randomized per game)
rather than uniform-random legal moves -- plain random self-play reaches chaotic,
blundered positions almost immediately, which is a poor match for the kind of
positions a real eval function needs to judge. A small epsilon of genuinely
random moves is kept on top for exploration/diversity.

Usage:
    python3 ai/dataset/gen_data.py --count 30000
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

_AI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_AI_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chess_game import Board

TERMINAL_STATUSES = {"checkmate", "stalemate", "draw-fifty", "draw-repetition", "draw-material"}
LABEL_SEARCH_DEPTH = 10
MOVE_SEARCH_DEPTH = 5
MOVE_ELO_RANGE = (1320, 2200)  # 1320 is Stockfish's UCI_Elo floor
RANDOM_MOVE_EPSILON = 0.08  # small chance of a genuinely random move, for diversity
SKIP_PLIES = 6
PROGRESS_EVERY = 500


class _StockfishProcess:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["stockfish"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("stockfish process ended unexpectedly")
        return line.strip()

    def _wait_for(self, token):
        while True:
            if self._readline() == token:
                return

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class StockfishEval(_StockfishProcess):
    """Full-strength engine used only to label sampled positions."""

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self._send("isready")
        self._wait_for("readyok")

    def evaluate(self, fen):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        last_score = None
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                break
            if " score cp " in line:
                last_score = int(line.split(" score cp ")[1].split()[0])
            elif " score mate " in line:
                mate_in = int(line.split(" score mate ")[1].split()[0])
                sign = 1 if mate_in > 0 else -1
                last_score = sign * (10000 - abs(mate_in) * 100)
        if last_score is None:
            raise RuntimeError(f"no score parsed for fen: {fen}")
        return last_score


class StockfishMover(_StockfishProcess):
    """Elo-limited engine used to pick self-play moves, so games look like
    plausible (if imperfect) chess instead of undirected random walks."""

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self._send("isready")
        self._wait_for("readyok")

    def set_elo(self, elo):
        self._send("setoption name UCI_LimitStrength value true")
        self._send(f"setoption name UCI_Elo value {elo}")
        self._send("isready")
        self._wait_for("readyok")

    def best_move_uci(self, fen):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                return line.split()[1]


def play_one_game(mover):
    board = Board()
    target_plies = random.randint(20, 60)
    sample_interval = random.choice((2, 3))
    mover.set_elo(random.randint(*MOVE_ELO_RANGE))
    ply = 0
    while ply < target_plies:
        if board.status() in TERMINAL_STATUSES:
            return
        moves = board.legal_moves()
        if not moves:
            return
        if random.random() < RANDOM_MOVE_EPSILON:
            move_uci = random.choice(moves).uci()
        else:
            move_uci = mover.best_move_uci(board.fen())
        if not board.push_uci(move_uci):
            board.push_uci(random.choice(moves).uci())  # defensive: bestmove should always be legal
        ply += 1
        if ply >= SKIP_PLIES and (ply - SKIP_PLIES) % sample_interval == 0:
            yield board.fen()


def sample_stream(mover):
    while True:
        yield from play_one_game(mover)


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled chess positions for NNUE training via self-play + Stockfish eval."
    )
    parser.add_argument("--count", type=int, required=True, help="number of labeled positions to generate")
    parser.add_argument("--out", type=str, default=None, help="output path (default: ai/data/positions.jsonl)")
    args = parser.parse_args()

    out_dir = os.path.join(_AI_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out if args.out else os.path.join(out_dir, "positions.jsonl")

    label_engine = StockfishEval(LABEL_SEARCH_DEPTH)
    mover = StockfishMover(MOVE_SEARCH_DEPTH)
    start = time.time()
    written = 0
    last_report = 0
    try:
        with open(out_path, "w") as out_f:
            for fen in sample_stream(mover):
                if written >= args.count:
                    break
                cp = label_engine.evaluate(fen)
                out_f.write(json.dumps({"fen": fen, "cp": cp}) + "\n")
                out_f.flush()
                written += 1
                if written - last_report >= PROGRESS_EVERY:
                    elapsed = time.time() - start
                    rate = written / elapsed if elapsed > 0 else 0.0
                    print(
                        f"{written} positions | {rate:.1f} pos/s | {elapsed:.1f}s elapsed",
                        file=sys.stderr,
                    )
                    last_report = written
    finally:
        label_engine.close()
        mover.close()

    elapsed = time.time() - start
    print(f"Done: {written} positions written to {out_path} in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
