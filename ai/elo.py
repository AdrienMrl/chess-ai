#!/usr/bin/env python3
"""Estimate the Elo rating of the ai/ chess engine.

Plays calibrated games against Stockfish, pinned to a target strength via its
UCI_LimitStrength/UCI_Elo options (Stockfish 18 supports Elo 1320-3190), then
back-solves the standard Elo expected-score formula from the resulting score
fraction:

    E = 1 / (1 + 10^((opponent_elo - engine_elo) / 400))
    => engine_elo = opponent_elo + 400 * log10(E / (1 - E))

This is a coarse statistical estimate, not a rigorous rating -- accuracy
improves with more games and with picking --opponent-elo close to the
engine's true strength (extreme score fractions near 0% or 100% give wide,
unreliable estimates). Read the printed standard error accordingly.

Usage:
    python3 -m ai.elo --games 20 --opponent-elo 1500 --time 1.0 \
        --opponent-movetime 1.0 --eval classical
"""
import argparse
import math
import subprocess
import sys
import time

from chess_game import Board

from ai.engine import Engine

MAX_PLIES = 200
TERMINAL = {"checkmate", "stalemate", "draw-fifty", "draw-material", "draw-repetition"}
STOCKFISH_ELO_RANGE = (1320, 3190)


class StockfishOpponent:
    def __init__(self, target_elo, movetime_s):
        self.movetime_ms = int(movetime_s * 1000)
        self.proc = subprocess.Popen(
            ["stockfish"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name UCI_LimitStrength value true")
        self._send(f"setoption name UCI_Elo value {target_elo}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("stockfish process ended unexpectedly")
            if line.strip() == token:
                return

    def best_move(self, fen):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go movetime {self.movetime_ms}")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("stockfish process ended unexpectedly")
            line = line.strip()
            if line.startswith("bestmove"):
                return line.split()[1]

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_one_game(engine, opponent, engine_is_white):
    board = Board()
    for _ in range(MAX_PLIES):
        status = board.status()
        if status in TERMINAL:
            return status, board
        engine_to_move = (board.turn == 0) == engine_is_white
        if engine_to_move:
            move, _, _ = engine.pick_move(board)
            uci = move.uci()
        else:
            uci = opponent.best_move(board.fen())
        if not board.push_uci(uci):
            raise RuntimeError(f"illegal move proposed: {uci}")
    return "ply-cap", board


def score_for_engine(status, board, engine_is_white):
    """1.0 / 0.5 / 0.0 (win/draw/loss) from the engine's perspective."""
    if status == "checkmate":
        loser_is_white = (board.turn == 0)  # side to move is the one mated
        return 0.0 if loser_is_white == engine_is_white else 1.0
    if status in ("stalemate", "draw-fifty", "draw-material", "draw-repetition", "ply-cap"):
        return 0.5
    raise RuntimeError(f"unexpected terminal status: {status}")


def main():
    parser = argparse.ArgumentParser(description="Estimate the ai/ engine's Elo via calibrated games vs Stockfish.")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--opponent-elo", type=int, required=True,
                         help=f"Stockfish UCI_Elo target ({STOCKFISH_ELO_RANGE[0]}-{STOCKFISH_ELO_RANGE[1]})")
    parser.add_argument("--time", type=float, required=True, help="engine time budget per move (seconds)")
    parser.add_argument("--opponent-movetime", type=float, required=True,
                         help="Stockfish movetime per move (seconds)")
    parser.add_argument("--eval", choices=("classical", "nnue", "auto"), required=True)
    args = parser.parse_args()

    if args.games < 1:
        raise ValueError("--games must be >= 1")
    if not (STOCKFISH_ELO_RANGE[0] <= args.opponent_elo <= STOCKFISH_ELO_RANGE[1]):
        raise ValueError(f"--opponent-elo must be within {STOCKFISH_ELO_RANGE[0]}-{STOCKFISH_ELO_RANGE[1]}")

    use_nnue = {"classical": "off", "nnue": "on", "auto": "auto"}[args.eval]
    engine = Engine(time_budget=args.time, use_nnue=use_nnue)
    opponent = StockfishOpponent(args.opponent_elo, args.opponent_movetime)

    scores = []
    try:
        for g in range(args.games):
            engine_is_white = (g % 2 == 0)
            t0 = time.monotonic()
            status, board = play_one_game(engine, opponent, engine_is_white)
            s = score_for_engine(status, board, engine_is_white)
            scores.append(s)
            elapsed = time.monotonic() - t0
            color = "white" if engine_is_white else "black"
            print(f"game {g + 1}/{args.games}: engine={color:5s} status={status:10s} "
                  f"score={s} ({elapsed:.1f}s)", file=sys.stderr)
    finally:
        opponent.close()

    n = len(scores)
    total = sum(scores)
    frac = total / n
    # Clamp away from 0/1 so log10 stays finite; width shrinks as n grows.
    eps = 1 / (2 * n + 2)
    frac_clamped = min(max(frac, eps), 1 - eps)
    elo_diff = 400 * math.log10(frac_clamped / (1 - frac_clamped))
    estimated_elo = args.opponent_elo + elo_diff

    if 0 < frac < 1:
        se_frac = math.sqrt(frac * (1 - frac) / n)
        se_elo = se_frac * 400 / (math.log(10) * frac_clamped * (1 - frac_clamped))
    else:
        se_elo = float("inf")

    print()
    print(f"games played: {n}")
    print(f"score: {total:.1f}/{n} ({frac * 100:.1f}%)")
    print(f"opponent Elo (Stockfish, calibrated): {args.opponent_elo}")
    print(f"estimated engine Elo: {estimated_elo:.0f}  (+/- ~{se_elo:.0f}, 1 std err)")
    print()
    print("Coarse estimate from a small sample against one calibrated opponent strength.")
    print("More games, and/or --opponent-elo closer to the true strength, tighten this.")


if __name__ == "__main__":
    main()
