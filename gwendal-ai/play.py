#!/usr/bin/env python3
"""
Drive chess_game.py (the referee) as a subprocess and play using engine.py,
following the exact integration pattern documented in the project's CLAUDE.md
"Typical test loop": query `legal`, pick a move, submit `move <uci>`, repeat.

Usage:
    python3 play.py white 8.0     # engine plays White, 8s/move, opponent input via stdin uci moves
    python3 play.py self 8.0      # engine plays both sides (self-play demo)
"""
import subprocess
import sys

from engine import choose_move_from_fen
from chess_game import Board


def start_referee():
    return subprocess.Popen(
        ["python3", "chess_game.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )


def send(p, cmd):
    p.stdin.write(cmd + "\n")
    p.stdin.flush()
    return p.stdout.readline().strip()


def main():
    our_color = sys.argv[1] if len(sys.argv) > 1 else "self"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    ref = start_referee()
    ply = 0
    while True:
        legal = send(ref, "legal").removeprefix("ok ").split()
        if not legal:
            break
        turn = send(ref, "turn").removeprefix("ok ")
        fen = send(ref, "fen").removeprefix("ok ")

        if our_color == "self" or turn == our_color:
            move = choose_move_from_fen(fen, time_limit=time_limit)
        else:
            print(f"opponent to move ({turn}). legal: {' '.join(legal)}")
            move = input("your move (uci): ").strip()

        resp = send(ref, f"move {move}")
        ply += 1
        print(f"ply {ply}: {turn} played {move} | {resp}")
        if not resp.startswith("ok"):
            print("illegal move returned by engine, aborting")
            break
        if resp.endswith(("checkmate", "stalemate")) or "draw" in resp:
            break

    send(ref, "quit")


if __name__ == "__main__":
    main()
