#!/usr/bin/env python3
"""CLI to play/test the ai/ chess engine against a human, itself, or a random mover.

Usage:
    python3 -m ai.play --white engine --black random --time 2.0
    python3 -m ai.play --white human --black engine --time 3.0
    python3 -m ai.play --white engine --black engine --time 1.0 --games 5
    python3 -m ai.play --white engine --black random --time 2.0 --fen "<fen>"
"""
import argparse
import random
import sys

from chess_game import Board

from ai.engine import Engine

TERMINAL = {"checkmate", "stalemate", "draw-fifty", "draw-material", "draw-repetition"}


def random_move(board):
    moves = board.legal_moves()
    return random.choice(moves) if moves else None


def human_move(board):
    while True:
        legal = board.legal_moves()
        print("legal:", " ".join(m.uci() for m in legal))
        uci = input(f"{'White' if board.turn == 0 else 'Black'} to move> ").strip()
        m = board.find_legal(uci)
        if m is not None:
            return m
        print(f"illegal move: {uci}")


def make_player(kind, engine):
    if kind == "human":
        return human_move
    if kind == "random":
        return random_move
    if kind == "engine":
        def _play(board):
            move, score, depth = engine.pick_move(board)
            print(f"  engine: {move.uci() if move else None} (score {score}, depth {depth})", file=sys.stderr)
            return move
        return _play
    raise ValueError(f"unknown player kind: {kind}")


def play_game(white_kind, black_kind, engine, start_fen=None, max_plies=200, verbose=True):
    board = Board()
    if start_fen:
        board.load_fen(start_fen)
    white_fn = make_player(white_kind, engine)
    black_fn = make_player(black_kind, engine)

    for ply in range(max_plies):
        status = board.status()
        if status in TERMINAL:
            if verbose:
                print(f"game over after {ply} plies: {status}")
            return status
        player_fn = white_fn if board.turn == 0 else black_fn
        move = player_fn(board)
        if move is None:
            raise RuntimeError("player returned no move but game is not over")
        if move not in board.legal_moves():
            raise RuntimeError(f"player proposed illegal move: {move.uci()}")
        board.push_uci(move.uci())
        if verbose:
            print(f"ply {ply + 1}: {move.uci()} | {board.status()}")

    if verbose:
        print(f"reached {max_plies}-ply cap, stopping")
    return board.status()


def main():
    parser = argparse.ArgumentParser(description="Play/test the ai/ chess engine.")
    parser.add_argument("--white", choices=("human", "engine", "random"), required=True)
    parser.add_argument("--black", choices=("human", "engine", "random"), required=True)
    parser.add_argument("--time", type=float, required=True, help="engine time budget per move (seconds)")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--eval", choices=("classical", "nnue", "auto"), default="auto")
    parser.add_argument("--fen", type=str, default=None)
    args = parser.parse_args()

    use_nnue = {"classical": "off", "nnue": "on", "auto": "auto"}[args.eval]
    engine = Engine(time_budget=args.time, use_nnue=use_nnue)

    results = []
    for g in range(args.games):
        print(f"=== game {g + 1}/{args.games} ===")
        status = play_game(args.white, args.black, engine, start_fen=args.fen)
        results.append(status)

    print("results:", results)


if __name__ == "__main__":
    main()
