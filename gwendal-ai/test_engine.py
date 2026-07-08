#!/usr/bin/env python3
"""
Standalone correctness tests for chess_game.py / engine.py.
No pytest dependency -- plain unittest.

Run: python3 test_engine.py
"""
import time
import unittest

from chess_game import Board, perft
from engine import Engine

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
TACTICAL_MIDDLEGAME = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
ENDGAME = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
IN_CHECK = "rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq g3 0 2"


class TestPerft(unittest.TestCase):
    def test_startpos_depth4(self):
        b = Board()
        self.assertEqual(perft(b, 4), 197281)

    def test_kiwipete_depth3(self):
        b = Board()
        b.load_fen(KIWIPETE)
        self.assertEqual(perft(b, 3), 97862)


class TestEngineLegalMoves(unittest.TestCase):
    POSITIONS = [
        ("startpos", None),
        ("kiwipete", KIWIPETE),
        ("tactical_middlegame", TACTICAL_MIDDLEGAME),
        ("endgame", ENDGAME),
        ("in_check", IN_CHECK),
    ]

    def test_engine_returns_legal_move(self):
        for name, fen in self.POSITIONS:
            with self.subTest(position=name):
                b = Board()
                if fen:
                    b.load_fen(fen)
                eng = Engine()
                mv = eng.choose_move(b, time_limit=0.7)
                self.assertIsNotNone(mv, f"engine returned None for {name}")
                legal_ucis = {m.uci() for m in b.legal_moves()}
                self.assertIn(mv, legal_ucis,
                              f"{name}: engine move {mv} not in legal set {legal_ucis}")


class TestTimeSafety(unittest.TestCase):
    def test_respects_time_limit(self):
        for name, fen, tl in [
            ("startpos", None, 1.0),
            ("kiwipete", KIWIPETE, 1.0),
            ("tactical_middlegame", TACTICAL_MIDDLEGAME, 1.5),
        ]:
            with self.subTest(position=name):
                b = Board()
                if fen:
                    b.load_fen(fen)
                eng = Engine()
                t0 = time.time()
                eng.choose_move(b, time_limit=tl)
                elapsed = time.time() - t0
                self.assertLess(
                    elapsed, tl + 1.0,
                    f"{name}: choose_move took {elapsed:.2f}s for time_limit={tl}s"
                )


class TestTerminalPositions(unittest.TestCase):
    def test_checkmate_returns_none(self):
        b = Board()
        b.load_fen(FOOLS_MATE)
        self.assertEqual(b.status(), "checkmate")
        eng = Engine()
        mv = eng.choose_move(b, time_limit=0.5)
        self.assertIsNone(mv)

    def test_stalemate_returns_none(self):
        b = Board()
        b.load_fen(STALEMATE)
        self.assertEqual(b.status(), "stalemate")
        eng = Engine()
        mv = eng.choose_move(b, time_limit=0.5)
        self.assertIsNone(mv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
