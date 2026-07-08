"""Lightweight verification / benchmark script for the chess engine + search.

Not a full test suite - a fast smoke test meant to catch obvious regressions:
  1. Tactical spot-checks   - a handful of hand-verified positions with a known
                               correct move (mate-in-1s + one hanging-piece grab).
  2. Self-play no-crash     - engine plays several games against a random
                               mover, asserting every engine move is legal and
                               every game ends in a status from the known
                               vocabulary.
  3. Time-budget compliance - spot-checks that search_root/Engine.pick_move
                               never blows past the requested time budget by
                               more than a small slack, i.e. the TimeUp abort
                               mechanism actually works.

Run from the repo root:
    cd /Users/adri/dev/chess-ai && python3 -m ai.bench

Exits 0 if everything passes, 1 otherwise (usable as a CI/smoke-test gate).
"""
import random
import sys
import time

from chess_game import Board, WHITE
from ai.eval import evaluate
from ai.engine import Engine
from ai.search import search_root

VALID_STATUSES = {
    "ongoing",
    "check",
    "checkmate",
    "stalemate",
    "draw-fifty",
    "draw-repetition",
    "draw-material",
}

TIME_SLACK_S = 0.5  # search checks the deadline between nodes, not instantaneously

# --- Section 1: tactical spot-checks ---------------------------------------
# Every entry below was verified in-session by actually running search_root
# on the FEN and confirming the resulting move produces the claimed outcome
# (checkmate, or capturing the named hanging piece).
TACTICAL_TIME_BUDGET_S = 2.0
TACTICAL_POSITIONS = [
    {
        "name": "back-rank mate (Ra8#)",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "expected_move": "a1a8",
        "expect_checkmate": True,
    },
    {
        "name": "fool's mate (Qh4#)",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
        "expected_move": "d8h4",
        "expect_checkmate": True,
    },
    {
        "name": "smothered mate (Nf7#)",
        "fen": "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
        "expected_move": "g5f7",
        "expect_checkmate": True,
    },
    {
        "name": "hanging queen (Qxd4)",
        "fen": "4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1",
        "expected_move": "d2d4",
        "expect_checkmate": False,
    },
]

# --- Section 2: self-play no-crash harness ----------------------------------
SELF_PLAY_GAMES = 4
SELF_PLAY_TIME_BUDGET_S = 0.5
SELF_PLAY_PLY_CAP = 150
SELF_PLAY_SEED = 1234

# --- Section 3: time-budget compliance --------------------------------------
TIME_TEST_BUDGETS_S = (0.3, 1.0)
TIME_TEST_RANDOM_PLIES = 6
TIME_TEST_SEED = 99


def run_tactical_checks():
    print("=== Tactical spot-checks ===")
    all_ok = True
    passed = 0
    for case in TACTICAL_POSITIONS:
        board = Board()
        board.load_fen(case["fen"])
        move, score, depth = search_root(board, TACTICAL_TIME_BUDGET_S, evaluate)
        found_uci = move.uci() if move is not None else None
        ok = found_uci == case["expected_move"]
        detail = ""
        if ok and case["expect_checkmate"]:
            play_board = Board()
            play_board.load_fen(case["fen"])
            play_board.push_uci(found_uci)
            result_status = play_board.status()
            ok = result_status == "checkmate"
            detail = f", resulting status={result_status}"
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [{tag}] {case['name']}: expected {case['expected_move']}, "
            f"got {found_uci} (score={score}, depth={depth}){detail}"
        )
        all_ok = all_ok and ok
        passed += 1 if ok else 0
    print(f"  {passed}/{len(TACTICAL_POSITIONS)} tactical checks passed")
    print()
    return all_ok


def run_selfplay():
    print("=== Self-play no-crash test ===")
    all_ok = True
    rng = random.Random(SELF_PLAY_SEED)
    engine_timings = []  # (budget, elapsed) for every engine.pick_move call

    for game_idx in range(SELF_PLAY_GAMES):
        engine_is_white = (game_idx % 2 == 0)
        board = Board()
        engine = Engine(SELF_PLAY_TIME_BUDGET_S, "off")
        game_ok = True
        ply = 0

        while ply < SELF_PLAY_PLY_CAP:
            legal = board.legal_moves()
            if not legal:
                break

            engine_turn = (board.turn == WHITE) == engine_is_white
            if engine_turn:
                t0 = time.monotonic()
                move, score, depth = engine.pick_move(board)
                elapsed = time.monotonic() - t0
                engine_timings.append((SELF_PLAY_TIME_BUDGET_S, elapsed))

                if move is None:
                    # engine agrees no legal moves exist; loop condition above
                    # already confirmed legal moves exist, so this is a bug.
                    print(f"  [FAIL] game {game_idx}: engine returned None with legal moves available")
                    game_ok = False
                    break

                legal_ucis = {m.uci() for m in legal}
                if move.uci() not in legal_ucis:
                    print(f"  [FAIL] game {game_idx}: engine returned illegal move {move.uci()}")
                    game_ok = False
                    break

                if not board.push_uci(move.uci()):
                    print(f"  [FAIL] game {game_idx}: push_uci rejected engine move {move.uci()}")
                    game_ok = False
                    break
            else:
                move = rng.choice(legal)
                board.push_uci(move.uci())

            ply += 1

        status = board.status()
        status_ok = status in VALID_STATUSES
        if not status_ok:
            print(f"  [FAIL] game {game_idx}: status {status!r} not in known vocabulary")

        hit_cap = ply >= SELF_PLAY_PLY_CAP
        game_pass = game_ok and status_ok
        tag = "PASS" if game_pass else "FAIL"
        cap_note = " (hit ply cap)" if hit_cap else ""
        print(
            f"  [{tag}] game {game_idx} (engine={'white' if engine_is_white else 'black'}): "
            f"{ply} plies, status={status}{cap_note}"
        )
        all_ok = all_ok and game_pass

    print(f"  {SELF_PLAY_GAMES} games completed")
    print()
    return all_ok, engine_timings


def run_time_compliance(selfplay_timings):
    print("=== Time-budget compliance ===")
    all_ok = True

    if selfplay_timings:
        over_budget = [(b, e) for b, e in selfplay_timings if e > b + TIME_SLACK_S]
        worst_budget, worst_elapsed = max(selfplay_timings, key=lambda t: t[1] - t[0])
        ok = not over_budget
        tag = "PASS" if ok else "FAIL"
        print(
            f"  [{tag}] {len(selfplay_timings)} self-play engine moves within budget+{TIME_SLACK_S}s "
            f"(worst: budget={worst_budget}s elapsed={worst_elapsed:.3f}s)"
        )
        all_ok = all_ok and ok
    else:
        print("  [FAIL] no self-play timing samples were recorded")
        all_ok = False

    # A couple of dedicated fixed positions across a few time budgets, to
    # directly stress the TimeUp abort path independent of self-play.
    rng = random.Random(TIME_TEST_SEED)
    mid_board = Board()
    for _ in range(TIME_TEST_RANDOM_PLIES):
        legal = mid_board.legal_moves()
        if not legal:
            break
        mid_board.push_uci(rng.choice(legal).uci())

    fixed_positions = [
        ("start position", Board().fen()),
        ("randomized mid-game", mid_board.fen()),
    ]

    for budget in TIME_TEST_BUDGETS_S:
        for name, fen in fixed_positions:
            b = Board()
            b.load_fen(fen)
            t0 = time.monotonic()
            move, score, depth = search_root(b, budget, evaluate)
            elapsed = time.monotonic() - t0
            ok = elapsed <= budget + TIME_SLACK_S
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name} budget={budget}s -> elapsed={elapsed:.3f}s (depth={depth})")
            all_ok = all_ok and ok

    print()
    return all_ok


def main():
    start = time.monotonic()

    tactical_ok = run_tactical_checks()
    selfplay_ok, selfplay_timings = run_selfplay()
    time_ok = run_time_compliance(selfplay_timings)

    total_elapsed = time.monotonic() - start
    overall_ok = tactical_ok and selfplay_ok and time_ok

    print("=== Summary ===")
    print(f"  tactical spot-checks:   {'PASS' if tactical_ok else 'FAIL'}")
    print(f"  self-play no-crash:     {'PASS' if selfplay_ok else 'FAIL'}")
    print(f"  time-budget compliance: {'PASS' if time_ok else 'FAIL'}")
    print(f"  total runtime: {total_elapsed:.2f}s")
    print(f"  OVERALL: {'PASS' if overall_ok else 'FAIL'}")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
