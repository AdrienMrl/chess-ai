# Gwendal's chess AI

A from-scratch Python chess engine (no chess/AI libraries — only the project's
own rules referee is reused, in-process, for move generation/legality).

## Architecture

- **Search**: negamax with alpha-beta pruning, iterative deepening under a
  wall-clock time budget, a Zobrist-hashed transposition table (with
  mate-distance-adjusted scores and depth-preferred replacement), quiescence
  search on captures, null-move pruning, and check extensions (capped) so
  forcing sequences aren't cut off mid-tactic.
- **Move ordering**: TT move → MVV-LVA captures → killer moves → history
  heuristic.
- **Evaluation** (`eval_extra.py` + `engine.py`): tapered material + piece-square
  tables (separate middlegame/endgame king tables interpolated by material
  phase), bishop-pair bonus, pawn structure (doubled/isolated/passed pawns),
  rook activity (open/semi-open file, 7th rank), cheap knight/bishop mobility,
  and king safety (pawn shield + open-file-near-king penalty, phase-aware).
- **Move generation** (`chess_game.py`): the project's referee, sped up with a
  pin-aware fast-legality path (skips make/unmake verification for
  non-pinned, non-king, non-en-passant moves when not in check) — validated
  byte-identical against the original via perft on multiple positions
  (startpos depth 5, Kiwipete depth 4, a dedicated pin position), ~2.6x
  faster move generation.

## Usage

```bash
python3 play.py self 8.0      # self-play demo, 8s/move
python3 play.py white 8.0     # play as White against manual stdin input for the opponent
python3 -m unittest test_engine -v   # correctness tests (perft, legality, time budget, terminal positions)
```

Entry point for integration: `engine.choose_move(board, time_limit)` (takes a
`chess_game.Board`), or `engine.choose_move_from_fen(fen, time_limit)` for a
FEN-only interface matching the referee's documented protocol.

## Testing notes

Benchmarked against real Stockfish (skill-limited) via a custom UCI test
harness. Roughly 1650-1900 Elo at reduced test time controls; expected higher
at full match time budgets (10s/move) given the depth gained from search
improvements made after initial calibration.
