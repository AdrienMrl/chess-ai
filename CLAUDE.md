# chess-ai

A zero-dependency chess engine with a line-based command interface, built for **testing chess AIs**. Drive it as a subprocess: write commands to stdin, read one response line per command.

- Engine: `chess_game.py` (pure Python 3 stdlib, no install needed)
- Verified correct via perft (start position + Kiwipete match reference node counts exactly)

## Running

```bash
python3 chess_game.py            # interactive REPL
printf 'move e2e4\nmove e7e5\nfen\nquit\n' | python3 chess_game.py   # piped
```

## Command reference

| Command | Effect | Response |
|---|---|---|
| `move <uci>` | Make a move | `ok <uci> <fen> \| <status>` or `err illegal move: <uci>` |
| `legal` | All legal moves for side to move | `ok e2e4 e2e3 g1f3 …` |
| `legal <sq>` | Legal moves from one square | `ok e2e4 e2e3` |
| `board` | ASCII board | `ok` then 9 lines |
| `fen` | Current position | `ok <fen>` |
| `load <fen>` | Set position from FEN | `ok <fen>` |
| `turn` | Side to move | `ok white` / `ok black` |
| `status` | Game state | `ok ongoing` \| `check` \| `checkmate` \| `stalemate` \| `draw-fifty` \| `draw-repetition` \| `draw-material` |
| `moves` | Move history | `ok e2e4 e7e5 …` |
| `undo` | Undo last move | `ok undone` / `err nothing to undo` |
| `reset` | Starting position | `ok <fen>` |
| `perft <n>` | Count leaf nodes at depth n (self-test) | `ok perft(n)=…` |
| `help` | Show help | — |
| `quit` / `exit` | Exit | `ok bye` |

## Move format

UCI coordinate notation: `<from><to>[promotion]`
- `e2e4`, `g1f3` — normal moves
- `e1g1` — castling (king's move; the rook is moved automatically)
- `e5d6` — en passant (given as the destination square)
- `e7e8q` — promotion to queen (`q`/`r`/`b`/`n`). Bare `e7e8` defaults to queen.

## Protocol contract (for AI harnesses)

- Every command returns **exactly one line** starting with `ok ` or `err `, **except** `board` (prints `ok` then the grid) and `help`.
- After a successful `move`, the response includes the resulting FEN and status, so you rarely need a follow-up `fen`/`status` call.
- The engine never makes moves on its own — it's a rules referee. Your AI(s) supply every move.

## Typical test loop

```python
import subprocess

def start():
    return subprocess.Popen(
        ["python3", "chess_game.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )

def send(p, cmd):
    p.stdin.write(cmd + "\n"); p.stdin.flush()
    return p.stdout.readline().strip()

p = start()
while True:
    legal = send(p, "legal").removeprefix("ok ").split()
    if not legal:                       # no legal moves -> game over
        break
    move = your_ai_pick(legal)          # AI chooses from the legal list
    resp = send(p, f"move {move}")      # ok <uci> <fen> | <status>
    if resp.endswith(("checkmate", "stalemate")) or "draw" in resp:
        break
send(p, "quit")
```

To pit two AIs against each other, alternate which AI is called each turn based on `turn`, feeding each the current `fen`.

## Notes / limitations

- `draw-repetition` is listed in the status vocabulary but threefold-repetition tracking is not yet implemented; fifty-move (`draw-fifty`) and insufficient-material (`draw-material`) are.
- The engine validates strict legality (pins, check evasion, castling through/into check all handled correctly).
