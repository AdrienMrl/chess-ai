#!/usr/bin/env python3
"""Drive the running server via its REST API: one random legal move every N sec.
Usage: python3 autoplay.py [interval_seconds]  (default 10)
Bounded — stops at checkmate/stalemate/draw or after 120 plies, then exits."""
import json
import random
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
INTERVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 10


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


state = call("/reset", {})
print("reset:", state["fen"], flush=True)

for ply in range(1, 121):
    legal = state["legal"]
    if not legal or state["status"] in ("checkmate", "stalemate",
                                        "draw-fifty", "draw-material"):
        print(f"game over after {ply-1} plies: {state['status']}", flush=True)
        break
    # prefer captures/checks a bit for livelier games, else random
    move = random.choice(legal)
    state = call("/move", {"uci": move})
    print(f"ply {ply}: {state['turn']} to move | played {move} | {state['status']}",
          flush=True)
    time.sleep(INTERVAL)
else:
    print("reached 120-ply cap, stopping", flush=True)
