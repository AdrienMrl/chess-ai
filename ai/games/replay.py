#!/usr/bin/env python3
"""Replay a 'move <uci>' game file into the running server, 1s between moves.

Usage: python3 ai/games/replay.py ai/games/self_play_classical_001.txt
"""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def main():
    path = sys.argv[1]
    moves = [l.split()[1] for l in open(path) if l.startswith("move ")]
    call("/reset", {})
    for uci in moves:
        state = call("/move", {"uci": uci})
        if "error" in state:
            print("ERROR at", uci, state["error"])
            return
        print(uci, "|", state["status"])
        time.sleep(1)
    print("done:", state["status"])


if __name__ == "__main__":
    main()
