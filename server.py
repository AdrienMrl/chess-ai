#!/usr/bin/env python3
"""
Web UI + HTTP command API for the chess engine in chess_game.py.

Run:
    python3 server.py            # serves on http://localhost:8000
    python3 server.py 9000       # custom port

Browser UI: open the URL, click a piece then a destination to move.

HTTP API (for driving from a chess AI):
    GET  /state                 -> full game state as JSON
    POST /move    {"uci":"e2e4"} -> make a move
    POST /load    {"fen":"..."}  -> set position
    POST /undo                   -> undo last move
    POST /reset                  -> new game

Every JSON response includes: fen, turn, status, legal (list of uci),
history (list of uci), board (8 rows, rank 8 first), last (last move uci).
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chess_game import Board, sq, FILES

board = Board()


def state():
    grid = []
    for rank in range(7, -1, -1):
        grid.append([board.squares[sq(f, rank)] for f in range(8)])
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == 0 else "black",
        "status": board.status(),
        "legal": [m.uci() for m in board.legal_moves()],
        "history": list(board.uci_history),
        "board": grid,
        "last": board.uci_history[-1] if board.uci_history else None,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess AI Tester</title>
<style>
  :root{
    --light:#ebecd0; --dark:#739552; --sel:#f6f669; --dot:rgba(0,0,0,.18);
    --bg:#302e2b; --panel:#262421; --text:#e8e6e3; --muted:#9a9791; --accent:#7fa650;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
    display:flex;flex-wrap:wrap;gap:28px;justify-content:center;align-items:flex-start;
    padding:28px;min-height:100vh}
  h1{font-size:19px;margin:0 0 14px;font-weight:600}
  .board{display:grid;grid-template-columns:repeat(8,64px);grid-template-rows:repeat(8,64px);
    border-radius:6px;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.5);
    -webkit-user-select:none;user-select:none}
  .sq{width:64px;height:64px;display:flex;align-items:center;justify-content:center;
    font-size:46px;position:relative;cursor:pointer;line-height:1}
  .sq.light{background:var(--light)} .sq.dark{background:var(--dark)}
  .sq.sel{background:var(--sel)!important}
  .sq.last::after{content:"";position:absolute;inset:0;background:var(--sel);opacity:.35}
  .sq .glyph{position:relative;z-index:2}
  .sq.move::before{content:"";position:absolute;width:26%;height:26%;border-radius:50%;
    background:var(--dot);z-index:1}
  .sq.capture::before{content:"";position:absolute;inset:6%;border-radius:50%;
    border:5px solid var(--dot);width:auto;height:auto;z-index:1}
  .coord{position:absolute;font-size:11px;font-weight:600;opacity:.6;z-index:3}
  .coord.file{right:4px;bottom:2px} .coord.rank{left:3px;top:1px}
  .panel{background:var(--panel);border-radius:10px;padding:20px;width:300px;
    box-shadow:0 8px 30px rgba(0,0,0,.4)}
  .stat{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #3a3733}
  .stat span:first-child{color:var(--muted)}
  .status-badge{font-weight:600}
  .status-badge.check,.status-badge.checkmate{color:#e88}
  .status-badge.stalemate,.status-badge[class*="draw"]{color:#e8c069}
  .btns{display:flex;gap:8px;margin:16px 0}
  button{flex:1;background:#3a3733;color:var(--text);border:0;border-radius:6px;
    padding:9px;font-size:14px;cursor:pointer;font-weight:500}
  button:hover{background:var(--accent);color:#1a1a1a}
  .moves{margin-top:8px;max-height:220px;overflow:auto;font:13px/1.6 ui-monospace,Menlo,monospace;
    color:var(--muted);white-space:pre-wrap;word-break:break-word}
  .fen{margin-top:12px;font:11px/1.4 ui-monospace,Menlo,monospace;color:var(--muted);
    word-break:break-all;background:#1c1a18;padding:8px;border-radius:5px}
  .api{margin-top:14px;font-size:12px;color:var(--muted)}
  .api code{background:#1c1a18;padding:1px 5px;border-radius:3px;color:#cbd5b8}
</style>
</head>
<body>
  <div>
    <h1>♟ Chess AI Tester</h1>
    <div class="board" id="board"></div>
  </div>
  <div class="panel">
    <div class="stat"><span>Turn</span><b id="turn">white</b></div>
    <div class="stat"><span>Status</span><b class="status-badge" id="status">ongoing</b></div>
    <div class="stat"><span>Moves</span><b id="movecount">0</b></div>
    <div class="btns">
      <button onclick="act('reset')">New game</button>
      <button onclick="act('undo')">Undo</button>
    </div>
    <div class="moves" id="moves"></div>
    <div class="fen" id="fen"></div>
    <div class="api">
      Drive from an AI via HTTP:<br>
      <code>POST /move {"uci":"e2e4"}</code><br>
      <code>GET /state</code>
    </div>
  </div>

<script>
const GLYPH={K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"};
let S=null, sel=null, legalFrom=[];

function idx(f,r){return (7-r)*8+f;}         // grid index for file f, rank r (0-based)
function nameOf(f,r){return "abcdefgh"[f]+(r+1);}

async function api(path, body){
  const opt = body? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}
                   : {};
  const res = await fetch(path, opt);
  return res.json();
}
async function refresh(){ S = await api("/state"); render(); }
async function act(cmd){ S = await api("/"+cmd, {}); sel=null; render(); }

function render(){
  const b=document.getElementById("board"); b.innerHTML="";
  for(let r=7;r>=0;r--){
    for(let f=0;f<8;f++){
      const d=document.createElement("div");
      const dark=(f+r)%2===0;
      d.className="sq "+(dark?"dark":"light");
      const piece=S.board[7-r][f];
      const name=nameOf(f,r);
      if(sel){
        if(name===sel) d.classList.add("sel");
        if(legalFrom.some(m=>m.slice(2,4)===name)){
          d.classList.add(piece && piece!=="." ? "capture":"move");
        }
      }
      if(S.last && (S.last.slice(0,2)===name || S.last.slice(2,4)===name))
        d.classList.add("last");
      if(piece && piece!=="."){
        const g=document.createElement("span"); g.className="glyph";
        g.textContent=GLYPH[piece]; g.style.color = piece===piece.toUpperCase()?"#fff":"#111";
        g.style.textShadow = piece===piece.toUpperCase()?"0 1px 2px rgba(0,0,0,.4)":"none";
        d.appendChild(g);
      }
      if(f===0){const c=document.createElement("span");c.className="coord rank";c.textContent=r+1;
        c.style.color=dark?"var(--light)":"var(--dark)";d.appendChild(c);}
      if(r===0){const c=document.createElement("span");c.className="coord file";c.textContent="abcdefgh"[f];
        c.style.color=dark?"var(--light)":"var(--dark)";d.appendChild(c);}
      d.onclick=()=>click(f,r,name);
      b.appendChild(d);
    }
  }
  document.getElementById("turn").textContent=S.turn;
  const st=document.getElementById("status");
  st.textContent=S.status; st.className="status-badge "+S.status;
  document.getElementById("movecount").textContent=S.history.length;
  document.getElementById("fen").textContent=S.fen;
  // numbered move list
  let out="";
  for(let i=0;i<S.history.length;i+=2){
    out+=(i/2+1)+". "+S.history[i]+" "+(S.history[i+1]||"")+"  ";
  }
  document.getElementById("moves").textContent=out.trim();
}

async function click(f,r,name){
  if(sel){
    // try to move sel -> name (auto-queen on promotion)
    let uci=sel+name;
    const promo=legalFrom.find(m=>m.startsWith(sel) && m.slice(2,4)===name && m.length===5);
    if(promo) uci=promo;                   // uses whatever promo (defaults q) engine listed
    const cand=legalFrom.find(m=>m===uci || m.slice(0,4)===sel+name);
    if(cand){
      const res=await api("/move",{uci:cand.slice(0,5)||cand});
      if(!res.error){ S=res; sel=null; render(); return; }
    }
    sel=null; legalFrom=[];
    // fall through: allow selecting a new piece
  }
  const piece=S.board[7-r][f];
  if(piece && piece!=="."){
    sel=name;
    legalFrom=S.legal.filter(m=>m.slice(0,2)===name);
  }
  render();
}

refresh();
// auto-poll so the board reflects moves made via the API (e.g. by an AI driver).
// pause polling while the user has a piece selected, to not clobber their click.
setInterval(()=>{ if(!sel) refresh(); }, 1000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/state":
            self._send(200, json.dumps(state()))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        try:
            if self.path == "/move":
                uci = payload.get("uci", "")
                if not board.push_uci(uci):
                    self._send(400, json.dumps({"error": f"illegal move: {uci}", **state()}))
                    return
            elif self.path == "/load":
                board.load_fen(payload.get("fen", ""))
            elif self.path == "/undo":
                board.undo()
            elif self.path == "/reset":
                board.reset()
            else:
                self._send(404, json.dumps({"error": "not found"}))
                return
        except Exception as e:
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}", **state()}))
            return

        self._send(200, json.dumps(state()))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Chess UI running at http://localhost:{port}")
    print("HTTP API: GET /state | POST /move {\"uci\":\"e2e4\"} | POST /reset | POST /undo | POST /load")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
