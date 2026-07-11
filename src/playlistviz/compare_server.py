"""Local web UI for pairwise aesthetic comparisons.

Serves two candidate renders side by side; a click (or arrow key) records the
winner to comparisons.jsonl and advances to the next pair. Pairs are chosen
by the BT active-selection rule, refreshed as comparisons accumulate.

Stdlib only (http.server) so it runs anywhere.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from . import bt

PAGE = """<!DOCTYPE html>
<html><head><title>playlistviz: which A is more beautiful?</title>
<style>
  body { background: #111; color: #ddd; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  h2 { font-weight: 300; }
  .pair { display: flex; gap: 24px; }
  /* nearest-neighbor scaling: browser smoothing would average adjacent
     signed matrix entries and cancel pixel-exact candidates to flat gray */
  .pair img { width: 42vw; max-width: 640px; cursor: pointer;
              image-rendering: pixelated;
              border: 3px solid #333; border-radius: 4px; }
  .pair img:hover { border-color: #7af; }
  #status { margin-top: 12px; color: #888; }
  kbd { background: #333; padding: 1px 6px; border-radius: 3px; }
</style></head>
<body>
<h2>Which rendering of A do you prefer?</h2>
<div class="pair">
  <img id="left" title="prefer left">
  <img id="right" title="prefer right">
</div>
<p id="status"></p>
<p>Click an image, or press <kbd>&larr;</kbd> / <kbd>&rarr;</kbd>.
Comparisons are appended to <code>runs/comparisons.jsonl</code>.</p>
<script>
let pair = null;
async function next() {
  const r = await fetch('/pair'); const d = await r.json();
  if (d.done) {
    document.getElementById('status').textContent =
      'No more scheduled pairs (' + d.count + ' comparisons recorded). ' +
      'Run "playlistviz fit" to fit your weights.';
    return;
  }
  pair = d;
  document.getElementById('left').src = '/img/' + d.left + '?v=' + Date.now();
  document.getElementById('right').src = '/img/' + d.right + '?v=' + Date.now();
  document.getElementById('status').textContent =
    d.count + ' comparisons recorded';
}
async function choose(side) {
  if (!pair) return;
  const winner = side === 'left' ? pair.left : pair.right;
  const loser  = side === 'left' ? pair.right : pair.left;
  await fetch('/choice', {method: 'POST',
    body: JSON.stringify({winner: winner, loser: loser})});
  next();
}
document.getElementById('left').onclick = () => choose('left');
document.getElementById('right').onclick = () => choose('right');
document.onkeydown = (e) => {
  if (e.key === 'ArrowLeft') choose('left');
  if (e.key === 'ArrowRight') choose('right');
};
next();
</script>
</body></html>
"""


class CompareState:
    """Pair queue + comparison log, shared by request handlers."""

    def __init__(self, runs_dir: Path, comparisons_path: Path):
        self.runs_dir = Path(runs_dir)
        self.comparisons_path = Path(comparisons_path)
        self.blocks: dict[str, str] = {}
        self.loss_vectors = self._load_loss_vectors()
        self.queue: list[tuple[str, str]] = []
        self._refill()

    def _load_loss_vectors(self) -> dict[str, np.ndarray]:
        from .loss import loss_vector_from_phi_dict

        from .pool import iter_candidate_files

        out = {}
        for f in iter_candidate_files(self.runs_dir):
            d = json.loads(f.read_text())
            # rebuilt from phi by name: robust to metric-set changes
            out[d["id"]] = loss_vector_from_phi_dict(d["phi"])
            # blocked design: pairs only within one song, so each choice
            # compares parameter settings on identical content
            self.blocks[d["id"]] = d.get("song", "")
        return out

    def _seen_pairs(self) -> set[frozenset]:
        comps = bt.load_comparisons(self.comparisons_path)
        return {frozenset((c.winner, c.loser)) for c in comps}

    def _refill(self) -> None:
        comps = bt.load_comparisons(self.comparisons_path)
        fit = None
        if len(comps) >= 5:
            try:
                fit = bt.fit_bt(comps, self.loss_vectors)
            except (ValueError, KeyError):
                fit = None
        self.queue = bt.select_pairs(
            self.loss_vectors, fit, n_pairs=10,
            exclude=self._seen_pairs(),
            rng=np.random.default_rng(len(comps)),
            blocks=self.blocks,
        )

    def next_pair(self) -> tuple[str, str] | None:
        if not self.queue:
            self._refill()
        return self.queue.pop(0) if self.queue else None

    def record(self, winner: str, loser: str) -> None:
        bt.append_comparison(self.comparisons_path,
                             bt.Comparison(winner=winner, loser=loser))

    def count(self) -> int:
        return len(bt.load_comparisons(self.comparisons_path))

    def image_path(self, cand_id: str) -> Path | None:
        from .pool import candidate_dir

        d = candidate_dir(self.runs_dir, cand_id)
        if d is None:
            return None
        p = d / "presentation.png"
        return p if p.exists() else None


def make_handler(state: CompareState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, PAGE.encode(), "text/html")
            elif self.path.startswith("/pair"):
                pair = state.next_pair()
                if pair is None:
                    body = json.dumps({"done": True, "count": state.count()})
                else:
                    body = json.dumps({"left": pair[0], "right": pair[1],
                                       "count": state.count(), "done": False})
                self._send(200, body.encode(), "application/json")
            elif self.path.startswith("/img/"):
                cand_id = self.path.split("/img/")[1].split("?")[0]
                p = state.image_path(cand_id)
                if p is None:
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, p.read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/choice":
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n))
                state.record(d["winner"], d["loser"])
                self._send(200, b"{}", "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def serve(runs_dir: Path, comparisons_path: Path, port: int = 8765) -> None:
    state = CompareState(runs_dir, comparisons_path)
    if not state.loss_vectors:
        raise SystemExit("no candidates found in runs/ - run `playlistviz optimize` first")
    httpd = HTTPServer(("127.0.0.1", port), make_handler(state))
    print(f"{len(state.loss_vectors)} candidates loaded.")
    print(f"Open http://127.0.0.1:{port} and start comparing (Ctrl-C to stop).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
