"""Local web viewer for designing a diverging palette on one render.

Loads a grayscale render (a candidate's presentation.png, or any image),
recolors it live as you pick a preset or drag the six OKLab knobs, shows the
print-gamut verdict, and saves a reproducible config (spec + provenance +
preview) with one click.

Stdlib only (http.server) so it runs anywhere; the palette math and the
gamut check live in `palette.py`.
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import palette
from .palette import KNOB_RANGES, PRESETS, PaletteSpec

PAGE = """<!DOCTYPE html>
<html><head><title>playlistviz: palette designer</title>
<style>
  body { background:#111; color:#ddd; font-family:system-ui,sans-serif;
         margin:0; display:flex; height:100vh; }
  #view { flex:1; display:flex; align-items:center; justify-content:center;
          background:#0a0a0a; overflow:hidden; }
  /* nearest-neighbor: smoothing would average adjacent signed entries and
     wash a pixel-exact matrix to flat color */
  #view img { max-width:96%; max-height:96%; image-rendering:pixelated;
              box-shadow:0 0 24px #000; }
  #panel { width:320px; padding:16px 18px; overflow-y:auto;
           background:#181818; box-sizing:border-box; }
  h2 { font-weight:300; margin:.2em 0 .6em; font-size:1.1rem; }
  label { display:block; font-size:.8rem; margin:10px 0 2px; color:#bbb; }
  .row { display:flex; align-items:center; gap:8px; }
  input[type=range] { flex:1; }
  .val { width:44px; text-align:right; font-variant-numeric:tabular-nums;
         color:#9cf; font-size:.8rem; }
  select, button, input[type=text] { width:100%; padding:6px; margin-top:4px;
         background:#222; color:#eee; border:1px solid #333; border-radius:4px;
         box-sizing:border-box; }
  button { cursor:pointer; margin-top:12px; }
  button:hover { border-color:#7af; }
  #gamut { margin-top:14px; padding:8px; border-radius:4px; font-size:.8rem;
           line-height:1.5; }
  .ok { background:#12301a; color:#8f8; }
  .warn { background:#332a12; color:#fd8; }
  .bad { background:#301414; color:#f99; }
  #knobs.disabled { opacity:.4; pointer-events:none; }
  #msg { font-size:.75rem; color:#8a8; margin-top:8px; min-height:1em; }
  code { color:#9cf; }
</style></head>
<body>
<div id="view"><img id="preview"></div>
<div id="panel">
  <h2>palette designer</h2>
  <label>preset</label>
  <select id="preset"></select>
  <div id="knobs"></div>
  <div id="gamut"></div>
  <label>save as</label>
  <input type="text" id="name" value="custom">
  <button id="save">Save config + preview</button>
  <div id="msg"></div>
</div>
<script>
const KNOBS = __KNOBS__;
let spec = null;

function el(id){ return document.getElementById(id); }

function buildKnobs(){
  const box = el('knobs'); box.innerHTML = '';
  for(const k of Object.keys(KNOBS)){
    const [lo,hi,st] = KNOBS[k];
    const row = document.createElement('div');
    row.innerHTML = `<label>${k}</label><div class="row">`+
      `<input type="range" id="k_${k}" min="${lo}" max="${hi}" step="${st}">`+
      `<span class="val" id="v_${k}"></span></div>`;
    box.appendChild(row);
    el(`k_${k}`).oninput = () => { spec.kind='oklab'; readKnobs(); render(); };
  }
}
function setKnobs(s){
  for(const k of Object.keys(KNOBS)){
    if(s[k]===undefined) continue;
    el(`k_${k}`).value = s[k];
    el(`v_${k}`).textContent = (+s[k]).toFixed(2).replace(/\\.?0+$/,'');
  }
  el('knobs').className = s.kind==='oklab' ? '' : 'disabled';
}
function readKnobs(){
  for(const k of Object.keys(KNOBS)){
    spec[k] = parseFloat(el(`k_${k}`).value);
    el(`v_${k}`).textContent = (+spec[k]).toFixed(2).replace(/\\.?0+$/,'');
  }
}
async function loadPresets(){
  const r = await fetch('/presets'); const d = await r.json();
  const sel = el('preset');
  const groups = {};
  for(const p of d.presets){
    const g = p.group || 'Presets';
    (groups[g] = groups[g] || []).push(p);
  }
  for(const g of Object.keys(groups)){
    const og = document.createElement('optgroup'); og.label = g;
    for(const p of groups[g]){
      const o = document.createElement('option'); o.value=p.name;
      o.textContent = p.name;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.onchange = async () => {
    const r = await fetch('/preset?name='+encodeURIComponent(sel.value));
    spec = await r.json(); setKnobs(spec); render();
  };
  spec = d.presets[0]; setKnobs(spec); render();
}
let timer=null;
function render(){
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const r = await fetch('/gamut', {method:'POST', body:JSON.stringify(spec)});
    const g = await r.json();
    el('preview').src = '/render?v='+Date.now()+'&s='+
      encodeURIComponent(JSON.stringify(spec));
    const box = el('gamut');
    // three-state from the tier: faithful/good/unknown -> green, compressed
    // -> amber (usable, accents soften), poor/invalid -> red
    const green = ['faithful','good','unknown'].includes(g.tier);
    const amber = g.tier==='compressed';
    box.className = green ? 'ok' : (amber ? 'warn' : 'bad');
    const head = green ? 'PRINTS FAITHFULLY'
               : amber ? 'PRINTS - accents soften'
               : 'OUT OF PRINT GAMUT';
    let txt = head+' ('+g.tier+')<br>'+
      'method: '+g.method+'<br>'+
      'max OKLab chroma: '+g.max_oklab_chroma.toFixed(3);
    if(g.method==='cmyk-softproof')
      txt += '<br>out-of-gamut: '+(g.out_of_gamut_fraction*100).toFixed(1)+
             '% ('+g.icc+')';
    if(g.srgb_clip_fraction>0)
      txt += '<br>sRGB clip: '+(g.srgb_clip_fraction*100).toFixed(1)+'%';
    box.innerHTML = txt;
  }, 60);
}
el('save').onclick = async () => {
  spec.name = el('name').value || 'custom';
  const r = await fetch('/save', {method:'POST', body:JSON.stringify(spec)});
  const d = await r.json();
  el('msg').innerHTML = d.ok ? 'saved <code>'+d.path+'</code>'
                             : 'error: '+d.error;
};
buildKnobs(); loadPresets();
</script>
</body></html>
"""


class PaletteState:
    """Cached source render + full-res gray, shared by request handlers."""

    def __init__(self, source: Path, out_dir: Path, source_id: str = "",
                 preview_max: int = 640):
        from PIL import Image

        self.source = Path(source)
        self.out_dir = Path(out_dir)
        self.source_id = source_id
        gray = np.asarray(Image.open(self.source).convert("L"),
                          dtype=float) / 255.0
        self.full_gray = gray
        # downsample for live preview (nearest, to preserve pixel-exactness)
        n = max(gray.shape)
        if n > preview_max:
            step = int(np.ceil(n / preview_max))
            gray = gray[::step, ::step]
        self.preview_gray = gray

    def render_png(self, spec: PaletteSpec) -> bytes:
        from PIL import Image

        rgb = palette.apply_spec(self.preview_gray, spec)
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        return buf.getvalue()

    def save(self, spec: PaletteSpec) -> Path:
        return palette.save_config(spec, self.source, self.out_dir,
                                   source_id=self.source_id)


def make_handler(state: PaletteState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _spec_from_body(self) -> PaletteSpec:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n)) if n else {}
            return PaletteSpec.from_dict(d)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/" or u.path.startswith("/index"):
                knobs = json.dumps(KNOB_RANGES)
                self._send(200, PAGE.replace("__KNOBS__", knobs).encode(),
                           "text/html")
            elif u.path == "/presets":
                body = json.dumps({"presets": [p.to_dict()
                                               for p in PRESETS.values()]})
                self._send(200, body.encode(), "application/json")
            elif u.path == "/preset":
                name = parse_qs(u.query).get("name", [""])[0]
                p = PRESETS.get(name)
                if p is None:
                    self._send(404, b"no such preset", "text/plain")
                else:
                    self._send(200, json.dumps(p.to_dict()).encode(),
                               "application/json")
            elif u.path == "/render":
                try:
                    s = parse_qs(u.query).get("s", ["{}"])[0]
                    spec = PaletteSpec.from_dict(json.loads(s))
                    self._send(200, state.render_png(spec), "image/png")
                except Exception as e:  # malformed spec -> 400, keep serving
                    self._send(400, str(e).encode(), "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/gamut":
                try:
                    spec = self._spec_from_body()
                    body = json.dumps(palette.gamut_report(spec)).encode()
                    self._send(200, body, "application/json")
                except Exception as e:
                    self._send(400, str(e).encode(), "text/plain")
            elif self.path == "/save":
                try:
                    spec = self._spec_from_body()
                    path = state.save(spec)
                    body = {"ok": True, "path": str(path)}
                except Exception as e:  # surface to the UI, don't crash server
                    body = {"ok": False, "error": str(e)}
                self._send(200, json.dumps(body).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def serve(source: Path, out_dir: Path, source_id: str = "",
          port: int = 8766) -> None:
    state = PaletteState(source, out_dir, source_id=source_id)
    httpd = HTTPServer(("127.0.0.1", port), make_handler(state))
    print(f"palette designer for {source}")
    print(f"  saves configs to {out_dir}/")
    print(f"  open http://127.0.0.1:{port} (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
