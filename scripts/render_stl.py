"""Render relief STLs to PNG with f3d (headless, EGL) and a synthetic sun HDRI.

f3d has no light-direction flag and no shadows, so lighting comes from a
generated equirectangular Radiance .hdr: a bright sun blob at a chosen
direction plus dim ambient, with the headlight off and screen-space ambient
occlusion on. Verified with f3d 3.5.0 from `nix run nixpkgs#f3d` (first
fetch ~5.5 min, cached afterwards). Wall time per 2400 px view, 2 M triangle
plate: ~2 s; 4 M triangle bed: ~3 s; one strip: ~1 s.

Commands that produced runs/plate/armed_man_p2/render/*.png:
  python scripts/render_stl.py runs/plate/armed_man_p2/plate.stl --frame plate \
      --views front,threequarter,closeup
  python scripts/render_stl.py runs/plate/armed_man_p2/bed.stl --frame bed \
      --views bed
  python scripts/render_stl.py runs/plate/armed_man_p2/strips/B3.stl --frame bed \
      --views strip:front,back --size 2400x1000 --prefix strip_B3_ --out runs/plate/armed_man_p2/render
The underlying f3d call (front view) is:
  f3d plate.stl --output front.png --resolution 2400,2400 --up +Z \
      --hdri-file sun.hdr --hdri-ambient --hdri-skybox=0 --light-intensity 0 \
      --color 0.75,0.75,0.75 --roughness 0.85 --metallic 0 -q -a \
      --camera-direction=0,0,-1 --camera-view-up=0,1,0 --camera-orthographic
Binary: env LOOPVIZ_F3D, else `nix run nixpkgs#f3d --`.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Camera basis: r = image right (+X), u = in-plane up, n = relief normal (toward viewer).
FRAMES = {"plate": ((0, 0, 1), (0, 1, 0)), "bed": ((0, -1, 0), (0, 0, 1))}  # relief normal, in-plane up
# name: (camera dir as a*r + b*u + c*(-n), sun dir as a*r + b*u + c*n, zoom, ortho)
VIEWS = {
    "front":        ((0, 0, 1),            (-0.707, 0.707, 0.466), 0.95, True),   # straight on, upper-left sun 25 deg
    "threequarter": ((0.5, 0.65, 0.574),   (-0.6, 0.55, 0.6),      1.0, False),   # 35 deg above the plate plane
    "closeup":      ((0, 0, 1),            (-0.707, 0.707, 0.21),  1.0, False),   # 100 mm window, grazing 12 deg sun
    "bed":          ((0.35, -0.5, 0.8),    (-0.6, 0.55, 0.6),      1.3, False),   # standing strips, from above
    "strip":        ((0.25, -0.35, 0.9),   (-0.6, 0.55, 0.6),      0.9, False),   # one standing strip, from above
    "back":         ((-0.25, -0.35, -0.9), (0.7, 0.35, -0.6),      1.0, False),   # engraved label side
}


def write_hdr(path, sun_dir, w=1024, h=512, sun=25.0, radius=10.0, ambient=0.04, sky=0.3):
    """Equirectangular Radiance RGBE map: ambient + sky gradient + gaussian sun blob.
    f3d/VTK mirrors X of the naive lat-long convention; calibrated on a lit cube."""
    v, u = np.mgrid[0:h, 0:w]
    phi, theta = (u + 0.5) / w * 2 * np.pi, (v + 0.5) / h * np.pi
    d = np.stack([-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], -1)
    s = np.asarray(sun_dir, float) / np.linalg.norm(sun_dir)
    ang = np.degrees(np.arccos(np.clip(d @ s, -1, 1)))
    L = ambient + sky * np.clip(d[..., 2], 0, 1) + sun * np.exp(-(ang / radius) ** 2)
    e = np.floor(np.log2(np.maximum(L, 1e-32))) + 1
    rgbe = np.zeros((h, w, 4), np.uint8)
    rgbe[..., :3] = np.clip(L * (256.0 / np.exp2(e)), 0, 255).astype(np.uint8)[..., None]
    rgbe[..., 3] = (e + 128).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
        f.write(rgbe.tobytes())


def stl_bounds(path):
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        v = np.frombuffer(f.read(n * 50), np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))["v"]
    v = v.reshape(-1, 3)
    return v.min(0), v.max(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stl")
    ap.add_argument("--out", default=None, help="output dir (default: <stl dir>/render)")
    ap.add_argument("--views", default="front,threequarter,closeup", help="comma list of " + ",".join(VIEWS) + "; view:name renames the file")
    ap.add_argument("--size", default="2400", help="W or WxH pixels")
    ap.add_argument("--frame", default="plate", choices=FRAMES, help="plate: relief +Z, bed: relief -Y (strips standing)")
    ap.add_argument("--closeup-mm", type=float, default=100.0)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--tool", default="f3d", choices=["f3d"])
    a = ap.parse_args()
    stl = Path(a.stl).resolve()
    out = Path(a.out) if a.out else stl.parent / "render"
    out.mkdir(parents=True, exist_ok=True)
    w, h = (a.size.split("x") + [a.size])[:2]
    f3d = shlex.split(os.environ.get("LOOPVIZ_F3D", "nix run nixpkgs#f3d --"))
    n, u = (np.array(x, float) for x in FRAMES[a.frame])
    r = np.cross(u, n)
    lo, hi = stl_bounds(stl)
    centre = (lo + hi) / 2
    face_centre = np.where(n > 0, hi, np.where(n < 0, lo, centre))
    for spec in a.views.split(","):
        name, _, fname = spec.partition(":")  # view[:filename], e.g. strip:front
        cd, sd, zoom, ortho = VIEWS[name]
        cam = cd[0] * r + cd[1] * u - cd[2] * n
        sun = sd[0] * r + sd[1] * u + sd[2] * n
        vup = u if name in ("front", "closeup") else np.array([0, 0, 1.0])  # oblique views: world up
        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as t:
            write_hdr(t.name, sun)
        png = out / f"{a.prefix}{fname or name}.png"
        cmd = f3d + [str(stl), "--output", str(png), "--resolution", f"{w},{h}", "--up", "+Z",
                     "--hdri-file", t.name, "--hdri-ambient", "--hdri-skybox=0", "--light-intensity", "0",
                     "--color", "0.75,0.75,0.75", "--roughness", "0.85", "--metallic", "0", "-q", "-a",
                     "--background-color=#202020", "--camera-view-up=" + ",".join(f"{x:g}" for x in vup)]
        if name == "closeup":
            dist = a.closeup_mm / (2 * np.tan(np.radians(15)))
            pos = face_centre + n * dist
            cmd += ["--camera-position=" + ",".join(f"{x:.2f}" for x in pos),
                    "--camera-focal-point=" + ",".join(f"{x:.2f}" for x in face_centre), "--camera-view-angle=30"]
        else:
            cmd += ["--camera-direction=" + ",".join(f"{x:.3f}" for x in cam), f"--camera-zoom-factor={zoom}"]
            if ortho:
                cmd.append("--camera-orthographic")
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        os.unlink(t.name)
        if res.returncode or not png.exists():
            sys.exit(f"f3d failed for {name}:\n{res.stderr}\n{' '.join(cmd)}")
        print(f"{png}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
