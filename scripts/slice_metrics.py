"""Headless PrusaSlicer metrics: manifold check, print time, filament, layers.

Slices one or more STLs with a named settings preset and reports the slicer's own
estimates. Nothing here is printed; the G-code exists only so the footer can be read.

    uv run python scripts/slice_metrics.py runs/plate/smoke_p2/bed.stl --preset generic04
    uv run python scripts/slice_metrics.py --per-strip runs/plate/smoke_p2/strips --preset slow04

The slicer binary is found via LOOPVIZ_PRUSA_SLICER, then PATH, then the nix store
(`nix build nixpkgs#prusa-slicer --no-link --print-out-paths`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PRESETS: dict[str, dict[str, str]] = {}
PRESETS["generic04"] = {
    "nozzle-diameter": "0.4",
    "filament-diameter": "1.75",
    "filament-density": "1.24",
    "layer-height": "0.2",
    "first-layer-height": "0.2",
    "perimeters": "3",
    "top-solid-layers": "4",
    "bottom-solid-layers": "4",
    "fill-density": "15%",
    "fill-pattern": "gyroid",
    "brim-width": "5",
    "skirts": "0",
    "seam-position": "rear",
    "elefant-foot-compensation": "0.2",
    "bed-shape": "0x0,340x0,340x340,0x340",
    "max-print-height": "325",
    "gcode-flavor": "marlin2",
    "machine-limits-usage": "emit_to_gcode",
    "perimeter-speed": "80",
    "external-perimeter-speed": "50",
    "infill-speed": "150",
    "solid-infill-speed": "100",
    "top-solid-infill-speed": "60",
    "travel-speed": "300",
    "max-volumetric-speed": "12",
    "machine-max-acceleration-travel": "5000,5000",
    "machine-max-acceleration-extruding": "3000,3000",
    "machine-max-acceleration-x": "5000,5000",
    "machine-max-acceleration-y": "5000,5000",
}
PRESETS["slow04"] = {
    **PRESETS["generic04"],
    "perimeter-speed": "60",
    "external-perimeter-speed": "40",
    "infill-speed": "100",
    "solid-infill-speed": "80",
    "top-solid-infill-speed": "40",
    "machine-max-acceleration-travel": "2000,2000",
    "machine-max-acceleration-extruding": "2000,2000",
    "machine-max-acceleration-x": "2000,2000",
    "machine-max-acceleration-y": "2000,2000",
}

NIX_FALLBACK = (
    "/nix/store/8v43d0f5p0hpfdphv5kgy1dlzm6k40mb-prusa-slicer-2.9.4/bin/prusa-slicer"
)
PROGRESS_RE = re.compile(r"^\s*\d+\s*=>")


def find_slicer() -> str:
    for cand in (
        os.environ.get("LOOPVIZ_PRUSA_SLICER"),
        shutil.which("prusa-slicer"),
        NIX_FALLBACK,
    ):
        if cand and Path(cand).is_file():
            return cand
    out = (
        subprocess.run(
            ["nix", "build", "nixpkgs#prusa-slicer", "--no-link", "--print-out-paths"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return str(Path(out[-1]) / "bin" / "prusa-slicer")


def parse_duration_h(text: str) -> float:
    """'1d 2h 3m 4s' -> hours."""
    units = {"d": 24.0, "h": 1.0, "m": 1 / 60, "s": 1 / 3600}
    return sum(
        float(n) * units[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", text)
    )


def parse_gcode(path: Path) -> dict:
    """Single pass over the G-code: footer estimates plus layer count and max Z."""
    res: dict = {"layers": 0, "max_z_mm": 0.0}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(";LAYER_CHANGE"):
                res["layers"] += 1
            elif line.startswith(";Z:"):
                res["max_z_mm"] = max(res["max_z_mm"], float(line[3:]))
            elif line.startswith("; estimated printing time (normal mode) ="):
                res["time_h"] = round(parse_duration_h(line.split("=", 1)[1]), 4)
                res["time_text"] = line.split("=", 1)[1].strip()
            elif line.startswith(
                "; estimated first layer printing time (normal mode) ="
            ):
                res["first_layer_h"] = round(parse_duration_h(line.split("=", 1)[1]), 4)
            elif line.startswith("; total filament used [g] ="):
                res["filament_g"] = float(line.split("=", 1)[1])
            elif line.startswith("; filament used [cm3] ="):
                res["filament_cm3"] = float(line.split("=", 1)[1])
            elif line.startswith("; filament used [mm] ="):
                res["filament_mm"] = float(line.split("=", 1)[1])
    return res


def parse_info(text: str) -> dict:
    info: dict = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if k in ("manifold",):
            info[k] = v == "yes"
        elif k in ("number_of_facets", "number_of_parts"):
            info[k] = int(v)
        elif k in ("volume", "size_x", "size_y", "size_z"):
            info[k] = float(v)
    return info


def warnings_from(output: str) -> list[str]:
    keep = []
    for line in output.splitlines():
        s = line.strip()
        if not s or PROGRESS_RE.match(s) or s.startswith("Slicing result exported"):
            continue
        keep.append(s)
    return keep


def slice_one(slicer: str, stl: Path, settings: dict[str, str], gcode: Path) -> dict:
    t0 = time.perf_counter()
    info_run = subprocess.run(
        [slicer, "--info", str(stl)], capture_output=True, text=True, check=False
    )
    rec = {
        "stl": str(stl),
        "stl_mb": round(stl.stat().st_size / 1e6, 1),
        "info": parse_info(info_run.stdout),
    }
    cmd = [slicer, "--export-gcode", "--dont-arrange", "--loglevel", "2"]
    for k, v in settings.items():
        cmd += [f"--{k}", v]
    cmd += ["-o", str(gcode), str(stl)]
    rec["command"] = " ".join(cmd)
    t1 = time.perf_counter()
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    rec["slice_wall_s"] = round(time.perf_counter() - t1, 1)
    rec["warnings"] = warnings_from(run.stdout + run.stderr) + warnings_from(
        info_run.stderr
    )
    rec["returncode"] = run.returncode
    if run.returncode == 0 and gcode.exists():
        rec["gcode"] = str(gcode)
        rec["gcode_mb"] = round(gcode.stat().st_size / 1e6, 1)
        rec.update(parse_gcode(gcode))
    rec["total_wall_s"] = round(time.perf_counter() - t0, 1)
    return rec


def summary(rec: dict) -> str:
    inf = rec.get("info", {})
    if rec.get("returncode"):
        return f"{Path(rec['stl']).name}: FAILED rc={rec['returncode']} {rec['warnings'][-3:]}"
    return (
        f"{Path(rec['stl']).name}: manifold={inf.get('manifold')} facets={inf.get('number_of_facets')} "
        f"time={rec.get('time_text')} ({rec.get('time_h')} h) filament={rec.get('filament_g')} g "
        f"/ {rec.get('filament_cm3')} cm3 layers={rec.get('layers')} maxZ={rec.get('max_z_mm')} "
        f"slice={rec.get('slice_wall_s')} s"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stl", nargs="*", type=Path, help="STL files (default: bed.stl)")
    ap.add_argument(
        "--per-strip", type=Path, help="directory of strip STLs; slice each and sum"
    )
    ap.add_argument("--preset", choices=sorted(PRESETS), default="generic04")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a preset flag",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="JSON output path (default: slice_<preset>.json next to the first STL)",
    )
    ap.add_argument(
        "--gcode-dir", type=Path, help="where G-code goes (default: a fresh temp dir)"
    )
    ap.add_argument("--keep-gcode", choices=["true", "false"], default="true")
    a = ap.parse_args(argv)

    settings = dict(PRESETS[a.preset])
    for kv in a.set:
        k, _, v = kv.partition("=")
        settings[k.replace("_", "-")] = v
    stls = list(a.stl)
    if a.per_strip:
        stls += sorted(a.per_strip.glob("*.stl"))
    if not stls:
        stls = [Path("bed.stl")]
    out = a.out or stls[0].parent / f"slice_{a.preset}.json"
    gdir = a.gcode_dir or Path(tempfile.mkdtemp(prefix="loopviz_slice_"))
    gdir.mkdir(parents=True, exist_ok=True)
    slicer = find_slicer()
    print(f"slicer: {slicer}", file=sys.stderr)

    files = []
    for stl in stls:
        rec = slice_one(slicer, stl, settings, gdir / f"{stl.stem}_{a.preset}.gcode")
        print(summary(rec))
        for w in rec["warnings"]:
            print(f"  warning: {w}")
        if a.keep_gcode == "false" and "gcode" in rec:
            Path(rec.pop("gcode")).unlink()
        files.append(rec)

    ok = [f for f in files if not f.get("returncode")]
    result = {
        "preset": a.preset,
        "settings": settings,
        "slicer": slicer,
        "files": files,
        "total": {
            "n": len(ok),
            "time_h": round(sum(f.get("time_h", 0) for f in ok), 3),
            "filament_g": round(sum(f.get("filament_g", 0) for f in ok), 1),
            "filament_cm3": round(sum(f.get("filament_cm3", 0) for f in ok), 1),
            "slice_wall_s": round(sum(f.get("slice_wall_s", 0) for f in ok), 1),
            "all_manifold": all(f.get("info", {}).get("manifold") for f in ok),
            "facets": sum(f.get("info", {}).get("number_of_facets", 0) for f in ok),
        },
    }
    out.write_text(json.dumps(result, indent=1))
    t = result["total"]
    print(
        f"TOTAL [{a.preset}] n={t['n']} time={t['time_h']} h filament={t['filament_g']} g "
        f"slice={t['slice_wall_s']} s manifold={t['all_manifold']} -> {out}"
    )
    return 0 if len(ok) == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
