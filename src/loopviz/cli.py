"""Command-line pipeline.

  loopviz download --links-file links.txt      # or --link URL (repeat)
  loopviz build                                 # X, factors, verification
  loopviz optimize --schedule                   # baseline + Dirichlet sweep
  loopviz optimize --baseline                   # equal weights only
  loopviz compare                               # pairwise comparison UI
  loopviz fit                                   # Bradley-Terry weight fit
  loopviz optimize --weights fitted.json        # re-run under fitted w
  loopviz render <candidate> --resolution 4096  # print-quality export
  loopviz report                                # common-yardstick table
  loopviz relief sweep|build|demo               # 3D-printed height field
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import bt
from .config import AudioConfig, OptConfig, Paths, RenderConfig, ZConfig
from .ingest import (build_song_matrix, download, load_matrix, read_titles,
                     save_matrix, song_durations, window_seconds)
from .loss import (DEFAULT_SCHEDULE, equal_weights, loss_vector,
                   loss_vector_from_phi_dict, sample_weights, scalar_loss)
from .metrics import METRIC_NAMES, N_METRICS, features
from .operator import PlaylistOperator
from .optimize import make_objective, run_es
from .pool import candidate_dir, iter_candidate_files
from .render import render, save_png
from . import relief_cli
from .zspace import generate_Z, theta_to_params


def _paths(args) -> Paths:
    return Paths(root=Path(args.root))


def _load_operator(paths: Paths) -> PlaylistOperator:
    matrix_path = paths.data / "songs.npz"
    if not matrix_path.exists():
        raise SystemExit("no song matrix - run `loopviz build` first")
    X, _ = load_matrix(matrix_path)
    return PlaylistOperator.from_songs(X)


# -- commands -----------------------------------------------------------------

def cmd_download(args) -> None:
    paths = _paths(args)
    links = list(args.link or [])
    manifest = None
    if args.csv:
        import csv as csvmod
        rows = list(csvmod.DictReader(Path(args.csv).open()))
        usable = [r for r in rows if r.get("link_status", "ok") != "broken"]
        skipped = len(rows) - len(usable)
        links += [r["link"] for r in usable]
        manifest = {f"{i:03d}": {"issue": r.get("issue", ""),
                                 "title": r.get("song_title", ""),
                                 "artist": r.get("artist", ""),
                                 "link": r["link"]}
                    for i, r in enumerate(usable)}
        if skipped:
            print(f"skipping {skipped} rows with link_status=broken")
    if args.links_file:
        links += [ln.strip() for ln in Path(args.links_file).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not links:
        raise SystemExit("no links given (--link, --links-file or --csv)")
    download(links, paths.audio, args.sample_rate)
    if manifest is not None:
        (paths.data / "tracks.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {paths.data}/tracks.json ({len(manifest)} tracks)")
    print(f"downloaded {len(links)} tracks to {paths.audio}")


def cmd_build(args) -> None:
    paths = _paths(args)
    wavs = sorted(paths.audio.glob("[0-9][0-9][0-9].wav"))
    if len(wavs) < 2:
        raise SystemExit(f"need at least 2 wavs in {paths.audio}")
    if args.seconds is not None:
        seconds, stretch = args.seconds, False
    else:
        seconds, stretch = window_seconds(args.length_policy,
                                          song_durations(wavs))
    cfg = AudioConfig(sample_rate=args.sample_rate, excerpt_seconds=seconds,
                      stretch=stretch)
    print(f"building X from {len(wavs)} songs "
          f"({args.length_policy}, D = {cfg.dim}, "
          f"{cfg.excerpt_seconds:.1f}s @ {cfg.sample_rate} Hz)")
    X = build_song_matrix(wavs, cfg)
    titles = read_titles(paths.audio, wavs)
    save_matrix(X, titles, paths.data / "songs.npz", cfg,
                length_policy=args.length_policy)

    op = PlaylistOperator.from_songs(X)
    err = op.playback_error()
    cond = op.gram_condition()
    print(f"playlist operator: N = {op.N}, D = {op.D}")
    print(f"  max relative playback error |A x_n - x_n+1| / |x_n+1| = {err:.2e}")
    print(f"  Gram condition number = {cond:.2e} "
          f"({'ok' if cond < 1e8 else 'WARNING: near-duplicate songs'})")


def _save_candidate(paths: Paths, cand_id: str, w: np.ndarray, alpha,
                    theta: np.ndarray, phi: np.ndarray, loss_eq: float,
                    img_presentation: np.ndarray,
                    z_rank: int, es_seed: int) -> None:
    d = paths.runs / cand_id
    d.mkdir(parents=True, exist_ok=True)
    save_png(img_presentation, d / "presentation.png")
    (d / "candidate.json").write_text(json.dumps({
        "id": cand_id,
        "z_rank": z_rank,
        "es_seed": es_seed,
        "alpha": None if alpha is None else (None if np.isinf(alpha) else alpha),
        "weights": list(map(float, w)),
        "theta": list(map(float, theta)),
        "phi": {n: float(v) for n, v in zip(METRIC_NAMES, phi)},
        "loss_vector": list(map(float, loss_vector(phi))),
        "loss_eq": float(loss_eq),
    }, indent=2))


def _optimize_one(op: PlaylistOperator, w: np.ndarray, alpha, cand_id: str,
                  paths: Paths, zcfg: ZConfig, rcfg: RenderConfig,
                  ocfg: OptConfig, scalarization: str = "sum",
                  verbose: bool = True) -> float:
    objective = make_objective(op, w, zcfg, rcfg.opt_resolution,
                               stride=rcfg.opt_stride,
                               scalarization=scalarization)
    progress = (lambda g, l: print(f"  gen {g:2d}  loss {l:.3f}")) if verbose else None
    best, hist = run_es(objective, ocfg, progress=progress)

    # re-render the winner at presentation resolution (exact, stride 1)
    params = theta_to_params(best.theta)
    zf = generate_Z(params, op.D, zcfg, project_perp=op.project_perp)
    L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    img = render(L, R, min(rcfg.presentation_resolution, op.D), params)
    phi = features(img)
    loss_eq = scalar_loss(phi, equal_weights())
    _save_candidate(paths, cand_id, w, alpha, best.theta, phi, loss_eq, img,
                    zcfg.rank, ocfg.seed)

    # invariant: aesthetics never touched playback
    err = op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    print(f"[{cand_id}] done. L_eq = {loss_eq:.3f}, playback error = {err:.2e}")
    return loss_eq


def _run_job(job: dict) -> tuple[str, float]:
    """Worker for parallel optimization; re-creates state from paths."""
    paths = Paths(root=Path(job["root"]))
    X, _ = load_matrix(paths.data / "songs.npz")
    op = PlaylistOperator.from_songs(X)
    zcfg = ZConfig(rank=job["z_rank"])
    rcfg = RenderConfig()
    ocfg = OptConfig(generations=job["generations"], population=job["population"],
                     seed=job["es_seed"])
    loss_eq = _optimize_one(op, np.asarray(job["w"]), job["alpha"], job["id"],
                            paths, zcfg, rcfg, ocfg,
                            scalarization=job["scalarization"],
                            verbose=job["verbose"])
    return job["id"], loss_eq


def cmd_optimize(args) -> None:
    paths = _paths(args)
    _load_operator(paths)  # fail fast with a clear message
    rng = np.random.default_rng(args.seed)

    draws: list[tuple[str, np.ndarray, object]] = []
    if args.weights:
        w = np.asarray(json.loads(Path(args.weights).read_text())["w_simplex"])
        if w.size != N_METRICS:
            raise SystemExit(f"weights file has {w.size} entries, expected "
                             f"{N_METRICS}: re-run `loopviz fit`")
        draws.append(("fitted", w, None))
    elif args.baseline:
        draws.append(("baseline_eq", equal_weights(), float("inf")))
    else:  # full schedule
        for alpha, n_runs in DEFAULT_SCHEDULE:
            for r in range(n_runs):
                tag = "inf" if np.isinf(alpha) else f"{alpha:g}"
                cand_id = f"a{tag}_r{r}" if not np.isinf(alpha) else "baseline_eq"
                draws.append((cand_id, sample_weights(alpha, rng), alpha))

    # replicate each weight draw across ES seeds so a preference for a
    # candidate can be attributed to its weights, not one lucky basin
    jobs = []
    for idx, (cand_id, w, alpha) in enumerate(draws):
        for rep in range(args.replicates):
            full_id = cand_id if args.replicates == 1 else f"{cand_id}_s{rep}"
            jobs.append({
                "id": full_id, "w": list(map(float, w)), "alpha": alpha,
                "es_seed": args.seed + 1000 * idx + 101 * rep,
                "z_rank": args.z_rank, "root": str(paths.root),
                "generations": args.generations, "population": args.population,
                "scalarization": args.scalarization,
                "verbose": args.jobs == 1,
            })

    print(f"{len(jobs)} ES runs (Z rank {args.z_rank}, "
          f"{args.scalarization} scalarization, {args.jobs} parallel jobs)")

    if args.jobs > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_run_job, jobs))
    else:
        results = [_run_job(j) for j in jobs]

    print("\ncommon yardstick (equal-weight loss over active metrics):")
    for cand_id, loss_eq in sorted(results, key=lambda t: t[1]):
        print(f"  {cand_id:20s} L_eq = {loss_eq:.3f}")


def cmd_render(args) -> None:
    paths = _paths(args)
    op = _load_operator(paths)
    cdir = candidate_dir(paths.runs, args.candidate)
    if cdir is None:
        raise SystemExit(f"no candidate {args.candidate} in {paths.runs}")
    cand = json.loads((cdir / "candidate.json").read_text())

    params = theta_to_params(np.asarray(cand["theta"]))
    zcfg = ZConfig(rank=cand.get("z_rank", ZConfig().rank))
    zf = generate_Z(params, op.D, zcfg, project_perp=op.project_perp)
    L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    P = min(args.resolution, op.D)
    print(f"rendering {args.candidate} at {P}x{P}, {args.bits}-bit...")
    img = render(L, R, P, params)
    out = cdir / f"print_{P}.png"
    save_png(img, out, bit_depth=args.bits)
    print(f"wrote {out}")


def cmd_compare(args) -> None:
    from .compare_server import serve
    paths = _paths(args)
    serve(paths.runs, paths.comparisons, port=args.port)


def _resolve_source(paths: Paths, target: str) -> tuple[Path, Path, str]:
    """(source image, config out-dir, source id) for a candidate id or a path.

    A bare candidate id resolves to runs/<id>/presentation.png with configs
    written to runs/<id>/palettes/; a direct image path writes configs beside
    it under <image_dir>/palettes/.
    """
    p = Path(target)
    if p.exists() and p.is_file():
        return p, p.parent / "palettes", p.stem
    cdir = candidate_dir(paths.runs, target)
    if cdir is None or not (cdir / "presentation.png").exists():
        raise SystemExit(f"no image at {p} and no candidate "
                         f"{target} in {paths.runs}")
    return cdir / "presentation.png", cdir / "palettes", target


def _best_candidate(paths: Paths) -> str:
    """Highest-rated candidate id: ranked by fitted preference weights if
    available (runs/fitted_weights.json, lower w.loss is better, matching
    `top`), else by the equal-weight yardstick (lower loss_eq, matching
    `report`)."""
    cands = []
    for f in iter_candidate_files(paths.runs):
        if not (f.parent / "presentation.png").exists():
            continue
        cands.append(json.loads(f.read_text()))
    if not cands:
        raise SystemExit("no rated candidates in runs/ - run a sweep first, "
                         "or pass a candidate id / image path")
    w_path = paths.runs / "fitted_weights.json"
    if w_path.exists():
        score = bt.scorer_from_weights(json.loads(w_path.read_text()))
        def key(d):
            return score(d["phi"])
        basis = "fitted preference model"
    else:
        def key(d):
            return d["loss_eq"]
        basis = "equal-weight yardstick (no fitted_weights.json yet)"
    best = min(cands, key=key)
    print(f"most-rated candidate: {best['id']} (by {basis})")
    return best["id"]


def cmd_palette(args) -> None:
    from . import palette
    paths = _paths(args)
    target = args.target or _best_candidate(paths)
    source, out_dir, source_id = _resolve_source(paths, target)

    if args.apply:  # non-interactive: reproduce a saved recolor
        out = Path(args.out) if args.out else source.with_name(
            f"{source.stem}_{Path(args.apply).stem}.png")
        palette.apply_config(Path(args.apply), source, out)
        print(f"wrote {out}")
        return

    from .palette_server import serve
    serve(source, out_dir, source_id=source_id, port=args.port)


def cmd_fit(args) -> None:
    paths = _paths(args)
    comps = bt.load_comparisons(paths.comparisons)
    if not comps:
        raise SystemExit("no comparisons recorded - run `loopviz compare` first")
    # loss vectors are rebuilt from stored phi (by metric name) so candidates
    # saved under older metric sets stay usable
    loss_vectors = {}
    for f in iter_candidate_files(paths.runs):
        d = json.loads(f.read_text())
        loss_vectors[d["id"]] = loss_vector_from_phi_dict(d["phi"])
    # comparisons may reference candidates archived since they were made
    stale = [c for c in comps
             if c.winner not in loss_vectors or c.loser not in loss_vectors]
    if stale:
        comps = [c for c in comps if c not in stale]
        print(f"skipping {len(stale)} comparisons referencing "
              f"archived/removed candidates")
    fit = bt.fit_bt(comps, loss_vectors, l2=args.l2)
    se = fit.std_errors()

    print(f"Bradley-Terry fit on {fit.n_comparisons} comparisons "
          f"(log-likelihood {fit.log_likelihood:.2f})\n")
    print(f"  {'metric':22s} {'w_raw':>8s} {'stderr':>8s} {'w_simplex':>10s}")
    order = np.argsort(-fit.w_simplex)
    for i in order:
        flag = "  <- prefers HIGHER loss?!" if fit.w_raw[i] < -se[i] else ""
        print(f"  {METRIC_NAMES[i]:22s} {fit.w_raw[i]:8.3f} {se[i]:8.3f} "
              f"{fit.w_simplex[i]:10.3f}{flag}")

    # quadratic-interaction utility: fitted per-metric optima + pairwise
    # interactions; this is the model used for ranking (scorer_from_weights)
    phi_vecs = {}
    for f in iter_candidate_files(paths.runs):
        d = json.loads(f.read_text())
        phi_vecs[d["id"]] = np.array([d["phi"][n] for n in METRIC_NAMES])
    quad = bt.quad_fit(comps, phi_vecs, l2=args.quad_l2)
    names = bt.quad_term_names(METRIC_NAMES)
    zscores = np.abs(np.asarray(quad["theta"]) / np.asarray(quad["stderr"]))
    print(f"\nquadratic-interaction model "
          f"(log-likelihood {quad['log_likelihood']:.2f}), "
          f"top terms by |z|:")
    for i in np.argsort(-zscores)[:8]:
        print(f"  {names[i]:45s} {quad['theta'][i]:+8.3f} "
              f"(z={zscores[i]:.1f})")

    out = paths.runs / "fitted_weights.json"
    out.write_text(json.dumps({
        "w_raw": list(map(float, fit.w_raw)),
        "w_simplex": list(map(float, fit.w_simplex)),
        "stderr": list(map(float, se)),
        "n_comparisons": fit.n_comparisons,
        "metric_names": METRIC_NAMES,
        "quad": quad,
    }, indent=2))
    print(f"\nwrote {out}")
    print("re-optimize under your weights with:\n"
          f"  loopviz optimize --weights {out}")


def cmd_report(args) -> None:
    paths = _paths(args)
    cands = []
    for f in iter_candidate_files(paths.runs):
        cands.append(json.loads(f.read_text()))
    if not cands:
        raise SystemExit("no candidates yet")
    cands.sort(key=lambda d: d["loss_eq"])
    print(f"{'candidate':20s} {'alpha':>6s} {'L_eq':>8s}   top weighted metrics")
    for d in cands:
        w = np.asarray(d["weights"])
        top = np.argsort(-w)[:3]
        names = d.get("metric_names", METRIC_NAMES[:len(w)])
        tops = ", ".join(f"{names[i]}={w[i]:.2f}" for i in top if i < len(names))
        alpha = d.get("alpha")
        alpha_s = "eq" if alpha is None else f"{alpha:g}"
        print(f"{d['id']:20s} {alpha_s:>6s} {d['loss_eq']:8.3f}   {tops}")


def cmd_top(args) -> None:
    """Rank the candidate pool under a preference weight vector."""
    paths = _paths(args)
    w_path = Path(args.weights)
    score = bt.scorer_from_weights(json.loads(w_path.read_text()))
    cands = []
    for f in iter_candidate_files(paths.runs):
        d = json.loads(f.read_text())
        d["_score"] = score(d["phi"])
        cands.append(d)
    if not cands:
        raise SystemExit("no candidates yet")
    cands.sort(key=lambda d: d["_score"])
    ranked = cands                    # full pool, kept for per-song sheets
    if args.per_track:
        seen, picks = set(), []
        for d in cands:
            if d.get("song") not in seen:
                seen.add(d.get("song"))
                picks.append(d)
        cands = picks
    top = cands[:args.n]

    print(f"top {len(top)} under {w_path}"
          + (" (best per track)" if args.per_track else ""))
    for d in top:
        label = d.get("title") or d["id"]
        print(f"  {d['_score']:7.2f}  {label[:36]:36s} "
              f"f={d.get('f_hz', 0):.0f} n={d.get('n', '?')} "
              f"rho={d.get('rho', 0):.2f} clip={d.get('clip_pct', '?')} "
              f"drift {d.get('loop_drift_per_pass', 0):.0e}  [{d['id']}]")

    from PIL import Image
    from .sheet import make_sheet

    def sheet_of(cs, out):
        tiles = []
        for d in cs:
            img = np.asarray(
                Image.open(candidate_dir(paths.runs, d["id"])
                           / "presentation.png"),
                dtype=np.float64) / 255.0
            tiles.append((f"{(d.get('title') or d['id'])[:30]} | "
                          f"score {d['_score']:.2f} f={d.get('f_hz', 0):.0f} "
                          f"n={d.get('n', '?')} clip={d.get('clip_pct', '?')}",
                          img))
        make_sheet(tiles, out, tile_size=440, cols=4)  # 2x4 at n=8
        return out

    out = sheet_of(top, paths.runs / "top_preference.png")
    print(f"\nsheet: {out}")
    if args.per_song:
        # full ranked pool (not the per-track winners), one sheet per
        # (paper, song) so paper sizes never mix in one folder
        by_key = {}
        for d in ranked:
            key = (d.get("paper", "pre"), d.get("song", "?"))
            by_key.setdefault(key, []).append(d)
        outdir = paths.runs / "top_by_song"
        for (paper, song), cs in sorted(by_key.items()):
            stem = song.split(".")[0]
            title = (cs[0].get("title") or stem).replace("/", "-")[:40]
            pdir = outdir / paper
            pdir.mkdir(parents=True, exist_ok=True)
            sheet_of(cs[:args.n], pdir / f"{stem} {title}.png")
        print(f"per-song top-{args.n} sheets: {outdir}/<paper>/ "
              f"({len(by_key)} song x paper sheets)")
    print("(tiles are downscaled full matrices - open the listed "
          "runs/<id>/presentation.png at 100% to judge)")


# -- entry point ---------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="loopviz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project root (default: cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download audio from youtube links")
    p.add_argument("--link", action="append", help="youtube URL (repeatable)")
    p.add_argument("--links-file", help="file with one URL per line")
    p.add_argument("--csv", help="track CSV (issue,song_title,artist,album,"
                                 "link,link_status,notes); broken rows "
                                 "skipped; writes data/tracks.json manifest")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="storage rate of the wavs; sets the ceiling for "
                        "the operator's implied rate f")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("build", help="build song matrix X and verify the operator")
    p.add_argument("--sample-rate", type=int, default=1000,
                   help="Hz; the operator is exact at any rate, this only "
                        "sets playback fidelity (default 1000 while iterating)")
    p.add_argument("--length-policy", default="pad-longest",
                   choices=("crop-shortest", "pad-longest", "stretch"),
                   help="how songs of different lengths become one matrix")
    p.add_argument("--seconds", type=float, default=None,
                   help="override: fixed window in seconds (center-crop/pad)")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("optimize", help="ES-optimize the free part of A")
    p.add_argument("--baseline", action="store_true", help="equal weights only")
    p.add_argument("--weights", help="fitted_weights.json to optimize under")
    p.add_argument("--replicates", type=int, default=1,
                   help="ES runs per weight draw (different seeds), to "
                        "separate weight effects from basin luck")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel worker processes")
    p.add_argument("--z-rank", type=int, default=ZConfig().rank,
                   help="rank of the free part Z (texture richness vs cost)")
    p.add_argument("--scalarization", default="sum",
                   choices=("sum", "chebyshev"),
                   help="weighted sum, or augmented Chebyshev (reaches "
                        "non-convex Pareto points)")
    p.add_argument("--generations", type=int, default=14)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("render", help="print-quality render of a candidate")
    p.add_argument("candidate")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--bits", type=int, choices=(8, 16), default=16)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("compare", help="pairwise comparison web UI")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("palette", help="interactive diverging-palette designer "
                                       "for a candidate or image (printable)")
    p.add_argument("target", nargs="?", default=None,
                   help="candidate id (runs/<id>/presentation.png) or a path "
                        "to a grayscale image; default: highest-rated candidate")
    p.add_argument("--apply", help="non-interactive: reproduce the recolor "
                                   "from a saved palette config JSON")
    p.add_argument("-o", "--out", help="output PNG for --apply")
    p.add_argument("--port", type=int, default=8766)
    p.set_defaults(fn=cmd_palette)

    p = sub.add_parser("fit", help="fit Bradley-Terry weights from comparisons")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--quad-l2", type=float, default=3.0,
                   help="L2 for the quadratic-interaction utility model")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("report", help="common-yardstick candidate table")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("top", help="rank candidates under fitted preferences")
    p.add_argument("--weights", default="runs/fitted_weights.json")
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--per-track", action="store_true",
                   help="best candidate per song instead of overall top")
    p.add_argument("--per-song", action="store_true",
                   help="also write a top-n sheet for every song "
                        "(runs/top_by_song/)")
    p.set_defaults(fn=cmd_top)

    relief_cli.add_parser(sub)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
