"""Command-line pipeline.

  playlistviz download --links-file links.txt      # or --link URL (repeat)
  playlistviz build                                 # X, factors, verification
  playlistviz optimize --schedule                   # baseline + Dirichlet sweep
  playlistviz optimize --baseline                   # equal weights only
  playlistviz compare                               # pairwise comparison UI
  playlistviz fit                                   # Bradley-Terry weight fit
  playlistviz optimize --weights fitted.json        # re-run under fitted w
  playlistviz render <candidate> --resolution 4096  # print-quality export
  playlistviz report                                # common-yardstick table
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
from .render import render, save_png
from .zspace import generate_Z, theta_to_params


def _paths(args) -> Paths:
    return Paths(root=Path(args.root))


def _load_operator(paths: Paths) -> PlaylistOperator:
    matrix_path = paths.data / "songs.npz"
    if not matrix_path.exists():
        raise SystemExit("no song matrix - run `playlistviz build` first")
    X, _ = load_matrix(matrix_path)
    return PlaylistOperator.from_songs(X)


# -- commands -----------------------------------------------------------------

def cmd_download(args) -> None:
    paths = _paths(args)
    links = list(args.link or [])
    if args.links_file:
        links += [ln.strip() for ln in Path(args.links_file).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not links:
        raise SystemExit("no links given (--link or --links-file)")
    cfg = AudioConfig()
    download(links, paths.audio, cfg.sample_rate)
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
                             f"{N_METRICS}: re-run `playlistviz fit`")
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


def _run_embed_job(job: dict) -> tuple[str, float]:
    """Worker: 2D-target ES -> embed winner into the operator -> candidate."""
    from .embed import display, embed_image
    from .targets import (N_PARAMS_2D, apply_family, generate_target,
                          theta2d_to_params)

    paths = Paths(root=Path(job["root"]))
    X, _ = load_matrix(paths.data / "songs.npz")
    op = PlaylistOperator.from_songs(X)
    w = np.asarray(job["w"])
    family = job["family"]
    ocfg = OptConfig(generations=job["generations"], population=job["population"],
                     seed=job["es_seed"], subspace_rank=0)

    def objective(theta):
        from .optimize import EvalResult
        params = apply_family(theta2d_to_params(theta), family)
        img = generate_target(params, 384)
        phi = features(img)
        return EvalResult(theta=theta.copy(),
                          loss=scalar_loss(phi, w, job["scalarization"]), phi=phi)

    best, _ = run_es(objective, ocfg, n_params=N_PARAMS_2D)

    # regenerate the winner at presentation res (seed-stable) and embed it
    pres = job["embed_res"]
    T = generate_target(apply_family(theta2d_to_params(best.theta), family), pres)
    res = embed_image(op, T, rank=job["embed_rank"])
    img = display(res.achieved)
    phi = features(img)
    loss_eq = scalar_loss(phi, equal_weights())
    err = op.playback_error(U=res.U, Vp=res.Vp, scale=res.scale)

    d = paths.runs / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    save_png(img, d / "presentation.png")
    (d / "candidate.json").write_text(json.dumps({
        "id": job["id"],
        "kind": "embed",
        "family": job["family"],
        "embed_rank": res.rank,
        "embed_rel_error": res.rel_error,
        "es_seed": job["es_seed"],
        "alpha": job["alpha"],
        "weights": list(map(float, w)),
        "theta2d": list(map(float, best.theta)),
        "theta": [],
        "phi": {n: float(v) for n, v in zip(METRIC_NAMES, phi)},
        "loss_vector": list(map(float, loss_vector(phi))),
        "loss_eq": float(loss_eq),
    }, indent=2))
    print(f"[{job['id']}] done. L_eq = {loss_eq:.3f}, "
          f"embed err {res.rel_error:.2%}, playback error = {err:.2e}")
    return job["id"], loss_eq


def cmd_embed(args) -> None:
    """Candidate sweep via the fast path: 2D-target ES + embedding."""
    paths = _paths(args)
    _load_operator(paths)  # fail fast
    rng = np.random.default_rng(args.seed)

    draws: list[tuple[str, np.ndarray, object]] = []
    if args.weights:
        w = np.asarray(json.loads(Path(args.weights).read_text())["w_simplex"])
        draws.append(("emb_fitted", w, None))
    else:
        for alpha, n_runs in DEFAULT_SCHEDULE:
            for r in range(n_runs):
                tag = "inf" if np.isinf(alpha) else f"{alpha:g}"
                cand_id = f"emb_a{tag}_r{r}" if not np.isinf(alpha) else "emb_baseline"
                a = None if np.isinf(alpha) else alpha
                draws.append((cand_id, sample_weights(alpha, rng), a))

    jobs = []
    for idx, (cand_id, w, alpha) in enumerate(draws):
        for rep in range(args.replicates):
            full_id = cand_id if args.replicates == 1 else f"{cand_id}_s{rep}"
            jobs.append({
                "id": full_id, "w": list(map(float, w)), "alpha": alpha,
                "es_seed": args.seed + 1000 * idx + 101 * rep,
                "root": str(paths.root),
                "generations": args.generations, "population": args.population,
                "scalarization": args.scalarization, "family": args.family,
                "embed_res": args.embed_res, "embed_rank": args.embed_rank,
            })

    print(f"{len(jobs)} embed runs ({args.jobs} parallel jobs)")
    if args.jobs > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_run_embed_job, jobs))
    else:
        results = [_run_embed_job(j) for j in jobs]

    print("\ncommon yardstick (equal-weight loss):")
    for cand_id, loss_eq in sorted(results, key=lambda t: t[1]):
        print(f"  {cand_id:22s} L_eq = {loss_eq:.3f}")


def cmd_render(args) -> None:
    paths = _paths(args)
    op = _load_operator(paths)
    cand_file = paths.runs / args.candidate / "candidate.json"
    if not cand_file.exists():
        raise SystemExit(f"no candidate {args.candidate} in {paths.runs}")
    cand = json.loads(cand_file.read_text())

    if cand.get("kind") == "embed":
        from .embed import display, embed_image
        from .targets import apply_family, generate_target, theta2d_to_params
        P = min(args.resolution, op.D)
        rank = min(max(cand["embed_rank"], P // 8), P)
        print(f"re-embedding {args.candidate} at {P}x{P} (rank {rank}), "
              f"{args.bits}-bit...")
        params2d = apply_family(theta2d_to_params(np.asarray(cand["theta2d"])),
                                cand.get("family", "mixed"))
        T = generate_target(params2d, P)
        res = embed_image(op, T, rank=rank)
        img = display(res.achieved)
        err = op.playback_error(U=res.U, Vp=res.Vp, scale=res.scale)
        print(f"embed rel error {res.rel_error:.2%}, playback error {err:.2e}")
        out = paths.runs / args.candidate / f"print_{P}.png"
        save_png(img, out, bit_depth=args.bits)
        print(f"wrote {out}")
        return

    params = theta_to_params(np.asarray(cand["theta"]))
    zcfg = ZConfig(rank=cand.get("z_rank", ZConfig().rank))
    zf = generate_Z(params, op.D, zcfg, project_perp=op.project_perp)
    L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    P = min(args.resolution, op.D)
    print(f"rendering {args.candidate} at {P}x{P}, {args.bits}-bit...")
    img = render(L, R, P, params)
    out = paths.runs / args.candidate / f"print_{P}.png"
    save_png(img, out, bit_depth=args.bits)
    print(f"wrote {out}")


def cmd_compare(args) -> None:
    from .compare_server import serve
    paths = _paths(args)
    serve(paths.runs, paths.comparisons, port=args.port)


def cmd_fit(args) -> None:
    paths = _paths(args)
    comps = bt.load_comparisons(paths.comparisons)
    if not comps:
        raise SystemExit("no comparisons recorded - run `playlistviz compare` first")
    # loss vectors are rebuilt from stored phi (by metric name) so candidates
    # saved under older metric sets stay usable
    loss_vectors = {}
    for f in sorted(paths.runs.glob("*/candidate.json")):
        if f.parent.name.startswith("exp_"):
            continue
        d = json.loads(f.read_text())
        loss_vectors[d["id"]] = loss_vector_from_phi_dict(d["phi"])
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

    out = paths.runs / "fitted_weights.json"
    out.write_text(json.dumps({
        "w_raw": list(map(float, fit.w_raw)),
        "w_simplex": list(map(float, fit.w_simplex)),
        "stderr": list(map(float, se)),
        "n_comparisons": fit.n_comparisons,
        "metric_names": METRIC_NAMES,
    }, indent=2))
    print(f"\nwrote {out}")
    print("re-optimize under your weights with:\n"
          f"  playlistviz optimize --weights {out}")


def cmd_report(args) -> None:
    paths = _paths(args)
    cands = []
    for f in sorted(paths.runs.glob("*/candidate.json")):
        if f.parent.name.startswith("exp_"):
            continue
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


# -- entry point ---------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="playlistviz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project root (default: cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download audio from youtube links")
    p.add_argument("--link", action="append", help="youtube URL (repeatable)")
    p.add_argument("--links-file", help="file with one URL per line")
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

    p = sub.add_parser("embed", help="candidate sweep via 2D targets + embedding "
                                     "(~50x faster per candidate)")
    p.add_argument("--weights", help="fitted_weights.json to optimize under")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--jobs", type=int, default=2,
                   help="parallel workers (embedding is memory-heavy)")
    p.add_argument("--family", default="cloud",
                   choices=("cloud", "cells", "mixed"),
                   help="generator family: nebula/ink (cloud), crack "
                        "networks (cells), or unconstrained (mixed)")
    p.add_argument("--embed-res", type=int, default=768,
                   help="presentation embed resolution")
    p.add_argument("--embed-rank", type=int, default=220)
    p.add_argument("--scalarization", default="sum",
                   choices=("sum", "chebyshev"))
    p.add_argument("--generations", type=int, default=14)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("render", help="print-quality render of a candidate")
    p.add_argument("candidate")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--bits", type=int, choices=(8, 16), default=16)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("compare", help="pairwise comparison web UI")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("fit", help="fit Bradley-Terry weights from comparisons")
    p.add_argument("--l2", type=float, default=1.0)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("report", help="common-yardstick candidate table")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
