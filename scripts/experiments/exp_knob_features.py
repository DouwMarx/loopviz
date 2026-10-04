"""Do the generation knobs (n, rho, f, clip) predict preference beyond
what the 11 image metrics transmit?

The BT model sees candidates only through the metric losses; the knobs
are causes-of-causes. But the behavioral cuts showed systematic knob
preferences (2 kHz rejected, near-empty canvases rejected) - if the
metric bottleneck leaks that information, appending standardized knob
features should raise held-out prediction accuracy.

Ablation: k-fold cross-validated accuracy of (a) metrics-only vs
(b) metrics + knobs, on the archived demo comparisons. Knob prior mean
is 0 (no prior preference); metric prior stays uniform.

Run: .venv/bin/python scripts/exp_knob_features.py [--folds 5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loopviz import bt
from loopviz.loss import loss_vector_from_phi_dict
from loopviz.metrics import METRIC_NAMES, N_METRICS

ROOT = Path(__file__).parent.parent
ARCHIVE = ROOT / "runs" / "archive_v3_demo"
KNOBS = ("n", "rho", "f_hz", "clip_pct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cands = {}
    for cj in ARCHIVE.glob("songop_*/candidate.json"):
        d = json.loads(cj.read_text())
        cands[d["id"]] = d
    comps = bt.load_comparisons(ARCHIVE / "comparisons.jsonl")
    comps = [c for c in comps if c.winner in cands and c.loser in cands]
    print(f"{len(comps)} comparisons over {len(cands)} archived candidates")

    # feature sets
    raw_knobs = np.array([[cands[k][kn] for kn in KNOBS]
                          for k in sorted(cands)])
    mu, sd = raw_knobs.mean(axis=0), raw_knobs.std(axis=0) + 1e-12
    metrics_only, with_knobs = {}, {}
    for k, d in cands.items():
        lv = loss_vector_from_phi_dict(d["phi"])
        z = (np.array([d[kn] for kn in KNOBS]) - mu) / sd
        metrics_only[k] = lv
        with_knobs[k] = np.concatenate([lv, z])
    prior = np.concatenate([np.full(N_METRICS, 1.0 / N_METRICS),
                            np.zeros(len(KNOBS))])

    def cv_accuracy(vectors, prior_mean=None):
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(comps))
        folds = np.array_split(order, args.folds)
        accs = []
        for hold in folds:
            train = [comps[i] for i in order if i not in set(hold)]
            test = [comps[i] for i in hold]
            fit = bt.fit_bt(train, vectors, prior_mean=prior_mean)
            ok = sum(fit.w_raw @ (vectors[c.loser] - vectors[c.winner]) > 0
                     for c in test)
            accs.append(ok / len(test))
        return np.mean(accs), np.std(accs) / np.sqrt(len(accs))

    acc_m, se_m = cv_accuracy(metrics_only)
    acc_k, se_k = cv_accuracy(with_knobs, prior_mean=prior)
    print(f"\n{args.folds}-fold held-out accuracy:")
    print(f"  metrics only (11 features):     {acc_m:.1%} +- {se_m:.1%}")
    print(f"  metrics + knobs (15 features):  {acc_k:.1%} +- {se_k:.1%}")

    fit = bt.fit_bt(comps, with_knobs, prior_mean=prior)
    se = fit.std_errors()
    names = list(METRIC_NAMES) + [f"knob:{k}" for k in KNOBS]
    print("\nfull-data fit, features sorted by |w|/stderr:")
    idx = np.argsort(-np.abs(fit.w_raw) / (se + 1e-12))
    for i in idx:
        sig = "*" if abs(fit.w_raw[i]) > 2 * se[i] else " "
        print(f"  {sig} {names[i]:22s} w={fit.w_raw[i]:+7.3f} +- {se[i]:.3f}")
    print("\n(* = |w| > 2 stderr. knob features are z-scored: w is the "
          "preference shift per pool-standard-deviation of the knob; "
          "negative = prefers larger values... of the LOSS difference - "
          "for knobs, negative w means preferring the knob HIGH.)")


if __name__ == "__main__":
    main()
