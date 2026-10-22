"""Predicted-vs-true plots for ensemble (pooled out-of-fold or stacked)
shear predictions.

Rows: g+ and g×.  Columns:
  1. raw predictions against truth: density, binned means, 1:1 line and the
     fitted slope (the estimator as trained, shrunk toward 0);
  2. response-corrected predictions, (pred - c) / slope: the same estimator
     rescaled so its mean response is 1, which is what a shear measurement
     uses; the scatter about the 1:1 line is the per-galaxy shear noise;
  3. the distribution of the response-corrected error, with its width
     (sigma_eff) next to the spread of the true shears.

Works on oof_predictions.csv (from a --kfold run; raw predictions are
standardised per fold first because folds have different scales) and on
stack_predictions.csv (from kl_stack.py).

Usage:
    python kl_plot_ensemble.py --csv run/cnn_k5_feat/oof_predictions.csv
    python kl_plot_ensemble.py --csv run/stack_k5/stack_predictions.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ENS_COLORS = {"g+": "#2E86AB", "g×": "#E84855"}


def _binned(t, p, nbins):
    edges = np.quantile(t, np.linspace(0, 1, nbins + 1))
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
    bc, bm, be = [], [], []
    for k in range(nbins):
        s = which == k
        if s.sum() >= 5:
            bc.append(t[s].mean()); bm.append(p[s].mean())
            be.append(p[s].std() / np.sqrt(s.sum()))
    return np.array(bc), np.array(bm), np.array(be)


def plot_ensemble_pred_vs_true(preds, labels, out_path, title="", nbins=10, groups=None,
                               nboot=200, seed=0):
    """Three-panel predicted-vs-true figure per shear component.  Returns a
    dict of per-component summary numbers."""
    preds, labels = np.asarray(preds, float), np.asarray(labels, float)
    n = len(labels)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    out = {}
    for row, name in enumerate(["g+", "g×"]):
        t, p = labels[:, row], preds[:, row]
        col = _ENS_COLORS[name]
        sl, ic = np.polyfit(t, p, 1)
        r = np.corrcoef(t, p)[0, 1] if np.std(p) > 0 else 0.0
        boots = []
        for _ in range(nboot):
            ix = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
            if np.std(p[ix]) > 0:
                boots.append(np.corrcoef(t[ix], p[ix])[0, 1])
        r_err = float(np.std(boots)) if boots else np.nan
        tlim = np.quantile(np.abs(t), 0.995) * 1.1

        # 1. raw
        ax = axes[row, 0]
        plim = np.quantile(np.abs(p), 0.995) * 1.15
        ax.hexbin(t, p, gridsize=45, cmap="Greys", mincnt=1,
                  extent=(-tlim, tlim, -plim, plim), bins="log")
        bc, bm, be = _binned(t, p, nbins)
        ax.errorbar(bc, bm, yerr=be, fmt="o", color=col, ms=5, capsize=2,
                    label="binned mean")
        xs = np.array([-tlim, tlim])
        ax.plot(xs, ic + sl * xs, color=col, lw=1.5, label=f"fit slope {sl:.3f}")
        ax.plot(xs, xs, "k--", lw=0.8, label="1:1")
        ax.set_xlim(-tlim, tlim); ax.set_ylim(-plim, plim)
        ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"{name} predicted (raw)")
        ax.set_title(f"{name} raw:  r = {r:+.3f} ± {r_err:.3f}  ({r / r_err:.1f}σ)"
                     if r_err > 0 else f"{name} raw:  r = {r:+.3f}")

        # 2. response-corrected
        ax = axes[row, 1]
        if sl > 0 and r_err > 0 and r / r_err > 2:
            pc = (p - ic) / sl
            clim = np.quantile(np.abs(pc), 0.98) * 1.1
            ax.hexbin(t, pc, gridsize=45, cmap="Greys", mincnt=1,
                      extent=(-tlim, tlim, -clim, clim), bins="log")
            bc, bm, be = _binned(t, pc, nbins)
            ax.errorbar(bc, bm, yerr=be, fmt="o", color=col, ms=5, capsize=2,
                        label="binned mean")
            ax.plot(xs, xs, "k--", lw=1, label="1:1")
            ax.set_ylim(-clim, clim)
            ax.set_title(f"{name} response-corrected: (pred − c) / {sl:.3f}")
            err = pc - t
            sig_eff = float(np.std(err))
        else:
            pc, err, sig_eff = None, None, np.nan
            ax.text(0.5, 0.5, "no significant positive response:\nnothing to correct",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name} response-corrected")
        ax.set_xlim(-tlim, tlim)
        ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"{name} estimate")

        # 3. error distribution
        ax = axes[row, 2]
        if err is not None:
            lim = np.quantile(np.abs(err), 0.99)
            bins = np.linspace(-lim, lim, 60)
            ax.hist(err, bins=bins, density=True, color=col, alpha=0.6,
                    label=f"estimate − true, σ_eff = {sig_eff:.3f}")
            ax.hist(t, bins=bins, density=True, histtype="step", color="k",
                    label=f"true shear, σ = {np.std(t):.3f}")
            ax.set_xlabel("shear"); ax.set_ylabel("density")
            ax.set_title(f"{name}: per-galaxy noise σ_eff = {sig_eff:.3f}; "
                         f"ensemble error σ_eff/√N = {sig_eff / np.sqrt(n):.4f}")
        else:
            ax.axis("off")
        out[name] = dict(n=n, r=r, r_err=r_err, slope=sl, intercept=ic,
                         sigma_eff=sig_eff, ensemble_err=sig_eff / np.sqrt(n))
        for a in axes[row]:
            if a.has_data():
                a.grid(alpha=0.25); a.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{title}   (N = {n})", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}")
    for name, d in out.items():
        print(f"    {name}: r {d['r']:+.3f} ± {d['r_err']:.3f}  slope {d['slope']:.3f}  "
              f"sigma_eff {d['sigma_eff']:.3f}  ensemble error {d['ensemble_err']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="oof_predictions.csv or stack_predictions.csv")
    ap.add_argument("--pred", default=None,
                    help="prediction columns suffix (raw, pred, stack); default: "
                         "stack if present, else raw")
    ap.add_argument("--out", default=None, help="output png (default next to --csv)")
    ap.add_argument("--nbins", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    suf = args.pred or ("stack" if "g_plus_stack" in df else "raw")
    cols = [f"g_plus_{suf}", f"g_cross_{suf}"]
    if not all(c in df for c in cols):
        raise SystemExit(f"{args.csv} has no {cols}")
    P = df[cols].to_numpy(float).copy()
    lab = df[["g_plus_true", "g_cross_true"]].to_numpy(float)
    note = suf
    if suf == "raw" and "fold" in df and df["fold"].nunique() > 1:
        # fold models have different output scales: put each fold on a
        # common scale before pooling (fit-free: match the fold's spread)
        for f in df["fold"].unique():
            m = (df["fold"] == f).to_numpy()
            P[m] = (P[m] - P[m].mean(0)) / (P[m].std(0) + 1e-12) * P.std(0)
        note = "raw, per-fold standardised"
    out = Path(args.out) if args.out else Path(args.csv).with_name(f"pred_vs_true_ensemble_{suf}.png")
    groups = df["pop.id"].to_numpy() if "pop.id" in df else None
    plot_ensemble_pred_vs_true(P, lab, out, f"{Path(args.csv).parent.name}: {note}",
                               nbins=args.nbins, groups=groups)


if __name__ == "__main__":
    main()
