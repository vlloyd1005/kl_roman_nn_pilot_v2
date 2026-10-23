"""Predicted vs true shear, split into bins of a galaxy property (cos i by
default): one row per bin, columns g+ and g×.  Each panel shows the galaxies
as a density plot, the mean prediction in bins of true shear, the 1:1 line and
the fitted slope, and is titled with N, r ± error and sigma_eff, the
per-galaxy shear noise sigma_g * sqrt(1/r^2 - 1).

Usage:
    python kl_pred_vs_true_bins.py --pred run/fold0/test_predictions.csv
    python kl_pred_vs_true_bins.py --pred run/oof_predictions.csv --edges 0 0.3 0.7 1
    python kl_pred_vs_true_bins.py --pred run/test_predictions.csv \
        --col snr.line_total --edges 0 50 100 200 1e9 --parquet data.parquet
Any column missing from the predictions file is read from --parquet (joined
on src_row).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pred_vs_true_by_bins(preds, labels, values, edges, out_path, title="",
                              label="cos i", groups=None, nbins=8, nboot=200, seed=0):
    """Grid of predicted-vs-true panels, one row per bin of `values`."""
    preds, labels, values = (np.asarray(preds, float), np.asarray(labels, float),
                             np.asarray(values, float))
    groups = np.arange(len(labels)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    fig, axes = plt.subplots(nb, 2, figsize=(11, 3.6 * nb), squeeze=False)
    tlim = np.quantile(np.abs(labels), 0.995) * 1.1
    rows = []
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        m = (values > lo) & (values <= hi) if b > 0 else (values >= lo) & (values <= hi)
        for k, (name, col) in enumerate([("g+", "#2E86AB"), ("g×", "#E84855")]):
            ax = axes[b, k]
            t, p = labels[m, k], preds[m, k]
            if m.sum() < 20:
                ax.text(0.5, 0.5, f"{m.sum()} galaxies", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            r = np.corrcoef(t, p)[0, 1] if np.std(p) > 0 else 0.0
            g = groups[m]
            ug = np.unique(g)
            idx = {u: np.where(g == u)[0] for u in ug}
            boots = []
            for _ in range(nboot):
                ix = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
                if np.std(p[ix]) > 0:
                    boots.append(np.corrcoef(t[ix], p[ix])[0, 1])
            r_err = float(np.std(boots)) if boots else np.nan
            sig = t.std() * np.sqrt(1 / r ** 2 - 1) if r > 0 else np.inf
            sl, ic = np.polyfit(t, p, 1)
            plim = np.quantile(np.abs(preds[:, k]), 0.995) * 1.15
            ax.scatter(t, p, s=4, alpha=0.3, color=col, linewidths=0, rasterized=True)
            e = np.quantile(t, np.linspace(0, 1, nbins + 1))
            w = np.clip(np.searchsorted(e, t, side="right") - 1, 0, nbins - 1)
            bc = [t[w == j].mean() for j in range(nbins) if (w == j).sum() >= 5]
            bm = [p[w == j].mean() for j in range(nbins) if (w == j).sum() >= 5]
            be = [p[w == j].std() / np.sqrt((w == j).sum()) for j in range(nbins)
                  if (w == j).sum() >= 5]
            ax.errorbar(bc, bm, yerr=be, fmt="o", color=col, ms=4, capsize=2,
                        label="binned mean")
            xs = np.array([-tlim, tlim])
            ax.plot(xs, ic + sl * xs, color=col, lw=1.4, label=f"slope {sl:.3f}")
            ax.plot(xs, xs, "k--", lw=0.8, label="1:1")
            ax.set_xlim(-tlim, tlim); ax.set_ylim(-plim, plim)
            ax.set_title(f"{name}, {label} {lo:g}–{hi:g}:  N={m.sum()}  "
                         f"r={r:+.3f}±{r_err:.3f}  σ_eff={sig:.3f}", fontsize=10)
            ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"{name} predicted")
            ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
            rows.append(dict(bin_lo=lo, bin_hi=hi, component=name, n=int(m.sum()),
                             r=r, r_err=r_err, sigma_eff=sig, slope=sl, intercept=ic))
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}")
    for d in rows:
        print(f"    {d['component']} {label} {d['bin_lo']:g}-{d['bin_hi']:g}: n={d['n']:5d}  "
              f"r={d['r']:+.3f}±{d['r_err']:.3f}  sigma_eff={d['sigma_eff']:.3f}  "
              f"slope={d['slope']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="test_predictions.csv, oof_predictions.csv or stack_predictions.csv")
    ap.add_argument("--col", default="truth.cosi", help="column to bin on")
    ap.add_argument("--edges", type=float, nargs="+", default=[0, 0.3, 0.7, 1.0])
    ap.add_argument("--parquet", default=None, help="to read --col if not in --pred")
    ap.add_argument("--which", default=None, help="prediction suffix: raw, pred or stack")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    suf = args.which or ("stack" if "g_plus_stack" in df else "raw")
    P = df[[f"g_plus_{suf}", f"g_cross_{suf}"]].to_numpy(float).copy()
    lab = df[["g_plus_true", "g_cross_true"]].to_numpy(float)
    note = suf
    if suf == "raw" and "fold" in df and df["fold"].nunique() > 1:
        # pooled folds: put each fold's raw output on a common scale
        for f in df["fold"].unique():
            m = (df["fold"] == f).to_numpy()
            P[m] = (P[m] - P[m].mean(0)) / (P[m].std(0) + 1e-12) * P.std(0)
        note = "raw, per-fold standardised"
    if args.col in df:
        vals = df[args.col].to_numpy(float)
    else:
        if not args.parquet or "src_row" not in df:
            raise SystemExit(f"'{args.col}' is not in {args.pred}; pass --parquet "
                             f"(and use a predictions file with src_row)")
        vals = pd.read_parquet(args.parquet, columns=[args.col])[args.col] \
            .to_numpy(float)[df["src_row"].to_numpy()]
    label = args.col.replace("truth.", "").replace("cosi", "cos i")
    out = Path(args.out) if args.out else Path(args.pred).with_name(
        f"pred_vs_true_by_{args.col.replace('.', '_')}_{suf}.png")
    groups = df["pop.id"].to_numpy() if "pop.id" in df else None
    plot_pred_vs_true_by_bins(P, lab, vals, args.edges, out,
                              f"{Path(args.pred).parent.name}: {note}", label=label,
                              groups=groups)


if __name__ == "__main__":
    main()