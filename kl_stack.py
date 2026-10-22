"""Combine the out-of-fold predictions of several k-fold runs of
train_kl_grism_v3.py (e.g. the feature baseline and the CNN) into one
shear estimate per component.

Cross-fitting: the combination weights used for the galaxies of fold f are
fitted only on the other folds, so no galaxy is scored with weights that
saw its own label.  Each model's raw predictions are first standardised per
fold (raw scales differ between folds).

Usage:
    python kl_stack.py --runs feat_k5 cnn_k5 --outdir stack_k5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMP = [("g_plus", "g+"), ("g_cross", "g×")]


def ensemble_response(preds, labels, groups, nboot=500, seed=0):
    """m, c, r per component with bootstrap errors over whole galaxies."""
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.default_rng(seed)

    def stats(ix):
        out = []
        for k in range(labels.shape[1]):
            t, p = labels[ix, k], preds[ix, k]
            sl, ic = np.polyfit(t, p, 1)
            r = np.corrcoef(t, p)[0, 1] if np.std(p) > 0 else 0.0
            out.append((sl - 1, ic, r))
        return np.array(out)

    base = stats(np.arange(len(labels)))
    boots = [stats(np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))]))
             for _ in range(nboot)]
    return base, np.nanstd(np.array(boots), axis=0)


def binned_panel(ax, t, p, col, label, nbins=10):
    edges = np.quantile(t, np.linspace(0, 1, nbins + 1))
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
    bc = [t[which == b].mean() for b in range(nbins)]
    bm = [p[which == b].mean() for b in range(nbins)]
    be = [p[which == b].std() / np.sqrt((which == b).sum()) for b in range(nbins)]
    sl = np.polyfit(t, p, 1)[0]
    ax.errorbar(bc, bm, yerr=be, fmt="o-", color=col, capsize=3, ms=4,
                label=f"{label} (slope {sl:.3f})")


def load(run):
    df = pd.read_csv(Path(run) / "oof_predictions.csv")
    need = {"row", "fold", "g_plus_raw", "g_cross_raw", "g_plus_true", "g_cross_true"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{run}: oof_predictions.csv lacks {sorted(miss)}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="k-fold output directories (each with oof_predictions.csv)")
    ap.add_argument("--names", nargs="+", default=None, help="labels for the runs")
    ap.add_argument("--outdir", default="stack")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="ridge penalty on the weights (in units of n)")
    ap.add_argument("--nboot", type=int, default=500)
    args = ap.parse_args()
    names = args.names or [Path(r).name for r in args.runs]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # ── join the runs on the parquet row ─────────────────────────────────────
    dfs = [load(r) for r in args.runs]
    jkey = "src_row" if all("src_row" in d for d in dfs) else "row"
    if jkey == "row":
        print("NOTE: joining on 'row' (index after the S/N cut).  This is only valid\n"
              "      when every run used the same parquet, --snr_col and --min_line_snr.\n"
              "      Rerun with the current train_kl_grism_v3.py to get 'src_row'.")
    base = None
    for df, nm in zip(dfs, names):
        keep = [jkey, "fold", "g_plus_raw", "g_cross_raw", "g_plus_true", "g_cross_true"]
        if base is None and "pop.id" in df:
            keep.append("pop.id")
        df = df[keep].rename(columns={"g_plus_raw": f"gp_{nm}", "g_cross_raw": f"gx_{nm}",
                                      "fold": f"fold_{nm}", "g_plus_true": f"tp_{nm}",
                                      "g_cross_true": f"tx_{nm}"})
        base = df if base is None else base.merge(df, on=jkey, how="inner")
    n = len(base)
    print(f"Joined {len(names)} runs on '{jkey}': {n} galaxies in common "
          f"(run sizes: {', '.join(f'{nm} {len(d)}' for nm, d in zip(names, dfs))})")
    if n < 50:
        raise SystemExit("Too few galaxies in common to stack.")
    # the same galaxy must carry the same true shear in every run
    for nm in names[1:]:
        bad = ~(np.isclose(base[f"tp_{nm}"], base[f"tp_{names[0]}"], atol=1e-6) &
                np.isclose(base[f"tx_{nm}"], base[f"tx_{names[0]}"], atol=1e-6))
        if bad.any():
            raise SystemExit(
                f"{bad.sum()}/{n} joined galaxies have different true shears in "
                f"'{names[0]}' and '{nm}': the runs are not matched galaxy by galaxy.\n"
                f"They were probably made with different S/N cuts or parquets.  Rerun "
                f"both with the same --parquet/--snr_col/--min_line_snr (and the current "
                f"script, which writes 'src_row').")
    base["g_plus_true"], base["g_cross_true"] = base[f"tp_{names[0]}"], base[f"tx_{names[0]}"]
    folds0 = base[f"fold_{names[0]}"].to_numpy()
    same = all((base[f"fold_{nm}"].to_numpy() == folds0).all() for nm in names[1:])
    if not same:
        print("  Note: the runs used different fold assignments.  Each prediction is\n"
              "  still out-of-fold for its own model, so the stack is valid; it is\n"
              "  cross-fitted on the first run's folds.")
    folds = np.unique(folds0)
    groups = base["pop.id"].to_numpy() if "pop.id" in base else np.arange(n)
    lab = base[["g_plus_true", "g_cross_true"]].to_numpy()

    # ── per-fold standardisation of each model's raw output ─────────────────
    Z = {}
    for nm in names:
        for key in ("gp", "gx"):
            v = base[f"{key}_{nm}"].to_numpy().astype(np.float64).copy()
            own = base[f"fold_{nm}"].to_numpy()          # this model's own folds
            for f in np.unique(own):
                m = own == f
                v[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-12)
            Z[f"{key}_{nm}"] = v

    # ── cross-fitted linear stack ────────────────────────────────────────────
    stack = np.zeros((n, 2))
    weights = []
    for k, key in enumerate(["gp", "gx"]):
        X = np.stack([Z[f"{key}_{nm}"] for nm in names], 1)
        for f in folds:
            tr, te = folds0 != f, folds0 == f
            A = X[tr]
            # lstsq rather than solve: identical or collinear inputs (e.g. the
            # same run passed twice) give a minimum-norm split, not a crash
            G = A.T @ A + args.ridge * tr.sum() * np.eye(A.shape[1])
            w = np.linalg.lstsq(G, A.T @ lab[tr, k], rcond=1e-10)[0]
            stack[te, k] = X[te] @ w
            weights.append({"component": COMP[k][1], "fold": f,
                            **{nm: w[j] for j, nm in enumerate(names)}})
    wdf = pd.DataFrame(weights)

    # ── report ───────────────────────────────────────────────────────────────
    print("\nWeights per held-out fold (on per-fold standardised predictions):")
    print(wdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nCorrelation between the models' predictions (same component):")
    for k, key in enumerate(["gp", "gx"]):
        C = np.corrcoef(np.stack([Z[f"{key}_{nm}"] for nm in names]))
        print(f"  {COMP[k][1]}: " + "  ".join(
            f"{names[i]}~{names[j]} {C[i, j]:+.2f}"
            for i in range(len(names)) for j in range(i + 1, len(names))))

    rows = []
    series = [(nm, np.stack([Z[f"gp_{nm}"], Z[f"gx_{nm}"]], 1) * lab.std(0)) for nm in names]
    series.append(("stack", stack))
    print(f"\n  {'':>14} {'comp':>4} {'m (response − 1)':>18} {'c':>20} {'r':>16}  r/err")
    for nm, P in series:
        b, e = ensemble_response(P, lab, groups, nboot=args.nboot)
        for k in range(2):
            m, c, r = b[k]; em, ec, er = e[k]
            print(f"  {nm:>14} {COMP[k][1]:>4} {m:+8.3f} ± {em:.3f} {c:+10.5f} ± {ec:.5f}"
                  f" {r:+7.3f} ± {er:.3f}  {r / er if er > 0 else 0:5.1f}σ")
            rows.append({"model": nm, "component": COMP[k][1], "m": m, "m_err": em,
                         "c": c, "c_err": ec, "r": r, "r_err": er})
    print("  (single models: per-fold standardised raw output scaled to the shear spread;\n"
          "   stack: cross-fitted least-squares estimate of g, so 1+m is its calibration)")

    # ── outputs ──────────────────────────────────────────────────────────────
    res = base[[jkey, "fold_" + names[0]] + (["pop.id"] if "pop.id" in base else [])].copy()
    res = res.rename(columns={"fold_" + names[0]: "fold"})
    res["g_plus_true"], res["g_cross_true"] = lab[:, 0], lab[:, 1]
    res["g_plus_stack"], res["g_cross_stack"] = stack[:, 0], stack[:, 1]
    res.to_csv(out / "stack_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "stack_summary.csv", index=False)
    wdf.to_csv(out / "stack_weights.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    palette = ["#8E8E8E", "#2E86AB", "#F2A541", "#6A4C93"]
    for k, ax in enumerate(axes):
        t = lab[:, k]
        for i, (nm, P) in enumerate(series[:-1]):
            # put each single model on the stack's scale with a fit-free rescale
            s = np.std(stack[:, k]) / (np.std(P[:, k]) + 1e-12)
            binned_panel(ax, t, P[:, k] * s, palette[i % len(palette)], nm)
        binned_panel(ax, t, stack[:, k], "#E84855", "stack")
        lim = np.quantile(np.abs(t), 0.98)
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8, label="1:1")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"{COMP[k][1]} true"); ax.set_ylabel(f"mean {COMP[k][1]} predicted")
        ax.set_title(f"{COMP[k][1]}: out-of-fold binned response")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "binned_response_stack.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kl_plot_ensemble import plot_ensemble_pred_vs_true
        print("\nStack predicted vs true:")
        plot_ensemble_pred_vs_true(stack, lab, out / "pred_vs_true_stack.png",
                                   f"stack of {', '.join(names)} (cross-fitted)",
                                   groups=groups)
    except ImportError:
        print("(put kl_plot_ensemble.py next to kl_stack.py for pred_vs_true_stack.png)")
    print(f"\nSaved stack_predictions.csv, stack_summary.csv, stack_weights.csv and "
          f"binned_response_stack.png in {out.resolve()}")


if __name__ == "__main__":
    main()