"""Corner (triangle) plot of galaxy properties for the test galaxies of a
finished run, each point coloured by that galaxy's shear error.

One figure per shear component.  Variables (when present): the true shear of
that component, cos i, theta_int, z, v sin i, log R_d, log line S/N and
log F158 S/N.  Lower triangle: scatter, coloured by the error (largest
errors drawn on top).  Diagonal: histogram of the variable with the RMS
error in bins of it overlaid (right-hand axis), so the dependence of the
error on each property can be read off directly.

--kind picks the error (all in shear units):
  residual (default): prediction minus the model's mean response at the
      galaxy's true shear, pred - (slope * true + c), with slope and c fitted
      per fold.  It removes the shrinkage toward zero and any additive offset,
      so the colours show where the per-galaxy scatter is large rather than
      where the true shear is large.
  raw: prediction minus truth, as is (dominated by -(1 - slope) * true).
  corrected: (pred - c) / slope - true, the residual divided by the slope
      (raw out-of-fold predictions standardised per fold first).

Usage:
    python kl_corner_errors.py --pred run/cnn_k5_feat/oof_predictions.csv \
        --parquet kl_roman_nn_v3_lsnr10x/kl_roman_nn_v3_lsnr10x.parquet
    # also: a single fold's test_predictions.csv, or kl_stack.py's
    # stack_predictions.csv (add --props_from <an oof_predictions.csv> if it
    # lacks src_row)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE_CANDIDATES = ["truth.F158.disk_rscale", "truth.F158.rscale", "truth.disk_rscale",
                   "truth.rscale", "truth.F129.disk_rscale", "truth.F129.rscale",
                   "pop.rscale_arcsec"]


def load_props(parquet, rows, size_col=None):
    names = pq.read_schema(parquet).names
    size = size_col or next((c for c in SIZE_CANDIDATES if c in names), None)
    want = ["truth.g_plus", "truth.g_cross", "truth.cosi", "truth.theta_int", "truth.z",
            "truth.vel.vcirc", "snr_eff.line_total", "snr_eff.F158"] + ([size] if size else [])
    cols = [c for c in want if c in names]
    df = pd.read_parquet(parquet, columns=cols).iloc[rows].reset_index(drop=True)
    return df, size


def build_variables(P, size_col, comp):
    V = {}
    V[f"{comp} true"] = P["truth.g_plus" if comp == "g+" else "truth.g_cross"]
    if "truth.cosi" in P:
        V["cos i"] = P["truth.cosi"]
    if "truth.theta_int" in P:
        V["θ_int [rad]"] = np.mod(P["truth.theta_int"], 2 * np.pi)
    if "truth.z" in P:
        V["z"] = P["truth.z"]
    if "truth.vel.vcirc" in P and "truth.cosi" in P:
        V["v sin i [km/s]"] = P["truth.vel.vcirc"] * np.sqrt(1 - P["truth.cosi"] ** 2)
    if size_col and size_col in P:
        V["log R_d"] = np.log10(P[size_col].clip(lower=1e-6))
    if "snr_eff.line_total" in P:
        V["log S/N line"] = np.log10(P["snr_eff.line_total"].clip(lower=1e-3))
    if "snr_eff.F158" in P:
        V["log S/N F158"] = np.log10(P["snr_eff.F158"].clip(lower=1e-3))
    return pd.DataFrame(V)


def corner(V, err, out_path, title, nbins=12, vmax_q=0.95, max_points=20000, seed=0,
           signed=False, label_note=""):
    names = list(V.columns)
    k = len(names)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(V))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    mag = np.abs(err)
    idx = idx[np.argsort(mag[idx])]                     # largest errors on top
    vmax = np.quantile(mag, vmax_q)
    cmap, vmin = ("RdBu_r", -vmax) if signed else ("viridis", 0.0)
    lims = {c: tuple(np.quantile(V[c], [0.005, 0.995])) for c in names}
    fig, axes = plt.subplots(k, k, figsize=(2.3 * k + 1.2, 2.3 * k))
    sc = None
    for i, ci in enumerate(names):
        for j, cj in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                x = V[ci].to_numpy()
                lo, hi = lims[ci]
                ax.hist(x, bins=30, range=(lo, hi), color="0.8")
                ax.set_yticks([])
                edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
                which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
                bc = [np.median(x[which == b]) for b in range(nbins) if (which == b).sum() > 10]
                br = [np.sqrt(np.mean(mag[which == b] ** 2)) for b in range(nbins)
                      if (which == b).sum() > 10]
                tw = ax.twinx()
                tw.plot(bc, br, "o-", color="#C0392B", ms=3, lw=1.2)
                tw.set_ylim(0, max(br) * 1.15 if br else 1)
                tw.tick_params(axis="y", labelsize=6, colors="#C0392B")
                if i == 0:
                    tw.set_ylabel("RMS error", fontsize=7, color="#C0392B")
                ax.set_xlim(lo, hi)
            else:
                sc = ax.scatter(V[cj].to_numpy()[idx], V[ci].to_numpy()[idx], c=err[idx],
                                s=2, cmap=cmap, vmin=vmin, vmax=vmax,
                                rasterized=True, linewidths=0)
                ax.set_xlim(*lims[cj]); ax.set_ylim(*lims[ci])
            if i == k - 1:
                ax.set_xlabel(cj, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(ci, fontsize=8)
            elif j > 0 or i == 0:
                if i != j:
                    ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.06, top=0.94, wspace=0.06, hspace=0.06)
    if sc is not None:
        cax = fig.add_axes([0.90, 0.35, 0.015, 0.4])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(("error" if signed else "|error|")
                     + label_note, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="oof_predictions.csv, test_predictions.csv or stack_predictions.csv")
    ap.add_argument("--parquet", required=True, help="the parquet the run was trained on")
    ap.add_argument("--props_from", default=None,
                    help="an oof_predictions.csv with src_row, if --pred lacks it")
    ap.add_argument("--which", default=None, help="prediction suffix: raw, pred or stack")
    ap.add_argument("--size_col", default=None, help="disk scale column (auto-detected)")
    ap.add_argument("--error", choices=["abs", "signed"], default="abs",
                    help="colour by |error| (default) or signed error")
    ap.add_argument("--min_cosi", type=float, default=0.17,
                    help="plot only test galaxies with truth.cosi above this, matching "
                         "the training cut (0 = all); the response and errors are "
                         "computed on the plotted galaxies")
    ap.add_argument("--kind", choices=["residual", "raw", "corrected"], default="residual",
                    help="error definition (see the module docstring)")
    ap.add_argument("--corrected", action="store_true",
                    help="use the response-corrected estimate (pred - c)/slope "
                         "instead of the prediction as is")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    if "src_row" not in df and args.props_from:
        pf = pd.read_csv(args.props_from)
        key = "row" if "row" in df else None
        if key is None or "src_row" not in pf:
            raise SystemExit("cannot find src_row through --props_from")
        df = df.merge(pf[["row", "src_row"]], on="row", how="left")
    if "src_row" in df:
        rows = df["src_row"].to_numpy(int)
    elif "row" in df:
        print("[NOTE] no src_row column (run made before it existed): using 'row', "
              "which is only right if the run applied no S/N cut and no --max_rows")
        rows = df["row"].to_numpy(int)
    else:
        raise SystemExit("predictions file has neither src_row nor row")

    suf = args.which or ("stack" if "g_plus_stack" in df else "raw")
    P = df[[f"g_plus_{suf}", f"g_cross_{suf}"]].to_numpy(float).copy()
    lab = df[["g_plus_true", "g_cross_true"]].to_numpy(float)
    if args.corrected:
        args.kind = "corrected"
    folds = df["fold"].to_numpy() if "fold" in df else np.zeros(len(df), int)
    if args.kind == "corrected" and suf == "raw" and "fold" in df and df["fold"].nunique() > 1:
        for f in df["fold"].unique():
            m = (df["fold"] == f).to_numpy()
            P[m] = (P[m] - P[m].mean(0)) / (P[m].std(0) + 1e-12)

    props, size_col = load_props(args.parquet, rows, args.size_col)
    for k, c in enumerate(["truth.g_plus", "truth.g_cross"]):
        if c in props and not np.allclose(props[c].to_numpy(), lab[:, k], atol=1e-6):
            raise SystemExit(f"{c} in the parquet does not match the predictions file: "
                             f"wrong parquet, or rows misaligned (see the src_row note)")
    print(f"{len(df)} test galaxies; size column: {size_col or 'none found'}")
    if args.min_cosi > 0:
        if "truth.cosi" not in props:
            raise SystemExit("--min_cosi needs truth.cosi in the parquet")
        keep = (props["truth.cosi"] > args.min_cosi).to_numpy()
        P, lab, folds = P[keep], lab[keep], folds[keep]
        props = props[keep].reset_index(drop=True)
        print(f"cos i > {args.min_cosi}: plotting {keep.sum()}/{len(keep)} test galaxies "
              f"(--min_cosi 0 for all)")

    out = Path(args.outdir) if args.outdir else Path(args.pred).parent
    out.mkdir(parents=True, exist_ok=True)
    for k, comp in enumerate(["g+", "g×"]):
        t, p = lab[:, k], P[:, k]
        sl, ic = np.polyfit(t, p, 1)
        r = np.corrcoef(t, p)[0, 1]
        if args.kind == "corrected":
            if sl <= 0:
                print(f"  {comp}: no positive response (slope {sl:.3g}); skipped")
                continue
            e = (p - ic) / sl - t
        elif args.kind == "residual":
            e = np.empty_like(p)
            fits = []
            for f in np.unique(folds):        # each fold's model has its own response
                m = folds == f
                a, b = np.polyfit(t[m], p[m], 1)
                e[m] = p[m] - (a * t[m] + b)
                fits.append((a, b))
            print(f"  {comp}: per-fold response removed: slope "
                  f"{min(f[0] for f in fits):.3f}-{max(f[0] for f in fits):.3f}, "
                  f"offset {min(f[1] for f in fits):+.4f}-{max(f[1] for f in fits):+.4f}")
        else:
            e = p - t
        V = build_variables(props, size_col, comp)
        tag = (("gplus" if comp == "g+" else "gcross")
               + ("_signed" if args.error == "signed" else "")
               + ("" if args.kind == "residual" else f"_{args.kind}"))
        corner(V, e, out / f"corner_errors_{tag}.png",
               f"{Path(args.pred).parent.name}: {comp} test error"
               + (f", cos i > {args.min_cosi}" if args.min_cosi > 0 else "")
               + f"   (N = {len(t)}, "
               + f"r = {r:+.3f}, slope {sl:.3f}, RMS {np.sqrt(np.mean(e ** 2)):.4f})",
               signed=args.error == "signed",
               label_note={"residual": "  (pred − mean response)",
                           "corrected": "  (response-corrected)",
                           "raw": f"  ({suf} prediction − true)"}[args.kind])


if __name__ == "__main__":
    main()