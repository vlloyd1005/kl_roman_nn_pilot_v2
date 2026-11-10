#!/usr/bin/env python3
"""Evaluate a shearnet prediction file (pred_test.npz) in shear-pipeline terms.

Prints / plots, per component:
  * Pearson r, raw RMSE, linear response (1+m) and additive bias c;
  * sig_eff: per-galaxy noise of the response-corrected (unbiased) estimator,
    unweighted and with inverse-variance weights from the predicted sigma;
  * the same in bins of cos(i) and line SNR;
  * calibration of the predicted sigma (pull = residual / sigma).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--data", default="/home/user/data/v3")
ap.add_argument("--split", default="test")
ap.add_argument("--out", default=None)
a = ap.parse_args()

t = pd.read_parquet(Path(a.data) / "table.parquet")
summary = {}
for run in a.runs:
    z = np.load(Path(run) / f"pred_{a.split}.npz")
    idx, pred, sig, true = z["idx"], z["pred"], z["sigma"], z["true"]
    cosi = t["truth.cosi"].to_numpy()[idx]
    snr = t["snr.line_total"].to_numpy()[idx]
    res = {}
    print(f"\n=== {run}  ({a.split}, n={len(idx)})")
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for k, comp in enumerate(["g+", "gx"]):
        m = C.calib_metrics(pred[:, k], true[:, k])
        m["sig_eff_w"] = C.weighted_sig_eff(pred[:, k], true[:, k], sig[:, k])
        pull = (pred[:, k] - true[:, k]) / sig[:, k]
        m["pull_std"] = float(pull.std())
        res[comp] = m
        print(f"{comp}: r={m['r']:.3f} rmse={m['rmse']:.4f} 1+m={1 + m['m']:.3f}±{m['m_err']:.3f} "
              f"c={m['c']:+.4f} sig_eff={m['sig_eff']:.3f} sig_eff(weighted)={m['sig_eff_w']:.3f} pull_std={m['pull_std']:.2f}")
        # de-shrunk per-galaxy estimator: g_hat = pred / R, R = 1 - sigma^2 / sigma_prior^2,
        # likelihood variance sigma^2 / R; inverse-variance weighted ensemble
        R = np.clip(1 - sig[:, k] ** 2 / C.G_SCALE**2, 0.02, 1)
        gh, vl = pred[:, k] / R, sig[:, k] ** 2 / R
        w = 1 / vl
        A = np.column_stack([true[:, k], np.ones(len(gh))]) * np.sqrt(w)[:, None]
        (a1, b1), *_ = np.linalg.lstsq(A, gh * np.sqrt(w), rcond=None)
        pull_l = (gh - true[:, k]) / np.sqrt(vl)
        neff_sig = 1 / np.sqrt(np.sum(w) / len(w))
        res[comp]["deshrunk_response"] = float(a1)
        res[comp]["deshrunk_sig_per_gal"] = float(neff_sig)
        print(f"     de-shrunk g/R: weighted response {a1:.3f} (target 1), additive {b1:+.4f}, "
              f"pull std {pull_l.std():.2f}, predicted per-galaxy sigma (inverse-variance) {neff_sig:.3f}")
        for lo, hi in [(0.05, 0.3), (0.3, 0.6), (0.6, 0.95)]:
            s = (cosi >= lo) & (cosi < hi)
            mm = C.calib_metrics(pred[s, k], true[s, k])
            res[f"{comp}_cosi_{lo}_{hi}"] = mm
            print(f"     cosi [{lo:.2f},{hi:.2f}) n={s.sum():4d} r={mm['r']:.3f} sig_eff={mm['sig_eff']:.3f}")
        # binned response
        bins = np.linspace(-0.1, 0.1, 11)
        cen = 0.5 * (bins[1:] + bins[:-1])
        mu = [pred[(true[:, k] >= lo) & (true[:, k] < hi), k].mean() for lo, hi in zip(bins[:-1], bins[1:])]
        ax[k].scatter(true[:, k], pred[:, k], s=2, alpha=0.25, color="0.6")
        ax[k].plot(cen, mu, "o-", color="C0", label="binned mean")
        ax[k].plot([-0.1, 0.1], [-0.1, 0.1], "k--", lw=1, label="1:1")
        ax[k].plot(cen, (1 + m["m"]) * cen + m["c"], color="C3", lw=1, label=f"response {1 + m['m']:.2f}")
        ax[k].set_xlabel(f"true {comp}")
        ax[k].set_ylabel(f"predicted {comp}")
        ax[k].set_title(f"{comp}: r={m['r']:.3f}")
        ax[k].legend(fontsize=8)
        # r vs cos i
        cb = np.linspace(0.05, 0.95, 7)
        rr = [np.corrcoef(pred[(cosi >= lo) & (cosi < hi), k], true[(cosi >= lo) & (cosi < hi), k])[0, 1]
              for lo, hi in zip(cb[:-1], cb[1:])]
        ax[2].plot(0.5 * (cb[1:] + cb[:-1]), rr, "o-", label=comp)
        ax[3].hist(np.clip(pull, -5, 5), 60, histtype="step", density=True, label=f"{comp} std={pull.std():.2f}")
    ax[2].set_xlabel("cos i")
    ax[2].set_ylabel("r(pred, true)")
    ax[2].legend()
    ax[2].set_title("information vs inclination")
    xx = np.linspace(-5, 5, 200)
    ax[3].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
    ax[3].set_title("pull = (pred - true) / sigma_pred")
    ax[3].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(run) / f"eval_{a.split}.png", dpi=90)
    plt.close()
    summary[run] = res
if a.out:
    json.dump(summary, open(a.out, "w"), indent=1)
