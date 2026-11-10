#!/usr/bin/env python3
"""Summary figure: test-set correlation vs inclination for a run, against the
oracle (perfect-observable) ceiling for g+, plus the calibrated response."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

run = Path(sys.argv[1])
out = sys.argv[2]
t = pd.read_parquet("/home/user/data/v3/table.parquet")
z = np.load(run / "pred_test.npz")
idx, pred, sig, true = z["idx"], z["pred"], z["sigma"], z["true"]
cosi = t["truth.cosi"].to_numpy()[idx]

edges = np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95])
cen = 0.5 * (edges[1:] + edges[:-1])
oracle_bins = {(0.05, 0.3): 0.34, (0.3, 0.6): 0.45, (0.6, 0.95): 0.85}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for k, (comp, col) in enumerate([("g+", "C0"), ("g×", "C1")]):
    r = [np.corrcoef(pred[(cosi >= lo) & (cosi < hi), k], true[(cosi >= lo) & (cosi < hi), k])[0, 1]
         for lo, hi in zip(edges[:-1], edges[1:])]
    ax[0].plot(cen, r, "o-", color=col, label=f"shearnet {comp}")
for (lo, hi), v in oracle_bins.items():
    ax[0].hlines(v, lo, hi, color="C0", ls="--", lw=1.5)
ax[0].plot([], [], "--", color="C0", label="g+ ceiling (perfect data)")
ax[0].axhline(1.0, color="C1", ls="--", lw=1.0, label="g× ceiling (perfect data)")
ax[0].set_xlabel("cos i (0 = edge-on)")
ax[0].set_ylabel("r(predicted, true) on test galaxies")
ax[0].set_ylim(0, 1.05)
ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title("Information vs inclination")

# calibrated, de-shrunk estimator: binned response on test
import json
cal = json.load(open(run / "calibration.json"))
for k, (comp, key, col) in enumerate([("g+", "g_plus", "C0"), ("g×", "g_cross", "C1")]):
    R = np.clip(1 - sig[:, k] ** 2 / C.G_SCALE**2, 0.02, 1)
    gh = (pred[:, k] / R - cal[key]["additive"]) / cal[key]["response"]
    w = R / sig[:, k] ** 2
    b = np.linspace(-0.1, 0.1, 9)
    mids = 0.5 * (b[1:] + b[:-1])
    mu, er = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        s = (true[:, k] >= lo) & (true[:, k] < hi)
        ww = w[s] * cal[key]["response"] ** 2
        mu.append(np.sum(ww * gh[s]) / np.sum(ww))
        er.append(1 / np.sqrt(np.sum(ww)))
    ax[1].errorbar(mids + (k - 0.5) * 0.004, mu, er, fmt="o", color=col, label=f"{comp} (weighted, calibrated)")
ax[1].plot([-0.1, 0.1], [-0.1, 0.1], "k--", lw=1)
ax[1].set_xlabel("true shear in bin")
ax[1].set_ylabel("ensemble estimate")
ax[1].set_title("Unbiased ensemble response (test)")
ax[1].legend(fontsize=8)

# comparison bars
labels = ["previous best\n(run04 imgfit_f0_v2)", "this work"]
prev = dict(rgp=0.372, rgx=0.491, sgp=0.145, sgx=0.10)
res = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else None
rgp = np.corrcoef(pred[:, 0], true[:, 0])[0, 1]
rgx = np.corrcoef(pred[:, 1], true[:, 1])[0, 1]
x = np.arange(2)
ax[2].bar(x - 0.2, [prev["rgp"], rgp], 0.4, label="r(g+)", color="C0")
ax[2].bar(x + 0.2, [prev["rgx"], rgx], 0.4, label="r(g×)", color="C1")
for xi, v in zip(list(x - 0.2) + list(x + 0.2), [prev["rgp"], rgp, prev["rgx"], rgx]):
    ax[2].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].set_ylim(0, 0.75)
ax[2].set_title("Test correlation (different test splits of\nunseen galaxies; ±0.02)")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=110)
print("saved", out)
