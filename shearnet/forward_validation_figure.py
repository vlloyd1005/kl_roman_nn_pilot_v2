#!/usr/bin/env python3
"""Validation figure for forward.py: chi^2 of the model at the true parameters
against the noiseless stamps, for random rows, plus example residual maps."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forward as FW

data = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/v3"
out = sys.argv[2] if len(sys.argv) > 2 else "figs/forward_validation.png"
FW.load_psf_table(sys.argv[3] if len(sys.argv) > 3 else "/home/user/data/psf_os5.npz")
t = pd.read_parquet(f"{data}/table.parquet")
C = np.load(f"{data}/clean.npy", mmap_mode="r")
V = np.load(f"{data}/var.npy", mmap_mode="r")
rows = np.random.default_rng(0).choice(len(t), 200, replace=False)
rolls = [0, 10, 167, 177]
chi_im, chi_gr = [], []
for i in rows:
    p = FW.truth_dict(t.iloc[i])
    chi_im += [float(((FW.render_imaging(p, b) - C[i, k]) ** 2 / V[i, k]).sum()) for k, b in enumerate(["F129", "F158"])]
    chi_gr += [float(((FW.render_grism(p, r) - C[i, 2 + j]) ** 2 / V[i, 2 + j]).sum()) for j, r in enumerate(rolls)]
chi_im, chi_gr = np.array(chi_im), np.array(chi_gr)
fig = plt.figure(figsize=(14, 6.5))
ax = fig.add_subplot(2, 5, (1, 7))
b = np.logspace(-2, 4, 50)
ax.hist(chi_im, b, histtype="step", lw=2, label=f"imaging (median {np.median(chi_im):.2f})")
ax.hist(chi_gr, b, histtype="step", lw=2, label=f"grism (median {np.median(chi_gr):.1f})")
ax.axvline(1024, color="k", ls="--", label="pure noise (1024 px)")
ax.set_xscale("log")
ax.set_xlabel(r"$\chi^2$ per 32x32 stamp: (model - noiseless)$^2$ / var")
ax.set_ylabel("stamps")
ax.set_title(f"forward.py vs noiseless mocks, {len(rows)} random rows")
ax.legend(fontsize=9)
ex = rows[np.argsort([chi_gr[4 * k:4 * k + 4].sum() for k in range(len(rows))])[len(rows) // 2]]
p = FW.truth_dict(t.iloc[ex])
for j, r in enumerate(rolls[:3]):
    m = FW.render_grism(p, r)
    a = fig.add_subplot(2, 5, 3 + j)
    a.imshow(C[ex, 2 + j][6:26, 6:26], origin="lower")
    a.set_title(f"noiseless roll {r}", fontsize=9)
    a.axis("off")
    a = fig.add_subplot(2, 5, 8 + j)
    a.imshow(((C[ex, 2 + j] - m) / np.sqrt(V[ex, 2 + j]))[6:26, 6:26], origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    a.set_title("residual / noise sigma (+-1)", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=100)
print(f"imaging chi2 median {np.median(chi_im):.2f} p95 {np.percentile(chi_im, 95):.1f}; "
      f"grism chi2 per stamp median {np.median(chi_gr):.1f} mean {chi_gr.mean():.1f} p95 {np.percentile(chi_gr, 95):.0f}")
