#!/usr/bin/env python3
"""Calibrate the de-shrunk estimator on the validation galaxies and apply to test.

Per galaxy: R = 1 - sigma^2/sigma_prior^2, g_hat = pred/R, var = sigma^2/R.
On validation, fit the inverse-variance-weighted linear map g_hat = (1+m) g + c
per component, then report the test ensemble metrics of (g_hat - c)/(1+m):
  * weighted multiplicative / additive bias (should be ~0 within errors),
  * effective per-galaxy noise sqrt(N * var(weighted mean shear)), computed from
    the actual test residuals (not the predicted sigma).
Writes calibration.json next to the predictions (used by predict.py --calib).
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

run = Path(sys.argv[1])
v = np.load(run / "pred_val.npz")
t = np.load(run / "pred_test.npz")


def deshrink(z, k):
    R = np.clip(1 - z["sigma"][:, k] ** 2 / C.G_SCALE**2, 0.02, 1)
    return z["pred"][:, k] / R, z["sigma"][:, k] ** 2 / R


def wfit(g, est, var):
    w = 1 / var
    A = np.column_stack([g, np.ones_like(g)]) * np.sqrt(w)[:, None]
    (a, b), *_ = np.linalg.lstsq(A, est * np.sqrt(w), rcond=None)
    res = est - a * g - b
    # error on a from weighted least squares with empirical residual scale
    cov = np.linalg.inv(A.T @ A) * (np.sum(w * res**2) / (len(g) - 2))
    return a, b, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), res, w


cal = {}
for k, comp in enumerate(["g_plus", "g_cross"]):
    ev, vv = deshrink(v, k)
    a, b, *_ = wfit(v["true"][:, k], ev, vv)
    et, vt = deshrink(t, k)
    ec, vc = (et - b) / a, vt / a**2
    a2, b2, ea, eb, res, w = wfit(t["true"][:, k], ec, vc)
    sig_eff = np.sqrt(np.sum(w**2 * res**2) / np.sum(w) ** 2 * len(res))
    pull = res / np.sqrt(vc)
    cal[comp] = dict(response=float(a), additive=float(b))
    print(f"{comp}: val calibration 1+m={a:.3f} c={b:+.4f} | TEST after calibration: "
          f"m={a2 - 1:+.3f}±{ea:.3f}  c={b2:+.4f}±{eb:.4f}  sig_eff(per galaxy, inv-var weighted)={sig_eff:.3f}  "
          f"pull std={pull.std():.2f}")
json.dump(cal, open(run / "calibration.json", "w"), indent=1)
