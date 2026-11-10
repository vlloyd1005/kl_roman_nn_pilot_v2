#!/usr/bin/env python3
"""Predict galaxy-frame shear (g+, gx) and per-galaxy uncertainties with one or
more trained shearnet checkpoints (several = ensemble).

Input: a parquet with the same layout as kl_roman_nn_v3_lsnr10x.parquet (the six
noisy img.* stamps, the sigma_bkg.* columns and the observable catalogue columns
listed in common.SCALAR_COLS).  Output: a CSV with one row per input row:
  g_plus, g_cross            posterior-mean estimates (shrunk toward 0 where the
                             data are uninformative -- see README, "Using the outputs")
  sigma_g_plus, sigma_g_cross  predicted per-galaxy uncertainty (ensemble spread added)
With --calib calibration.json (written by calibrate.py) also:
  ghat_plus, ghat_cross      de-shrunk, bias-calibrated per-galaxy estimates
                             (unbiased for an ensemble mean)
  sigma_ghat_plus/cross      their per-galaxy uncertainty; weight = 1/sigma^2

Example:
  python predict.py --ckpt runs/E4/model.pt runs/E4b/model.pt \
      --parquet new_galaxies.parquet --out shear_pred.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import common as C
from canon import CanonPreproc
from train2 import ShearNet2


def load_stamps(path):
    t = pq.read_table(path)
    n = t.num_rows
    x = np.zeros((n, 6, 32, 32), np.float32)
    for k, ch in enumerate(C.CHANNELS):
        col = t.column(f"img.{ch}").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        x[:, k] = col.reshape(n, 32, 32)
    scal = t.drop([c for c in t.column_names if c.startswith("img.")]).to_pandas()
    return x, scal


def model_from_ckpt(path):
    ck = torch.load(path, weights_only=False)
    a = ck["args"]
    pre = CanonPreproc(ck["sig"], ck["rolls"], crop=a["crop"], canon=not a["no_canon"], kin=a.get("kin", False))
    model = ShearNet2(pre.n_ch, ck["n_scal"], ck["n_aux"], w=a["width"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, pre, ck["s_mean"], ck["s_std"]


@torch.no_grad()
def predict(ckpts, x, scal, bs=512):
    mus, s2s = [], []
    for path in ckpts:
        model, pre, m, s = model_from_ckpt(path)
        S = ((C.scalars(scal) - m) / s).astype(np.float32)
        mu_l, sg_l = [], []
        for i in range(0, len(x), bs):
            xin, ang = pre(torch.from_numpy(x[i:i + bs]))
            mu, ls, _ = model(xin, torch.cat([torch.from_numpy(S[i:i + bs]), ang], 1))
            mu_l.append(mu.numpy())
            sg_l.append(torch.exp(ls.clamp(-4, 3)).numpy())
        mus.append(np.concatenate(mu_l) * C.G_SCALE)
        s2s.append((np.concatenate(sg_l) * C.G_SCALE) ** 2)
    mus = np.stack(mus)
    mu = mus.mean(0)
    sigma = np.sqrt(np.mean(s2s, 0) + mus.var(0))
    return mu, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--calib", default=None, help="calibration.json from calibrate.py")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    x, scal = load_stamps(a.parquet)
    mu, sigma = predict(a.ckpt, x, scal)
    out = pd.DataFrame({"g_plus": mu[:, 0], "g_cross": mu[:, 1],
                        "sigma_g_plus": sigma[:, 0], "sigma_g_cross": sigma[:, 1]})
    if a.calib:
        cal = json.load(open(a.calib))
        for k, comp in enumerate(["g_plus", "g_cross"]):
            R = np.clip(1 - sigma[:, k] ** 2 / C.G_SCALE**2, 0.02, 1)
            c = cal[comp]
            short = comp.split("_")[1]
            out[f"ghat_{short}"] = (mu[:, k] / R - c["additive"]) / c["response"]
            out[f"sigma_ghat_{short}"] = sigma[:, k] / np.sqrt(R) / c["response"]
    for c in ("db_row_id", "fit_id", "pop.id"):
        if c in scal:
            out.insert(0, c, scal[c].to_numpy())
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
