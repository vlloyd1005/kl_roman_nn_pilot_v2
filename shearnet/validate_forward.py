#!/usr/bin/env python3
"""Check the forward model against the noiseless stamps: chi^2 = sum (model-clean)^2/var
per stamp (pure noise would give ~1024), for a grid of convention choices."""
import argparse
import itertools

import numpy as np
import pandas as pd

import forward as FW

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--grid", action="store_true")
ap.add_argument("--set", nargs="*", default=[], help="key=value convention overrides")
a = ap.parse_args()

t = pd.read_parquet("/home/user/data/v3/table.parquet")
C = np.load("/home/user/data/v3/clean.npy", mmap_mode="r")
V = np.load("/home/user/data/v3/var.npy", mmap_mode="r")
rolls = t[[f"roll_deg_{k}" for k in range(4)]].iloc[0].to_numpy()

base = {}
for kv in a.set:
    k, v = kv.split("=")
    base[k] = type(FW.CONV[k])(float(v)) if not isinstance(FW.CONV[k], bool) else v == "1"

if a.grid:
    combos = [dict(roll_sign=r, vel_sign=v, disp_sign=d)
              for r, v, d in itertools.product([1, -1], [1, -1], [1, -1])]
else:
    combos = [{}]
for cv in combos:
    cv = dict(base, **cv)
    chis = []
    for i in a.rows:
        p = FW.truth_dict(t.iloc[i])
        row = []
        for j in range(4):
            m = FW.render_grism(p, rolls[j], cv)
            row.append(float(((m - C[i, 2 + j]) ** 2 / V[i, 2 + j]).sum()))
        chis.append(row)
    chis = np.array(chis)
    print(cv, "median chi2 per stamp", np.round(np.median(chis, 0), 1), "max", np.round(chis.max(), 1))
