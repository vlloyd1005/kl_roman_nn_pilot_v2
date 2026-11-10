#!/usr/bin/env python3
"""Write pred_val.npz / pred_test.npz (real held-out galaxies) for one or more
checkpoints (several = ensemble), in the format evaluate.py reads."""
import argparse
from pathlib import Path

import numpy as np
import torch

import common as C
from predict import predict

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/home/user/data/v3")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
t, noisy, _, _, _, _ = C.load(a.data)
tr, va, te = C.group_split(t, seed=0)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for name, idx in [("val", va), ("test", te)]:
    mu, sig = predict(a.ckpt, noisy[idx], t.iloc[idx])
    np.savez(out / f"pred_{name}.npz", idx=idx, pred=mu, sigma=sig, true=t[C.TARGETS].to_numpy()[idx])
    r = [np.corrcoef(mu[:, k], t[C.TARGETS].to_numpy()[idx, k])[0, 1] for k in range(2)]
    print(name, "r_g+ %.3f r_gx %.3f" % tuple(r))
