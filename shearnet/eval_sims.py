#!/usr/bin/env python3
"""Evaluate checkpoints on a simulation set (fixed noise seed) to measure the
sim-domain performance, for comparison with real held-out galaxies."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as C
from predict import model_from_ckpt
from train2 import Source


def patch_ckpt(path, data="/home/user/data/v3"):
    """Older checkpoints lack the input-normalisation stats; recompute them
    exactly as train2.py does (same deterministic split)."""
    ck = torch.load(path, weights_only=False)
    if "s_mean" in ck:
        return
    t, _, _, _, sig, rolls = C.load(data)
    tr, _, _ = C.group_split(t, seed=0, fold=ck["args"].get("fold"))
    S = C.scalars(t)
    ck.update(s_mean=S[tr].mean(0), s_std=S[tr].std(0), sig=sig, rolls=rolls,
              n_scal=S.shape[1] + (15 if ck["args"].get("kin") else 5), n_aux=C.aux_targets(t).shape[1])
    torch.save(ck, path)
    print("patched", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--sims", required=True)
    ap.add_argument("--data", default="/home/user/data/v3")
    a = ap.parse_args()
    torch.set_num_threads(4)
    st = pd.read_parquet(Path(a.sims) / "table.parquet")
    sc = np.load(Path(a.sims) / "clean.npy")
    sig = np.array([st[f"sigma_bkg.{c}"].iloc[0] for c in C.CHANNELS], np.float32)
    src = Source(sc, st, sig)
    x = src.noisy(np.arange(len(st)), torch.Generator().manual_seed(123)).numpy()
    y = st[C.TARGETS].to_numpy()
    for p in a.ckpt:
        patch_ckpt(p, a.data)
        model, pre, m, s = model_from_ckpt(p)
        S = ((C.scalars(st) - m) / s).astype(np.float32)
        with torch.no_grad():
            mu = []
            for i in range(0, len(x), 512):
                xin, ang = pre(torch.from_numpy(x[i:i + 512]))
                mu.append(model(xin, torch.cat([torch.from_numpy(S[i:i + 512]), ang], 1))[0].numpy())
        mu = np.concatenate(mu) * C.G_SCALE
        r = [np.corrcoef(mu[:, k], y[:, k])[0, 1] for k in range(2)]
        print(f"{p}: sims {a.sims} n={len(y)}  r_g+ {r[0]:.3f}  r_gx {r[1]:.3f}")


if __name__ == "__main__":
    main()
