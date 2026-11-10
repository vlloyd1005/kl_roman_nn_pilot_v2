#!/usr/bin/env python3
"""Generate simulated noiseless stamps with the validated forward model.

Each simulation takes the physical parameters of a random TRAINING-split row
(sizes, fluxes, inclination, redshift, dispersion, gains) and re-draws the
quantities that are independent of the selection:
  theta_int ~ U(0, 2pi),  (g+, gx) ~ U(-0.1, 0.1)^2,
  vcirc = TF prior mean * 10^N(0, 0.049 dex),
  per-component centroids ~ N(0, 0.11"),  sign of v0.
Noise is NOT added here; training adds a fresh realisation each time.

Output (in --out): clean.npy (N,6,32,32) float32, table.parquet (truths + the
row's observables/gains, same column names as the real table).
"""
import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import forward as FW

TF_DEX = 0.049
KEEP = None
TABLE = None


def _init(table_path, psf_path):
    global TABLE
    TABLE = pd.read_parquet(table_path)
    FW.load_psf_table(psf_path)
    FW._psf("J129"), FW._psf("H158")


def _one(args):
    row_i, seed = args
    rng = np.random.default_rng(seed)
    r = TABLE.iloc[row_i]
    p = FW.truth_dict(r)
    th = rng.uniform(0, 2 * math.pi)
    gp, gx = rng.uniform(-0.1, 0.1, 2)
    g = (gp + 1j * gx) * np.exp(2j * th)
    p["theta_int"], p["g1"], p["g2"] = th, g.real, g.imag
    p["vel.vcirc"] = r["pop.prior_vcirc_mu_kms"] * 10 ** (TF_DEX * rng.standard_normal())
    p["vel.v0"] *= rng.choice([-1.0, 1.0])
    for comp, s in [("F129", 0.11), ("F158", 0.11), ("Halpha", 0.11), ("Halpha.cont", 0.123)]:
        for ax in ("x0", "y0"):
            v = rng.normal(0, s)
            while abs(v) > 0.5:
                v = rng.normal(0, s)
            p[f"{comp}.{ax}"] = v
    rolls = [r[f"roll_deg_{k}"] for k in range(4)]
    stamps = [FW.render_imaging(p, "F129"), FW.render_imaging(p, "F158")]
    stamps += [FW.render_grism(p, ro) for ro in rolls]
    rec = {f"truth.{k}": v for k, v in p.items()}
    rec["truth.g_plus"], rec["truth.g_cross"] = gp, gx
    rec["src_row"] = row_i
    return np.stack(stamps).astype(np.float32), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/user/data/v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--psf_table", default="/home/user/data/psf_os5.npz")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train",
                    help="which split's galaxies to take parameters from (train for training sims)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(Path(a.data) / "table.parquet")
    pool_rows = dict(zip(["train", "val", "test"], C.group_split(t, seed=0)))[a.split]
    rng = np.random.default_rng(a.seed)
    rows = rng.choice(pool_rows, a.n)
    seeds = rng.integers(0, 2**62, a.n)
    arr = np.lib.format.open_memmap(out / "clean.npy", mode="w+", dtype=np.float32,
                                    shape=(a.n, 6, 32, 32))
    recs = [None] * a.n
    with mp.Pool(a.workers, initializer=_init, initargs=(str(Path(a.data) / "table.parquet"), a.psf_table)) as pool:
        for i, (st, rec) in enumerate(pool.imap(_one, zip(rows, seeds), chunksize=64)):
            arr[i] = st
            recs[i] = rec
            if i % 5000 == 0:
                print(i, flush=True)
    arr.flush()
    sim = pd.DataFrame(recs)
    # carry over the base row's observables, gains and constant columns
    src = t.iloc[sim["src_row"].to_numpy()].reset_index(drop=True)
    carry = [c for c in t.columns if not c.startswith("truth.") and not c.startswith("img.")]
    sim = pd.concat([sim.reset_index(drop=True), src[carry]], axis=1)
    sim["pop.id"] = -1 - sim["src_row"]          # never collides with real ids
    sim.to_parquet(out / "table.parquet")
    print("done", a.n)


if __name__ == "__main__":
    main()
