#!/usr/bin/env python3
"""
kl_diagnostics.py
=================
No training.  Uses the NOISELESS stamps and the truth columns to check, step
by step, whether the shear signal is in the data and whether every convention
the training script relies on is right.  Needs train_kl_grism.py alongside.

  1. Label convention   do truth.g_plus / g_cross follow from g1, g2 and
                        theta_int, and with which sign of theta?
  2. Image orientation  does the measured photometric position angle match
                        theta_int, and under which mapping (theta, -theta,
                        90-theta)?  Fixes how array axes relate to the sky.
  3. Oracle shear test  rotate the measured ellipticity into the TRUE galaxy
                        frame.  The intrinsic cross ellipticity is zero there,
                        so e_x should track g_cross almost perfectly in
                        noiseless data.  If this fails, the labels and images
                        disagree and no network can learn shear.
  4. Kinematics / roll  fit the velocity gradient from each grism roll pair for
                        both roll signs.  The right sign gives a kinematic
                        axis aligned with theta_int and an amplitude that
                        tracks v sin i.  This settles the roll sign.
  5. KL observable      misalignment between kinematic and photometric axes
                        vs g_cross: the signal a network has to find WITHOUT
                        truth.  Its strength is the realistic ceiling.
  6. Sizes              galaxy size in pixels (resolution check).

Usage (Colab):
  !python kl_diagnostics.py --parquet "{DATA}/pilot_v2_all.parquet" \
      --meta "{DATA}/meta_pilot_v2_all.json" --outdir "{OUT}/diagnostics"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import train_kl_grism as T


def wrap_axis(a):
    """Wrap an axis-angle difference into (-pi/2, pi/2]."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def axis_agreement(a, b):
    """Mean cos 2(a - b): 1 = axes aligned, 0 = unrelated, -1 = perpendicular."""
    return float(np.mean(np.cos(2 * (a - b))))


def rs(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 10 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--meta", default=None)
    p.add_argument("--outdir", default="./kl_diagnostics")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--weight_sigma", type=float, default=4.0,
                   help="Gaussian weight (px) for moments of noiseless stamps")
    p.add_argument("--incl_cut", type=float, default=0.6,
                   help="'Inclined' subset: cos i below this")
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, meta, noisy, clean, var, sigma = T.load_table(args.parquet, args.meta)
    if args.max_rows:
        m = args.max_rows
        df = df.iloc[:m].reset_index(drop=True)
        noisy, clean, var, sigma = noisy[:m], clean[:m], var[:m], sigma[:m]
    n, npix = len(df), clean.shape[-1]
    rolls = T.get_roll_angles(df, meta, None)

    need = ["truth.theta_int", "truth.cosi", "truth.g_plus", "truth.g_cross"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")
    theta = df["truth.theta_int"].to_numpy(np.float64)
    cosi  = df["truth.cosi"].to_numpy(np.float64)
    gp    = df["truth.g_plus"].to_numpy(np.float64)
    gx    = df["truth.g_cross"].to_numpy(np.float64)
    incl  = cosi < args.incl_cut
    vsini = (df["truth.vel.vcirc"].to_numpy() * np.sqrt(1 - cosi ** 2)
             if "truth.vel.vcirc" in df.columns else None)
    verdict = []

    # ── 1. Label convention ──────────────────────────────────────────────────
    print("\n1. LABEL CONVENTION  (g_plus/g_cross recomputed from g1, g2, theta)")
    if {"truth.g1", "truth.g2"} <= set(df.columns):
        g1, g2 = df["truth.g1"].to_numpy(), df["truth.g2"].to_numpy()
        for name, th in [("theta", theta), ("-theta", -theta)]:
            c, s = np.cos(2 * th), np.sin(2 * th)
            dp = np.abs(g1 * c + g2 * s - gp).max()
            dx_pos = np.abs(-g1 * s + g2 * c - gx).max()
            dx_neg = np.abs(-g1 * s + g2 * c + gx).max()
            print(f"   using {name:>6}: max|Δg+| = {dp:.2e}   "
                  f"max|Δg×| = {dx_pos:.2e} (same sign) / {dx_neg:.2e} (flipped)")
    else:
        print("   truth.g1/g2 not present; skipped")

    # ── 2. Image orientation ─────────────────────────────────────────────────
    xx, yy = T.pixel_coords(npix, "cpu")
    W = torch.exp(-(xx ** 2 + yy ** 2) / (2 * args.weight_sigma ** 2))
    coadd = torch.from_numpy(clean[:, 0] + clean[:, 1])
    flux, qxx, qyy, qxy = [t.numpy().astype(np.float64)
                           for t in T.weighted_moments(coadd, xx, yy, W)]
    Tsz = qxx + qyy
    e1, e2 = (qxx - qyy) / Tsz, 2 * qxy / Tsz
    phi = 0.5 * np.arctan2(e2, e1)          # photometric axis, x = column, y = row

    print("\n2. IMAGE ORIENTATION  mean cos2(φ_measured − ψ) for inclined galaxies"
          " (1 = match)")
    cands = {"theta": theta, "-theta": -theta, "90-theta": np.pi / 2 - theta,
             "theta+90": theta + np.pi / 2}
    scores = {k: axis_agreement(phi[incl], v[incl]) for k, v in cands.items()}
    for k, v in scores.items():
        print(f"   ψ = {k:>9}: {v:+.3f}")
    best = max(scores, key=scores.get)
    psi = cands[best]
    print(f"   -> theta_int maps to array angle ψ = {best}")
    if scores[best] < 0.8:
        verdict.append("Photometric PA does not clearly match theta_int under any "
                       "simple mapping: check the stamp orientation / theta "
                       "definition with the data producers.")

    # ── 3. Oracle shear test ─────────────────────────────────────────────────
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    e_plus = e1 * c2 + e2 * s2
    e_cross = -e1 * s2 + e2 * c2
    X = np.column_stack([np.ones(n), cosi, cosi ** 2, cosi ** 3])
    if "truth.h_over_r" in df.columns:
        h = df["truth.h_over_r"].to_numpy()
        X = np.column_stack([X, h, h * cosi])
    coef, *_ = np.linalg.lstsq(X, e_plus, rcond=None)
    e_plus_res = e_plus - X @ coef
    r_x, r_p = np.corrcoef(e_cross, gx)[0, 1], np.corrcoef(e_plus_res, gp)[0, 1]
    print("\n3. ORACLE SHEAR TEST  (noiseless ellipticity in the TRUE galaxy frame)")
    print(f"   corr(e×, g×)                    = {r_x:+.3f}   "
          f"slope de×/dg× = {np.polyfit(gx, e_cross, 1)[0]:+.2f}")
    print(f"   corr(e+ − f(cos i, h/r), g+)    = {r_p:+.3f}   "
          f"slope = {np.polyfit(gp, e_plus_res, 1)[0]:+.2f}")
    print("   (expect |r| ≳ 0.8 and |slope| of order 1–2 if images carry the shear)")
    if abs(r_x) < 0.5:
        verdict.append("ORACLE FAILS: even with the true orientation, noiseless "
                       "ellipticity barely tracks g×. Images and labels disagree "
                       "(label convention, wrong stamp, or shear not applied) — "
                       "no network can learn this. Raise with the data producers.")
    elif r_x < 0:
        verdict.append("Oracle works but with NEGATIVE sign: the g× handedness "
                       "is opposite to the image axes. Harmless for training "
                       "(a network learns the sign) but worth knowing.")

    # ── 4. Kinematic axis and roll sign ──────────────────────────────────────
    print("\n4. KINEMATICS  velocity-gradient fit from roll pairs, per roll sign")
    pairs = [(0, 3), (1, 2)]
    data = T.DeviceData(noisy, clean, var, sigma, rolls,
                        df[T.TARGETS].to_numpy(np.float32), None, None, dev)
    kin = {}
    for sgn in (+1.0, -1.0):
        prep = T.GPUPreprocessor(sgn, True, pairs, 2.0, False, False, False,
                                 True, False, 0.5, 2.0, 1.0, 3.0, npix, dev)
        prep.noise_scale = 0.0
        feats = []
        with torch.no_grad():
            for s in range(0, n, 512):
                b = torch.arange(s, min(s + 512, n), device=dev)
                feats.append(prep.features(prep.prepare(*data.batch(b), False)).cpu())
        f = pd.DataFrame(torch.cat(feats).numpy(), columns=prep.feature_names())
        kin[sgn] = f
        for i, j in pairs:
            a, b_ = f[f"p{i}{j}_kin_a"].to_numpy(), f[f"p{i}{j}_kin_b"].to_numpy()
            kappa = np.arctan2(b_, a)
            amp = np.hypot(a, b_)
            line = (f"   sign {sgn:+.0f} pair {i}-{j}: axis agreement with ψ "
                    f"{axis_agreement(kappa[incl], psi[incl]):+.3f}")
            if vsini is not None:
                line += f"   corr(|w|, v sin i) {rs(amp, vsini):+.3f}"
            print(line)
    sc = {s: np.mean([axis_agreement(np.arctan2(kin[s][f'p{i}{j}_kin_b'],
                                                kin[s][f'p{i}{j}_kin_a'])[incl],
                                     psi[incl]) for i, j in pairs])
          for s in kin}
    sbest = max(sc, key=sc.get)
    print(f"   -> roll_sign {sbest:+.0f} gives the better kinematic alignment "
          f"({sc[sbest]:+.3f} vs {sc[-sbest]:+.3f}); use --roll_sign "
          f"{int(sbest)}")
    if sc[sbest] < 0.3:
        verdict.append("The grism velocity fit barely finds the kinematic axis "
                       "even without noise: kinematic information is very weak "
                       "at this resolution, or the grism geometry (roll angle "
                       "convention, dispersion axis) differs from what the "
                       "script assumes.")

    # ── 5. KL observable ─────────────────────────────────────────────────────
    print("\n5. KL OBSERVABLE  misalignment Δ = kinematic axis − photometric axis "
          "vs g× (no truth used)")
    fb = kin[sbest]
    best_r = 0.0
    for i, j in pairs:
        kappa = np.arctan2(fb[f"p{i}{j}_kin_b"], fb[f"p{i}{j}_kin_a"]).to_numpy()
        delta = wrap_axis(kappa - phi)
        r_all, r_inc = rs(delta, gx), rs(delta[incl], gx[incl])
        best_r = max(best_r, abs(r_all), abs(r_inc))
        print(f"   pair {i}-{j}: corr(Δ, g×) all {r_all:+.3f}   inclined {r_inc:+.3f}"
              f"   (±{1 / np.sqrt(max(incl.sum(), 2)):.3f})")

    # ── 6. Sizes ─────────────────────────────────────────────────────────────
    print("\n6. SIZES")
    size_px = np.sqrt(np.clip(Tsz, 0, None) / 2)
    print(f"   measured size sqrt(T/2) of PSF-convolved coadd: median "
          f"{np.median(size_px):.2f} px, 10–90% {np.percentile(size_px, 10):.2f}–"
          f"{np.percentile(size_px, 90):.2f} px  (weight σ={args.weight_sigma} px)")
    for c in [c for c in df.columns if "rscale" in c.lower()][:6]:
        v = df[c].to_numpy()
        print(f"   {c}: median {np.median(v):.3f} arcsec = {np.median(v) / 0.11:.2f} px")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].scatter(np.mod(psi, np.pi), np.mod(phi, np.pi), s=3, alpha=0.4,
                  c=cosi, cmap="viridis")
    ax[0].set_xlabel(f"ψ from theta_int ({best}) mod π")
    ax[0].set_ylabel("measured photometric PA mod π")
    ax[0].set_title("2. orientation (colour = cos i)")
    ax[1].scatter(gx, e_cross, s=3, alpha=0.4)
    ax[1].set_xlabel("g× true"); ax[1].set_ylabel("e× in true galaxy frame")
    ax[1].set_title(f"3. oracle: r = {r_x:+.3f}")
    kb = np.arctan2(fb["p03_kin_b"], fb["p03_kin_a"]).to_numpy()
    ax[2].scatter(np.mod(psi[incl], np.pi), np.mod(kb[incl], np.pi), s=3, alpha=0.4)
    ax[2].set_xlabel("ψ mod π"); ax[2].set_ylabel("kinematic axis (pair 0-3) mod π")
    ax[2].set_title(f"4. kinematic axis, roll_sign {sbest:+.0f}, inclined")
    d03 = wrap_axis(kb - phi)
    ax[3].scatter(gx[incl], d03[incl], s=3, alpha=0.4)
    ax[3].set_xlabel("g× true"); ax[3].set_ylabel("Δ (kin − phot), rad")
    ax[3].set_title("5. KL observable, inclined")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved {out / 'diagnostics.png'}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\nSUMMARY")
    if best_r < 0.15 and abs(r_x) >= 0.5:
        verdict.append("Shear IS in the images (oracle works), but the observable "
                       "KL signal (kinematic/photometric misalignment) is weak "
                       "even noiseless: the per-galaxy information is limited by "
                       "resolution, not by a bug. A flat pred-vs-true is the "
                       "honest answer at this prior width; focus on ensemble "
                       "response, larger-shear training data, or better-"
                       "resolved galaxies.")
    if not verdict:
        verdict.append("Labels, orientation and kinematics all check out and the "
                       "KL observable carries signal: the training setup is "
                       "what needs work.")
    for v in verdict:
        print(" • " + v)


if __name__ == "__main__":
    main()
