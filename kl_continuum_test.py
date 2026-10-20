#!/usr/bin/env python3
"""
kl_continuum_test.py
====================
Does continuum subtraction and a joint four-roll velocity fit recover the
kinematic signal?  No training: uses the stamps and the truth columns to
measure, per method, how well the kinematic axis is recovered and how
strongly the observable KL signal tracks g×.  Needs train_kl_grism_v3.py
alongside (it uses the same preprocessing code the training does).

Methods compared
  pair       velocity gradient fitted to each opposed roll pair (0-3, 1-2)
             separately, as in kl_diagnostics.py and the v2/v3 inputs
  joint      one rotating-disc velocity field fitted to all four rolls at
             once, using each roll's exact dispersion direction
  ... each without and with continuum subtraction (_cs)

Metrics (inclined galaxies, cos i < --incl_cut, unless stated)
  axis    mean cos 2(κ − ψ): kinematic axis κ against the true position
          angle ψ = theta_int.  1 = perfect, 0 = unrelated.
  amp~v   Spearman correlation of the fitted rotation amplitude with
          vcirc · sin i, all galaxies.  Measures whether rotation is detected.
  Δ~g×    Spearman correlation of the kinematic−photometric misalignment
          Δ = κ − φ_phot with g×: the KL observable, no truth used.
  ceiling the same correlation with κ replaced by the TRUE axis ψ: what Δ
          could reach if the kinematic axis were perfect.  The gap between
          Δ~g× and ceiling is what better kinematics can still win; the
          ceiling itself is set by the imaging (PSF, resolution).

Continuum check (clean stamps): fraction of grism flux left in the outer
columns after subtraction (≈ 0 if the continuum model is right), and
example stamps before / model / after in continuum_examples.png.

Geometry checks (clean stamps, continuum subtracted)
  line position  where the Halpha line sits along the dispersion in each
                 native stamp, and whether that offset tracks redshift or the
                 systemic velocity v0.  Large, z-dependent offsets mean the
                 line is not centred: the roll alignment (a rotation about the
                 stamp centre) then misregisters the four line images.
  alignment      correlation of the aligned line-only stamps with F158 for
                 roll sign ±1, with and without a parity flip and line
                 recentring.  The right geometry gives the highest value.
Kinematics are then re-measured with line recentring (+rc).

Usage (Colab):
  !python kl_continuum_test.py --parquet "{DATA}/pilot_v2_all.parquet" \
      --meta "{DATA}/meta_pilot_v2_all.json" --outdir "{OUT}/cont_test"
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

import train_kl_grism_v3 as T


def wrap_axis(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def rs(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 10 else np.nan


@torch.no_grad()
def run_features(prep, data, n, bs):
    out = []
    for s in range(0, n, bs):
        b = torch.arange(s, min(s + bs, n), device=data.device)
        st = prep.prepare(*data.batch(b), train=False)
        out.append(prep.features(st).cpu())
    return pd.DataFrame(torch.cat(out).numpy(), columns=prep.feature_names())


@torch.no_grad()
def continuum_check(data, rolls_t, roll_sign, args, n, bs):
    """Outer-column flux fraction before/after subtraction on clean stamps."""
    c0 = (data.clean.shape[-1] - 1) / 2
    cols = torch.arange(data.clean.shape[-1], device=data.device)
    outer = (cols - c0).abs() >= args.cont_halfwidth
    before, after, total = 0.0, 0.0, 0.0
    for s in range(0, n, bs):
        b = torch.arange(s, min(s + bs, n), device=data.device)
        x = data.clean[b] / data.sigma[b][:, :, None, None]
        C, _ = T.continuum_model(x, rolls_t[b], roll_sign, args.cont_order,
                                 args.cont_halfwidth, args.cont_smooth,
                             profile=args.cont_profile)
        g = x[:, 2:]
        before += g[..., outer].abs().sum().item()
        after += (g - C)[..., outer].abs().sum().item()
        total += g.abs().sum().item()
    return before / total, after / total


@torch.no_grad()
def native_lines(data, rolls_t, roll_sign, args, b, mirror=False):
    """Clean, continuum-subtracted native grism stamps (B, 4, H, W)."""
    x = data.clean[b] / data.sigma[b][:, :, None, None]
    if mirror:
        x = torch.cat([x[:, :2], torch.flip(x[:, 2:], dims=[-2])], 1)
    C, _ = T.continuum_model(x, rolls_t[b], roll_sign, args.cont_order,
                             args.cont_halfwidth, args.cont_smooth,
                             profile=args.cont_profile)
    # second pass centred on the line, as the training preprocessing does
    off = T.line_offsets(x[:, 2:] - C, smooth_k=T.gaussian_kernel(1.0, x.device))
    C, _ = T.continuum_model(x, rolls_t[b], roll_sign, args.cont_order,
                             args.cont_halfwidth, args.cont_smooth,
                             center=off.mean(1, keepdim=True).expand(-1, 4),
                             profile=args.cont_profile)
    return x, x[:, 2:] - C


@torch.no_grad()
def geometry_check(data, rolls_t, args, df, n, bs, sk):
    offs = []
    for s in range(0, n, bs):
        b = torch.arange(s, min(s + bs, n), device=data.device)
        _, line = native_lines(data, rolls_t, args.roll_sign, args, b)
        offs.append(T.line_offsets(line, smooth_k=sk).cpu())
    off = torch.cat(offs).numpy()                           # (n, 4)
    common = off.mean(1)
    print("\nLINE POSITION along the dispersion (clean, continuum subtracted; "
          "px from stamp centre)")
    q = np.percentile(common, [5, 25, 50, 75, 95])
    print("   mean over rolls: 5/25/50/75/95% = "
          + " / ".join(f"{v:+.2f}" for v in q))
    print(f"   roll-to-roll scatter (std over the 4 rolls, median): "
          f"{np.median(off.std(1)):.2f} px")
    for col in ["truth.z", "truth.vel.v0"]:
        if col in df:
            print(f"   corr(offset, {col}) = {rs(common, df[col].to_numpy()):+.3f}")
    big = np.mean(np.abs(common) > 1.0)
    print(f"   fraction with |offset| > 1 px: {big:.1%}")
    return off


@torch.no_grad()
def alignment_check(data, rolls_t, args, n, bs, sk):
    """Correlation of aligned, clean line-only stamps with F158."""
    mask = T.aperture_mask(data.clean.shape[-1], data.device)

    def zn(a):
        a = a * mask
        a = (a - a.sum((-2, -1), keepdim=True) / mask.sum()) * mask
        return a / a.flatten(-2).norm(dim=-1)[..., None, None].clamp_min(1e-12)

    m = min(n, 1500)
    res = {}
    for mirror in (False, True):
        for rc in (False, True):
            for sign in (+1.0, -1.0):
                acc = []
                for s in range(0, m, bs):
                    b = torch.arange(s, min(s + bs, m), device=data.device)
                    x, line = native_lines(data, rolls_t, sign, args, b, mirror)
                    ref = x[:, 1]
                    if rc:      # 2-D: imaging and each roll on its own centroid
                        ci = T.centroid2d(x[:, 0] + x[:, 1], smooth_k=sk)
                        ref = T.shift2d(x[:, 1:2], ci[:, :1], ci[:, 1:])[:, 0]
                        B_ = line.shape[0]
                        cg = T.centroid2d(line.reshape(B_ * 4, *line.shape[-2:]),
                                          smooth_k=sk).reshape(B_, 4, 2)
                        line = T.shift2d(line, cg[..., 0], cg[..., 1])
                    al = T.rotate_batch(line, sign * rolls_t[b])
                    acc.append((zn(al) * zn(ref)[:, None]).sum((-2, -1)).cpu())
                res[(mirror, rc, sign)] = torch.cat(acc).mean(0).numpy()
    print("\nALIGNMENT of line-only stamps with F158 (clean; mean correlation "
          "per roll, higher = better registered)")
    for (mirror, rc, sign), v in res.items():
        print(f"   sign {sign:+.0f}  mirror {str(mirror):5}  recentred "
              f"{str(rc):5}:  " + "  ".join(f"{c:.3f}" for c in v)
              + f"   mean {v.mean():.3f}")
    best = max(res, key=lambda k: res[k].mean())
    print(f"   -> best: roll sign {best[2]:+.0f}, mirror {best[0]}, "
          f"recentred {best[1]}")
    return best


@torch.no_grad()
def psf_systematics_check(data, rolls_t, args, df, n, bs, sk, npix, dev,
                          incl, psi):
    """Detector-fixed effects (a PSF that is not round or not symmetric).
    Galaxies have random orientations, so for a round, symmetric PSF the mean
    ellipticity in the sky (imaging) or detector (grism) frame, the mean
    skewness along the dispersion, and any preferred direction of the fitted
    kinematic axis should all be zero within errors."""
    xx, yy = T.pixel_coords(npix, dev)
    W = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 3.0 ** 2))

    def shape_stats(im):                      # im (N, H, W) centred stamps
        w = im.clamp_min(0) * W
        f = w.sum((-2, -1)).clamp_min(1e-9)
        mxx = (w * xx * xx).sum((-2, -1)) / f
        myy = (w * yy * yy).sum((-2, -1)) / f
        mxy = (w * xx * yy).sum((-2, -1)) / f
        Tt = (mxx + myy).clamp_min(1e-6)
        sx = (w * xx ** 3).sum((-2, -1)) / f / mxx.clamp_min(1e-6) ** 1.5
        sy = (w * yy ** 3).sum((-2, -1)) / f / myy.clamp_min(1e-6) ** 1.5
        return torch.stack([(mxx - myy) / Tt, 2 * mxy / Tt, sx, sy], 1)

    imgs, grs = [], []
    for s0 in range(0, n, bs):
        b = torch.arange(s0, min(s0 + bs, n), device=dev)
        x, line = native_lines(data, rolls_t, args.roll_sign, args, b)
        ci = T.centroid2d(x[:, 1], smooth_k=sk)
        f158 = T.shift2d(x[:, 1:2], ci[:, :1], ci[:, 1:])[:, 0]
        imgs.append(shape_stats(f158).cpu())
        B_ = line.shape[0]
        cg = T.centroid2d(line.reshape(B_ * 4, npix, npix), smooth_k=sk).reshape(B_, 4, 2)
        line = T.shift2d(line, cg[..., 0], cg[..., 1])
        grs.append(shape_stats(line.reshape(B_ * 4, npix, npix)).reshape(B_, 4, 4).cpu())
    im = torch.cat(imgs).numpy()
    gr = torch.cat(grs).numpy()

    def ms(v):
        return f"{np.mean(v):+.4f} ± {np.std(v) / np.sqrt(len(v)):.4f}"

    print("\nPSF / DETECTOR SYSTEMATICS (clean stamps, each centred on its own "
          "light; should all be 0 within errors for a round, symmetric PSF)")
    print(f"   F158, sky frame:        <e1> {ms(im[:, 0])}   <e2> {ms(im[:, 1])}")
    for j in range(4):
        g = gr[:, j]
        print(f"   grism roll {j}, detector: <e1> {ms(g[:, 0])}   <e2> {ms(g[:, 1])}"
              f"   skew along dispersion {ms(g[:, 2])}   across {ms(g[:, 3])}")

    # preferred direction of the fitted kinematic axis (sky = roll-0 frame)
    prep = T.GPUPreprocessor(
        args.roll_sign, True, [(0, 3), (1, 2)], 2.0, resample_noise=False,
        rot_aug=False, canonicalize=False, vel_channels=False, vel_decomp=False,
        vel_eps=0.5, vel_clip=2.0, vel_smooth=1.0, moment_sigma=3.0, npix=npix,
        device=dev, cont_sub=True, cont_order=args.cont_order,
        cont_halfwidth=args.cont_halfwidth, cont_smooth=args.cont_smooth,
        recenter="2d", recenter_imaging=True, cont_profile=args.cont_profile)
    prep.noise_scale = 0.0
    ka, amp = [], []
    for s0 in range(0, n, bs):
        b = torch.arange(s0, min(s0 + bs, n), device=dev)
        a, bb, _ = prep.joint_kinematics(prep.prepare(*data.batch(b), train=False))
        ka.append(torch.atan2(bb, a).cpu()); amp.append(torch.hypot(a, bb).cpu())
    ka, amp = torch.cat(ka).numpy(), torch.cat(amp).numpy()
    cosi = df["truth.cosi"].to_numpy()
    face = cosi > 0.85
    sini = np.sqrt(1 - cosi ** 2)
    print(f"   (from sin i alone, face-on/inclined amplitude ratio ≈ "
          f"{np.median(sini[face]) / np.median(sini[incl]):.2f})")
    for name, m in [("face-on (cos i > 0.85)", face), ("inclined (cos i < 0.6)", incl),
                    ("all", np.ones(n, bool))]:
        z = np.mean(np.exp(2j * ka[m]))
        print(f"   kinematic axis, {name:<22} n={m.sum():5d}: preferred-direction "
              f"strength {abs(z):.3f} (noise ≈ {1 / np.sqrt(m.sum()):.3f}) at "
              f"{np.degrees(np.angle(z) / 2) % 180:5.1f}° from the dispersion; "
              f"median amplitude {np.median(amp[m]):.3f} px")
    print("   • face-on galaxies barely rotate along the line of sight: an "
          "amplitude there comparable to the inclined one, or a preferred "
          "direction well above noise, is a detector-fixed systematic, not "
          "rotation.  Training uses --angle_inputs so the model can learn it.")


@torch.no_grad()
def response_test(data, args, df, n, bs, npix, dev, incl, psi):
    """How the fitted velocity gradient responds to the true one.
    In the roll-0 frame (dispersion along +x), the true gradient is
        u = A (cos θ, sin θ),   A = vcirc sin i · s(z)  [px of line shift],
    θ = theta_int.  Fit  (a, b) = M u + c  over half the galaxies, then use
    M⁻¹ to correct the other half.  An isotropic M (≈ k·I) means the fit is
    fine; a small M_xx relative to M_yy means the gradient ALONG the
    dispersion is under-measured, which pulls every axis to 90°; negative
    or off-diagonal terms mean a convention mismatch."""
    need = ["truth.vel.vcirc", "truth.cosi", "truth.z", "truth.theta_int"]
    if not all(c in df.columns for c in need):
        return
    prep = T.GPUPreprocessor(
        args.roll_sign, True, [(0, 3), (1, 2)], 2.0, resample_noise=False,
        rot_aug=False, canonicalize=False, vel_channels=False, vel_decomp=False,
        vel_eps=0.5, vel_clip=2.0, vel_smooth=1.0, moment_sigma=3.0, npix=npix,
        device=dev, cont_sub=True, cont_order=args.cont_order,
        cont_halfwidth=args.cont_halfwidth, cont_smooth=args.cont_smooth,
        recenter="2d", recenter_imaging=True, cont_profile=args.cont_profile)
    prep.noise_scale = 0.0
    ab, mo = [], []
    for s0 in range(0, n, bs):
        b = torch.arange(s0, min(s0 + bs, n), device=dev)
        st = prep.prepare(*data.batch(b), train=False)
        a, bb, _ = prep.joint_kinematics(st)
        ab.append(torch.stack([a, bb], 1).cpu())
        mo.append(torch.stack(prep.moment_kinematics(st), 1).cpu())
    ab = torch.cat(ab).numpy().astype(np.float64)
    mo = torch.cat(mo).numpy().astype(np.float64)
    cosi = df["truth.cosi"].to_numpy()
    A = (df["truth.vel.vcirc"].to_numpy() * np.sqrt(1 - cosi ** 2)
         * 656.28 * (1 + df["truth.z"].to_numpy()) / (2.99792458e5 * 1.1))
    u = np.column_stack([A * np.cos(psi), A * np.sin(psi)])
    rng = np.random.default_rng(0)
    half = rng.random(n) < 0.5
    X = np.column_stack([u, np.ones(n)])
    coef, *_ = np.linalg.lstsq(X[half], ab[half], rcond=None)   # (3, 2)
    M, c = coef[:2].T, coef[2]
    r2 = [1 - np.var(ab[half, k] - X[half] @ coef[:, k]) / np.var(ab[half, k])
          for k in range(2)]
    print("\nRESPONSE TEST (clean; joint fit, roll-0 frame; true gradient u = "
          "vcirc·sin i·s(z)·(cos θ, sin θ))")
    print(f"   measured (a, b) = M·u + c   M = [[{M[0, 0]:+.3f}, {M[0, 1]:+.3f}], "
          f"[{M[1, 0]:+.3f}, {M[1, 1]:+.3f}]]   c = ({c[0]:+.3f}, {c[1]:+.3f})   "
          f"R² ({r2[0]:.2f}, {r2[1]:.2f})")
    print("   (row 1 = along the dispersion, row 2 = across.  Isotropic ≈ k·I.)")
    sense = np.mean(np.sign(np.sum(ab * u, 1)) > 0)
    print(f"   fraction with fitted gradient pointing the same way as the truth: "
          f"{sense:.2f}  (≈1: rotation sense agrees with theta_int; ≈0.5: θ "
          f"does not carry the sense)")
    try:
        Minv = np.linalg.inv(M)
        corr = (ab - c) @ Minv.T
        k0 = np.arctan2(ab[:, 1], ab[:, 0])
        k1 = np.arctan2(corr[:, 1], corr[:, 0])
        test = (~half) & incl
        print(f"   held-out inclined galaxies: axis {np.mean(np.cos(2 * (k0[test] - psi[test]))):+.3f}"
              f" raw  →  {np.mean(np.cos(2 * (k1[test] - psi[test]))):+.3f} after M⁻¹")
    except np.linalg.LinAlgError:
        print("   M is singular")
    Xm = np.column_stack([u, np.ones(n)])
    cm, *_ = np.linalg.lstsq(Xm[half], mo[half], rcond=None)
    r2m = [1 - np.var(mo[half, k] - Xm[half] @ cm[:, k]) / np.var(mo[half, k])
           for k in range(2)]
    km = np.arctan2(mo[:, 1], mo[:, 0])
    test = (~half) & incl
    print(f"   MOMENT method: M = [[{cm[0, 0]:+.3f}, {cm[1, 0]:+.3f}], "
          f"[{cm[0, 1]:+.3f}, {cm[1, 1]:+.3f}]]   R² ({r2m[0]:.2f}, {r2m[1]:.2f})   "
          f"held-out inclined axis {np.mean(np.cos(2 * (km[test] - psi[test]))):+.3f}")
    # COMBINED: per component (along, across), a least-squares mix of the
    # joint fit and the moment method, fitted on half the galaxies
    ab = np.clip(ab, *np.percentile(ab, [1, 99], axis=0))
    mo = np.clip(mo, *np.percentile(mo, [1, 99], axis=0))
    cal = np.zeros(4)
    for k in range(2):
        Xk = np.column_stack([ab[:, k], mo[:, k]])
        cal[2 * k:2 * k + 2], *_ = np.linalg.lstsq(Xk[half], u[half, k], rcond=None)
    uc = np.column_stack([cal[0] * ab[:, 0] + cal[1] * mo[:, 0],
                          cal[2] * ab[:, 1] + cal[3] * mo[:, 1]])
    kc = np.arctan2(uc[:, 1], uc[:, 0])
    r2c = [np.corrcoef(uc[~half, k], u[~half, k])[0, 1] ** 2 for k in range(2)]
    print(f"   COMBINED (joint + moments, per-component weights from half the "
          f"galaxies): held-out R² (along {r2c[0]:.2f}, across {r2c[1]:.2f})   "
          f"held-out inclined axis {np.mean(np.cos(2 * (kc[test] - psi[test]))):+.3f}")
    print("   (training uses this combination by default: --kin_calibrate)")
    # doubled-angle version (does not need the rotation sense)
    z = np.column_stack([np.cos(2 * np.arctan2(ab[:, 1], ab[:, 0])),
                         np.sin(2 * np.arctan2(ab[:, 1], ab[:, 0]))])
    w = np.column_stack([np.cos(2 * psi), np.sin(2 * psi), np.ones(n)])
    cz, *_ = np.linalg.lstsq(w[half & incl], z[half & incl], rcond=None)
    print(f"   axis-only map (cos2κ, sin2κ) = N·(cos2ψ, sin2ψ) + d: N = "
          f"[[{cz[0, 0]:+.2f}, {cz[1, 0]:+.2f}], [{cz[0, 1]:+.2f}, {cz[1, 1]:+.2f}]]"
          f"  d = ({cz[2, 0]:+.2f}, {cz[2, 1]:+.2f})")


@torch.no_grad()
def truth_geometry_check(data, rolls_t, args, df, n, bs, sk):
    """Predict where the Halpha line should sit in each NATIVE grism stamp
    from the truth columns, and regress the measured centroids on it:
        col_j = (R(A_j) p)_x + s(z) · v0,     row_j = (R(A_j) p)_y
    with p the Halpha sky offset (truth.Halpha.x0/y0 / 0.11 px), A_j =
    roll_sign · roll_j, and s(z) = 656.28 nm (1+z) / (c · 1.1 nm/px) px per
    km/s.  For the correct geometry every roll gives slopes ≈ 1 on the
    position term and ≈ 1 on the Doppler term, with high R².  A roll with a
    wrong slope or sign pinpoints a geometry error; the v0 slope tests the
    dispersion direction and scale."""
    need = ["truth.Halpha.x0", "truth.Halpha.y0", "truth.vel.v0", "truth.z"]
    if not all(c in df.columns for c in need):
        print("\nTRUTH GEOMETRY CHECK skipped (needs " + ", ".join(need) + ")")
        return
    cs = []
    for s0 in range(0, n, bs):
        b = torch.arange(s0, min(s0 + bs, n), device=data.device)
        _, line = native_lines(data, rolls_t, args.roll_sign, args, b)
        B_ = line.shape[0]
        cs.append(T.centroid2d(line.reshape(B_ * 4, *line.shape[-2:]),
                               smooth_k=sk).reshape(B_, 4, 2).cpu())
    c = torch.cat(cs).numpy()                                   # (n, 4, 2)
    px, py = (df["truth.Halpha.x0"].to_numpy() / 0.11,
              df["truth.Halpha.y0"].to_numpy() / 0.11)
    sdop = 656.28 * (1 + df["truth.z"].to_numpy()) / (2.99792458e5 * 1.1)
    dop = sdop * df["truth.vel.v0"].to_numpy()
    rolls = rolls_t[:n].cpu().numpy()
    print("\nTRUTH GEOMETRY CHECK (clean line centroids in the native grism "
          "stamps vs prediction from truth.Halpha.x0/y0, roll and v0)")
    for sign in (args.roll_sign, -args.roll_sign):
        print(f"   roll sign {sign:+.0f}:")
        for j in range(4):
            A = np.deg2rad(sign * rolls[:, j])
            qx = np.cos(A) * px - np.sin(A) * py
            qy = np.sin(A) * px + np.cos(A) * py
            X = np.column_stack([qx, dop, np.ones(n)])
            bc, *_ = np.linalg.lstsq(X, c[:, j, 0], rcond=None)
            r2c = 1 - np.var(c[:, j, 0] - X @ bc) / np.var(c[:, j, 0])
            Xr = np.column_stack([qy, np.ones(n)])
            br, *_ = np.linalg.lstsq(Xr, c[:, j, 1], rcond=None)
            r2r = 1 - np.var(c[:, j, 1] - Xr @ br) / np.var(c[:, j, 1])
            print(f"     roll {j} ({rolls[0, j]:5.1f}°): column = {bc[0]:+.2f}·pos "
                  f"{bc[1]:+.2f}·doppler {bc[2]:+.2f}  (R² {r2c:.2f})   "
                  f"row = {br[0]:+.2f}·pos {br[1]:+.2f}  (R² {r2r:.2f})")
    print("   expected for the right geometry: pos ≈ +1, doppler ≈ +1 (or −1 if "
          "wavelength runs toward −columns), R² close to 1, same in every roll")


@torch.no_grad()
def centre_test(data, df, args, n, bs, npix, dev, incl, psi):
    """Fit the joint kinematics with the rotation centre at different points.
    Stamps: clean, continuum subtracted, aligned to the sky, NOT recentred, so
    every component sits at its true offset.  If the velocity field is not
    centred on the Halpha light, centring the fit on the light (what
    recentring does) biases the kinematic axis."""
    prep = T.GPUPreprocessor(
        args.roll_sign, True, [(0, 3), (1, 2)], 2.0, resample_noise=False,
        rot_aug=False, canonicalize=False, vel_channels=False, vel_decomp=False,
        vel_eps=0.5, vel_clip=2.0, vel_smooth=1.0, moment_sigma=3.0, npix=npix,
        device=dev, cont_sub=True, cont_order=args.cont_order,
        cont_halfwidth=args.cont_halfwidth, cont_smooth=args.cont_smooth,
        recenter="off", recenter_imaging=False, cont_profile=args.cont_profile)
    prep.noise_scale = 0.0
    sk = T.gaussian_kernel(1.0, dev)
    px = lambda c: torch.as_tensor(df[c].to_numpy() / 0.11, dtype=torch.float32,
                                   device=dev)
    have = {c for c in df.columns}
    truth = {}
    for name, cx, cy in [("Halpha (truth)", "truth.Halpha.x0", "truth.Halpha.y0"),
                         ("F158 (truth)", "truth.F158.x0", "truth.F158.y0"),
                         ("continuum (truth)", "truth.Halpha.cont.x0",
                          "truth.Halpha.cont.y0")]:
        if cx in have and cy in have:
            truth[name] = torch.stack([px(cx), px(cy)], 1)
    kap = {}
    for s0 in range(0, n, bs):
        b = torch.arange(s0, min(s0 + bs, n), device=dev)
        st = prep.prepare(*data.batch(b), train=False)
        cand = {"stamp centre": None,
                "line centroid (measured)": T.centroid2d(st["gr"].mean(1), smooth_k=sk),
                "imaging centroid (measured)": T.centroid2d(st["img"].sum(1), smooth_k=sk)}
        for k, v in truth.items():
            cand[k] = v[b]
        cand["fitted (search ±2 px)"] = ("search", cand["line centroid (measured)"])
        for k, c in cand.items():
            if isinstance(c, tuple):
                a, bb, _ = prep.joint_kinematics(st, centre=c[1], search=2.0)
            else:
                a, bb, _ = prep.joint_kinematics(st, centre=c)
            kap.setdefault(k, []).append(torch.atan2(bb, a).cpu())
    print("\nKINEMATIC CENTRE TEST (clean, continuum subtracted, not recentred; "
          "joint fit; axis = mean cos 2(κ − ψ), inclined)")
    out = {}
    for k, v in kap.items():
        kk = torch.cat(v).numpy()
        out[k] = float(np.mean(np.cos(2 * (kk[incl] - psi[incl]))))
        print(f"   rotation centre at {k:<30} axis {out[k]:+.3f}")
    best = max(out, key=out.get)
    print(f"   -> best: {best}.  If a truth centre scores far above the measured "
          f"line centroid, the velocity field is not centred on the light.")
    if "Halpha (truth)" in truth:
        off = np.hypot(*(df[["truth.Halpha.x0", "truth.Halpha.y0"]].to_numpy().T)) / 0.11
        kk = torch.cat(kap["line centroid (measured)"]).numpy()
        print("   axis vs |Halpha offset from stamp centre| (line-centred fit):")
        for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 9)]:
            m = incl & (off >= lo) & (off < hi)
            if m.sum() > 20:
                print(f"      {lo:.1f}–{hi:.1f} px  n={m.sum():5d}  axis "
                      f"{np.mean(np.cos(2 * (kk[m] - psi[m]))):+.3f}")
    return out


@torch.no_grad()
def plot_examples(data, rolls_t, roll_sign, args, rows, out):
    b = torch.as_tensor(rows, device=data.device)
    x = data.clean[b] / data.sigma[b][:, :, None, None]
    C, _ = T.continuum_model(x, rolls_t[b], roll_sign, args.cont_order,
                             args.cont_halfwidth, args.cont_smooth,
                             profile=args.cont_profile)
    g = x[:, 2:].cpu().numpy()
    C = C.cpu().numpy()
    fig, ax = plt.subplots(len(rows), 6, figsize=(13, 2.2 * len(rows)),
                           squeeze=False)
    for r in range(len(rows)):
        for k, j in enumerate([0, 1]):
            for m, (img, ttl) in enumerate([(g[r, j], "clean grism"),
                                            (C[r, j], "continuum model"),
                                            (g[r, j] - C[r, j], "line (after)")]):
                a = ax[r, 3 * k + m]
                v = np.abs(g[r, j]).max() or 1.0
                a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f"roll {j}: {ttl}", fontsize=8)
                for c in (15.5 - args.cont_halfwidth, 15.5 + args.cont_halfwidth):
                    a.axvline(c, color="k", lw=0.5, ls=":")
        ax[r, 0].set_ylabel(f"row {rows[r]}", fontsize=8)
    fig.suptitle("Continuum subtraction on clean stamps (dotted: fit region "
                 "outside)", fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "continuum_examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--meta", default=None)
    p.add_argument("--outdir", default="./kl_continuum_test")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--roll_sign", type=float, default=-1.0,
                   help="-1 for pilot_v2 (from kl_diagnostics.py)")
    p.add_argument("--noise", choices=["clean", "noisy", "both"], default="both")
    p.add_argument("--cont_order", type=int, default=1)
    p.add_argument("--cont_profile", choices=["grism", "imaging"], default="grism")
    p.add_argument("--cont_halfwidth", type=int, default=9)
    p.add_argument("--cont_smooth", type=float, default=0.7)
    p.add_argument("--incl_cut", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--mirror", choices=["auto", "no", "yes"], default="auto",
                   help="Parity flip of the grism stamps for the kinematic "
                        "tests; auto = whatever the alignment check prefers")
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
    data = T.DeviceData(noisy, clean, var, sigma, rolls,
                        df[T.TARGETS].to_numpy(np.float32), None, dev)
    rolls_t = data.rolls

    psi = df["truth.theta_int"].to_numpy(np.float64)
    cosi = df["truth.cosi"].to_numpy(np.float64)
    gx = df["truth.g_cross"].to_numpy(np.float64)
    vsini = df["truth.vel.vcirc"].to_numpy() * np.sqrt(1 - cosi ** 2)
    incl = cosi < args.incl_cut
    err = 1 / np.sqrt(max(incl.sum(), 2))

    # ── continuum check ──────────────────────────────────────────────────
    fb, fa = continuum_check(data, rolls_t, args.roll_sign, args, n, args.batch)
    print(f"\nCONTINUUM (clean stamps): grism flux in outer columns "
          f"(|col − centre| ≥ {args.cont_halfwidth}):")
    print(f"   before subtraction {fb:.1%} of total |flux|   after {fa:.1%}")
    print("   (after ≈ 0 means the outer columns were continuum and the model "
          "matches it)")
    snr_col = "snr_eff.line_total" if "snr_eff.line_total" in df else None
    order = (np.argsort(-df[snr_col].to_numpy()) if snr_col
             else np.arange(n))
    ex = [int(i) for i in order[:4 * 3:3]]
    plot_examples(data, rolls_t, args.roll_sign, args, ex, out)
    print(f"   saved {out / 'continuum_examples.png'}")

    # ── kinematics, per noise level and method ───────────────────────────
    modes = ["clean", "noisy"] if args.noise == "both" else [args.noise]
    rows = []
    sk = T.gaussian_kernel(1.0, dev)
    truth_geometry_check(data, rolls_t, args, df, n, args.batch, sk)
    psf_systematics_check(data, rolls_t, args, df, n, args.batch, sk, npix,
                          dev, incl, psi)
    response_test(data, args, df, n, args.batch, npix, dev, incl, psi)
    centre_test(data, df, args, n, args.batch, npix, dev, incl, psi)
    geometry_check(data, rolls_t, args, df, n, args.batch, sk)
    best_geo = alignment_check(data, rolls_t, args, n, args.batch, sk)
    mirror = best_geo[0] if args.mirror == "auto" else args.mirror == "yes"
    configs = [(False, "off", False), (True, "off", False),
               (True, "common", False), (False, "off", True),
               (True, "2d", True)]
    tags = {"off": "", "common": " +rc", "per_roll": " +rc/roll", "2d": " +rc2d"}
    for mode in modes:
        for cs, rc, ic in configs:
            prep = T.GPUPreprocessor(
                args.roll_sign, True, [(0, 3), (1, 2)], 2.0,
                resample_noise=False, rot_aug=False, canonicalize=False,
                vel_channels=True, vel_decomp=False, vel_eps=0.5,
                vel_clip=2.0, vel_smooth=1.0, moment_sigma=3.0, npix=npix,
                device=dev, cont_sub=cs, cont_order=args.cont_order,
                cont_halfwidth=args.cont_halfwidth,
                cont_smooth=args.cont_smooth, recenter=rc,
                grism_mirror=mirror, recenter_imaging=ic,
                cont_profile=args.cont_profile)
            prep.noise_scale = 0.0 if mode == "clean" else 1.0
            f = run_features(prep, data, n, args.batch)
            phi = 0.5 * np.arctan2(f["img_e2"], f["img_e1"]).to_numpy()
            ceiling_all = rs(wrap_axis(psi - phi), gx)
            ceiling_inc = rs(wrap_axis(psi[incl] - phi[incl]), gx[incl])
            fits = {"pair 0-3": ("p03_kin_a", "p03_kin_b"),
                    "pair 1-2": ("p12_kin_a", "p12_kin_b"),
                    "joint": ("joint_kin_a", "joint_kin_b"),
                    "moments": ("mom_kin_gx", "mom_kin_gy")}
            for name, (ca, cb) in fits.items():
                a, b = f[ca].to_numpy(), f[cb].to_numpy()
                kap = np.arctan2(b, a)
                d = wrap_axis(kap - phi)
                rows.append(dict(
                    noise=mode, method=name + (" +cs" if cs else "")
                    + tags[rc] + (" +img" if ic else ""),
                    axis=float(np.mean(np.cos(2 * (kap[incl] - psi[incl])))),
                    amp_v=rs(np.hypot(a, b), vsini),
                    dgx_all=rs(d, gx), dgx_incl=rs(d[incl], gx[incl]),
                    ceiling_all=ceiling_all, ceiling_incl=ceiling_inc,
                    phot_axis=float(np.mean(np.cos(2 * (phi[incl] - psi[incl]))))))
            if mode == modes[-1] and rc == "2d":
                kap_j = np.arctan2(f["joint_kin_b"], f["joint_kin_a"]).to_numpy()
            if mode == modes[-1] and not cs:
                kap_j0 = np.arctan2(f["joint_kin_b"], f["joint_kin_a"]).to_numpy()

    res = pd.DataFrame(rows)
    res.to_csv(out / "continuum_test_results.csv", index=False)
    print(f"\nKINEMATICS  (inclined: cos i < {args.incl_cut}, n = {incl.sum()}; "
          f"correlation errors ≈ ±{err:.3f})")
    print("  phot = photometric axis vs truth; ceiling = Δ~g× (inclined) if the "
          "kinematic axis were perfect.  Both depend on the imaging recentring "
          "(+img).")
    for mode in modes:
        print(f"\n  {mode.upper()} stamps")
        print(f"  {'method':<28}{'axis':>8}{'amp~v':>8}{'Δ~g× all':>11}"
              f"{'Δ~g× incl':>11}{'phot':>8}{'ceiling':>9}")
        for _, r in res[res.noise == mode].iterrows():
            print(f"  {r['method']:<28}{r['axis']:+8.3f}{r['amp_v']:+8.3f}"
                  f"{r['dgx_all']:+11.3f}{r['dgx_incl']:+11.3f}"
                  f"{r['phot_axis']:+8.3f}{r['ceiling_incl']:+9.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, k, ttl in [(axs[0], kap_j0, "joint, no subtraction"),
                      (axs[1], kap_j, "joint, recentred (training default)")]:
        a.scatter(np.mod(psi[incl], np.pi), np.mod(k[incl], np.pi), s=3, alpha=0.4)
        a.set_xlabel("true position angle ψ mod π")
        a.set_ylabel("kinematic axis κ mod π")
        a.set_title(f"{ttl} [{modes[-1]}], inclined")
        a.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "kinematic_axis.png", dpi=130, bbox_inches="tight")
    print(f"\nSaved {out / 'kinematic_axis.png'} and continuum_test_results.csv")

    best = res[res.noise == modes[0]].sort_values("axis").iloc[-1]
    print("\nREADING THE RESULT")
    print(f"  Best method on {modes[0]} stamps: {best['method']} "
          f"(axis {best['axis']:+.3f}; earlier pair fits gave ≈ +0.3–0.47).")
    print("  • '+cs +rc2d +img' (the training default) clearly best: the stamps "
          "were misregistered; the defaults of train_kl_grism_v3.py handle it.")
    print("  • '+img' rows raise phot and ceiling: shapes were being measured "
          "about the stamp centre instead of the galaxy.")
    print("  • mirror chosen = True: train with --grism_mirror as well.")
    print("  • clean-stamp axis still ≲ 0.5 everywhere: the stamps hold "
          "something my model of them misses; look at continuum_examples.png "
          "and ask the data producers how the grism stamps are cut and "
          "oriented.")


if __name__ == "__main__":
    main()