#!/usr/bin/env python3
"""
train_kl_grism_v3.py
====================
Shear regression for Roman kinematic-lensing mock stamps (imaging + slitless
grism), with no pretrained backbone and a training loop built around
validation shear error.

What changed from v2
--------------------
Model         EfficientNet is gone.  --arch small (default) is a two-stream
              CNN trained from scratch whose width (--width) and dropout
              (--dropout, --dropout2d) are adjustable; --arch features keeps
              the moment + kinematic-fit gradient-boosted-tree baseline.
Loss          Plain MSE on g / 0.03 (--loss mse, default).  The quantity you
              judge is val MSE, so the training loss is the same quantity;
              Huber is still available (--loss huber).  Each shear component
              can be weighted (--component_weights).
Shear only    No auxiliary targets: the network is trained on g+ and g×
              and nothing else, so no capacity or gradient goes to easier
              truths (cos i, vcirc) at the expense of shear.
EMA weights   An exponential moving average of the weights (--ema 0.995) is
              what gets evaluated and saved.  Averaging over many steps damps
              the epoch-to-epoch jitter and the late drift into memorisation
              that made val g+ RMSE climb.
Selection     Checkpoints are chosen on validation shear error only, never on
              total loss, and SEPARATELY for g+ and g× (--per_component_ckpt):
              in the v2 runs g+ was best after a few epochs while g× kept
              improving, so one epoch cannot be best for both.  At test time
              g+ comes from the g+-best weights and g× from the g×-best ones.
              Early stopping waits until neither component has improved for
              --patience epochs.
Calibration   A per-component linear map  g = a * pred + b  is fitted on the
              validation set (--calibrate shrink) and applied to test.  With
              a ≥ 0 fitted by least squares, the validation MSE after
              calibration can never exceed that of predicting the mean, so a
              noisy predictor is shrunk toward 0 instead of adding variance.
              --select calibrated (default) picks checkpoints by this
              calibrated val MSE, which is equivalent to picking the epoch
              with the highest val correlation; --select raw uses raw val MSE.
              Test metrics are printed both raw and calibrated.

Why not train on the validation loss directly
---------------------------------------------
Any gradient step taken on validation data turns it into training data, and
then it no longer measures generalisation (the val curve would just follow
the train curve).  What the script does instead is make every *decision*
(which epoch, which weights, how much to shrink) on validation shear MSE,
while gradients come only from the training set.

Preprocessing is unchanged from v2: noise control, roll alignment, canonical
frame, parity symmetry, velocity channels from roll pairs (see the v2 notes in
train_kl_grism.py).

Usage
-----
  python train_kl_grism_v3.py --parquet pilot_v2_all.parquet \
      --meta meta_pilot_v2_all.json --outdir run_v3 --min_line_snr 0

Dependencies: torch numpy pandas pyarrow scikit-learn matplotlib
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

IMG_CH        = ["bb_F129", "bb_F158"]
GRISM_CH      = ["gr_roll0", "gr_roll1", "gr_roll2", "gr_roll3"]
CHANNELS      = IMG_CH + GRISM_CH
TARGETS       = ["truth.g_plus", "truth.g_cross"]
DEFAULT_ROLLS = [0.0, 10.0, 167.0, 177.0]
G_SCALE       = 0.03   # width of the shear prior; networks predict g / G_SCALE


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _find_rolls_in_meta(obj):
    """Recursively look for a length-4 numeric list under a key containing 'roll'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if ("roll" in str(k).lower() and isinstance(v, (list, tuple))
                    and len(v) == 4 and all(isinstance(a, (int, float)) for a in v)):
                return [float(a) for a in v]
            found = _find_rolls_in_meta(v)
            if found is not None:
                return found
    return None


def get_roll_angles(df: pd.DataFrame, meta: dict, override) -> np.ndarray:
    """Return per-row roll angles (n, 4) in degrees."""
    n = len(df)
    if override is not None:
        rolls, src = [float(a) for a in override], "--rolls"
    else:
        cols = sorted(c for c in df.columns if "roll_deg" in c.lower())
        if len(cols) == 4:
            arr = df[cols].to_numpy(np.float32)
            print(f"Roll angles from columns {cols}; first row: {arr[0].tolist()}")
            return arr
        rolls, src = _find_rolls_in_meta(meta), "meta json"
        if rolls is None:
            rolls, src = DEFAULT_ROLLS, "README default"
    print(f"Roll angles ({src}): {rolls}")
    return np.tile(np.asarray(rolls, np.float32), (n, 1))


def load_table(parquet: str, meta_path: str):
    """Load the parquet into (n, 6, 32, 32) noisy / clean / var arrays."""
    t0 = time.time()
    df = pd.read_parquet(parquet)
    meta = {}
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
    shapes = meta.get("image_shapes", {})
    n = len(df)

    def stack(key):
        shape = tuple(shapes.get(key, (32, 32)))
        return (np.stack(df[f"img.{key}"].to_numpy())
                .astype(np.float32).reshape(n, *shape))

    noisy = np.stack([stack(c) for c in CHANNELS], axis=1)
    clean = np.stack([stack(f"{c}_clean") for c in CHANNELS], axis=1)
    var   = np.stack([stack(f"{c}_var") for c in CHANNELS], axis=1)
    df = df.drop(columns=[c for c in df.columns if c.startswith("img.")])

    sig_cols = [f"sigma_bkg.{c}" for c in CHANNELS]
    if all(c in df.columns for c in sig_cols):
        sigma = df[sig_cols].to_numpy(np.float32)
    else:
        print("[WARN] sigma_bkg.* columns not found; "
              "using sqrt(min pixel variance) per stamp")
        sigma = np.sqrt(var.min(axis=(2, 3))).astype(np.float32)

    print(f"Loaded {n} rows in {time.time() - t0:.1f} s "
          f"(images: {noisy.nbytes * 3 / 1e6:.0f} MB)")
    return df, meta, noisy, clean, var, sigma


def check_noise_model(noisy, clean, var):
    chi2 = ((noisy - clean) ** 2 / np.maximum(var, 1e-30)).mean(axis=(0, 2, 3))
    print("Noise check <(noisy - clean)^2 / var> per channel (expect ~1):")
    for c, v in zip(CHANNELS, chi2):
        print(f"  {c:>9}: {v:.3f}")


def split_by_group(groups, train_frac=0.70, val_frac=0.15, seed=42):
    """Row indices for train / val / test with no catalog galaxy in two splits."""
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_frac - val_frac,
                            random_state=seed)
    tv, te = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1,
                             test_size=val_frac / (train_frac + val_frac),
                             random_state=seed)
    tr_rel, va_rel = next(gss2.split(tv, groups=groups[tv]))
    return tv[tr_rel], tv[va_rel], te




# ═══════════════════════════════════════════════════════════════════════════════
# Image-space helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rotate_batch(x: torch.Tensor, angles_deg: torch.Tensor,
                 mode: str = "bicubic") -> torch.Tensor:
    """out(p) = in(R(angle) p) for every (sample, channel) of x (B, C, H, W),
    with p = (col, row) about the stamp centre.  A direction at angle phi in
    the input therefore lands on the +x (column) axis when angle = phi."""
    B, C, H, W = x.shape
    a = torch.deg2rad(angles_deg.reshape(-1).to(x.dtype))
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.zeros(B * C, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, (B * C, 1, H, W), align_corners=False)
    out = F.grid_sample(x.reshape(B * C, 1, H, W), grid, mode=mode,
                        padding_mode="zeros", align_corners=False)
    return out.reshape(B, C, H, W)


def aperture_mask(npix: int, device) -> torch.Tensor:
    c = torch.linspace(-1 + 1 / npix, 1 - 1 / npix, npix, device=device)
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return ((xx ** 2 + yy ** 2) <= 1.0).float()


def pixel_coords(npix: int, device):
    """(xx, yy) pixel offsets from the stamp centre; x = column, y = row."""
    c = torch.arange(npix, device=device, dtype=torch.float32) - (npix - 1) / 2
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return xx, yy


def weighted_moments(img, xx, yy, W):
    """Flux and centre-referenced second moments of img (B, H, W) under the
    weight W (H, W).  Returns flux, Qxx, Qyy, Qxy (each (B,))."""
    wi = img * W
    flux = wi.sum(dim=(-2, -1))
    f = flux.abs().clamp_min(1e-6) * torch.sign(flux + 1e-12)
    qxx = (wi * xx * xx).sum(dim=(-2, -1)) / f
    qyy = (wi * yy * yy).sum(dim=(-2, -1)) / f
    qxy = (wi * xx * yy).sum(dim=(-2, -1)) / f
    return flux, qxx, qyy, qxy


def gaussian_kernel(sigma_px: float, device) -> torch.Tensor:
    r = max(1, int(np.ceil(2.5 * sigma_px)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma_px ** 2))
    k = k / k.sum()
    return (k[:, None] * k[None, :])[None, None]          # (1, 1, K, K)


def conv_same(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Convolve (B, H, W) with a (1, 1, kh, kw) kernel, zero padded."""
    ph, pw = k.shape[-2] // 2, k.shape[-1] // 2
    return F.conv2d(img[:, None], k, padding=(ph, pw))[:, 0]


def wrap_axis_t(a: torch.Tensor) -> torch.Tensor:
    """Wrap an axis-angle difference (radians) into [-pi/2, pi/2)."""
    return torch.remainder(a + np.pi / 2, np.pi) - np.pi / 2


def solve_wls(Bm: torch.Tensor, t: torch.Tensor, w: torch.Tensor,
              ridge: float = 1e-3) -> torch.Tensor:
    """Batched weighted least squares:  argmin_s sum_p w_p (t_p - Bm_p · s)².
    Bm (N, P, K), t (N, P), w (P,) or (N, P).  A small ridge, relative to
    the mean diagonal, keeps degenerate cases finite."""
    Bm, t = Bm.double(), t.double()
    w = w.double()
    BtW = Bm.transpose(1, 2) * (w if w.dim() == 2 else w[None])[:, None, :]
    A = BtW @ Bm
    lam = ridge * A.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    A = A + lam[:, None, None] * torch.eye(A.shape[-1], device=A.device,
                                           dtype=A.dtype)
    return torch.linalg.solve(A, BtW @ t[..., None])[..., 0].float()


def line_offsets(gr: torch.Tensor, sigma: float = 3.0, iters: int = 3,
                 smooth_k: torch.Tensor = None) -> torch.Tensor:
    """Column offset (px from the stamp centre) of the line in each NATIVE
    grism stamp (B, 4, H, W): an iterated, Gaussian-windowed centroid of the
    positive part, so a noisy or partly subtracted continuum has little pull.
    Returns (B, 4)."""
    B, nr, H, W = gr.shape
    dev = gr.device
    g = gr.reshape(B * nr, H, W)
    if smooth_k is not None:
        g = conv_same(g, smooth_k)
    g = g.clamp_min(0)
    c0 = (W - 1) / 2
    cols = torch.arange(W, device=dev, dtype=g.dtype) - c0
    rows = torch.arange(H, device=dev, dtype=g.dtype) - (H - 1) / 2
    win_r = torch.exp(-rows ** 2 / (2 * sigma ** 2))[None, :, None]
    cx = torch.zeros(B * nr, device=dev, dtype=g.dtype)
    for _ in range(iters):
        win = win_r * torch.exp(-(cols[None, None] - cx[:, None, None]) ** 2
                                / (2 * sigma ** 2))
        w = g * win
        cx = (w * cols).sum((-2, -1)) / w.sum((-2, -1)).clamp_min(1e-6)
        cx = cx.clamp(-c0, c0)
    return cx.reshape(B, nr)


def shift_cols(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Shift every (B, C, H, W) image along columns so that column c of the
    output takes the input at c + dx (sub-pixel, bicubic): a feature at
    centre + dx moves to the centre.  dx is (B, C)."""
    B, C, H, W = x.shape
    theta = torch.zeros(B * C, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = dx.reshape(-1).to(x.dtype) * 2.0 / W
    grid = F.affine_grid(theta, (B * C, 1, H, W), align_corners=False)
    out = F.grid_sample(x.reshape(B * C, 1, H, W), grid, mode="bicubic",
                        padding_mode="zeros", align_corners=False)
    return out.reshape(B, C, H, W)


def centroid2d(img: torch.Tensor, sigma: float = 3.0, iters: int = 4,
               smooth_k: torch.Tensor = None) -> torch.Tensor:
    """Light centroid (dx, dy) in px from the stamp centre of each (N, H, W)
    image: an iterated, Gaussian-windowed centroid of the positive part
    (optionally smoothed first), robust to noise and faint streaks.
    Returns (N, 2) with x = column, y = row."""
    N, H, W = img.shape
    g = conv_same(img, smooth_k) if smooth_k is not None else img
    g = g.clamp_min(0)
    xs = torch.arange(W, device=img.device, dtype=g.dtype) - (W - 1) / 2
    ys = torch.arange(H, device=img.device, dtype=g.dtype) - (H - 1) / 2
    c = torch.zeros(N, 2, device=img.device, dtype=g.dtype)
    for _ in range(iters):
        wx = torch.exp(-(xs[None] - c[:, :1]) ** 2 / (2 * sigma ** 2))  # (N, W)
        wy = torch.exp(-(ys[None] - c[:, 1:]) ** 2 / (2 * sigma ** 2))  # (N, H)
        w = g * wy[:, :, None] * wx[:, None, :]
        tot = w.sum((-2, -1)).clamp_min(1e-6)
        cx = (w * xs[None, None]).sum((-2, -1)) / tot
        cy = (w * ys[None, :, None]).sum((-2, -1)) / tot
        c = torch.stack([cx.clamp(-W / 2 + 2, W / 2 - 2),
                         cy.clamp(-H / 2 + 2, H / 2 - 2)], 1)
    return c


def shift2d(x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Sub-pixel shift of every (B, C, H, W) image so that a feature at
    (centre + dx, centre + dy) moves to the centre.  dx, dy are (B, C)."""
    B, C, H, W = x.shape
    theta = torch.zeros(B * C, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = dx.reshape(-1).to(x.dtype) * 2.0 / W
    theta[:, 1, 2] = dy.reshape(-1).to(x.dtype) * 2.0 / H
    grid = F.affine_grid(theta, (B * C, 1, H, W), align_corners=False)
    out = F.grid_sample(x.reshape(B * C, 1, H, W), grid, mode="bicubic",
                        padding_mode="zeros", align_corners=False)
    return out.reshape(B, C, H, W)


def continuum_model(x: torch.Tensor, rolls: torch.Tensor, roll_sign: float,
                    order: int = 1, halfwidth: int = 9,
                    row_smooth: float = 0.7, center: torch.Tensor = None,
                    profile: str = "grism"):
    """Model the continuum streak in each NATIVE grism stamp.

    A source pixel's continuum is spread along its row over far more than the
    32-px stamp, so within the stamp the continuum is, to good accuracy,
        C_j(row, col) = R_j(row) · P_j(col),
    where R_j is the row profile of the galaxy in roll j's frame and P_j is a
    slowly varying spectral shape.  R_j comes from the imaging (F129 + F158)
    rotated into roll j's frame and summed along rows, so it is independent
    of the grism data (profile='imaging'), or is measured directly from the
    grism's own line-free outer columns (profile='grism', default).  The grism
    profile is needed when the continuum is a separate disk with its own
    position and size, as in the pilot sets (truth.Halpha.cont.*); P_j is a
    polynomial of degree `order` in the column,
    fitted by least squares on columns at least `halfwidth` px from the line
    (the stamp centre, or `center` (B, 4) column offsets when the line sits
    elsewhere), where there is no line emission, then extrapolated through
    the line region and subtracted.

    x : (B, 6, H, W) stamps in background-sigma units, native orientations.
    Returns the continuum (B, 4, H, W) and the polynomial coefficients
    (B, 4, order + 1)."""
    B, _, H, W = x.shape
    dev = x.device
    c0 = (W - 1) / 2
    ctr = (torch.zeros(B, 4, device=dev, dtype=x.dtype) if center is None
           else center.to(x.dtype))
    colpos = torch.arange(W, device=dev, dtype=x.dtype) - c0
    fit = ((colpos[None, None] - ctr[..., None]).abs() >= halfwidth).to(x.dtype)
    if profile == "grism":
        # row profile of the continuum streak = mean over line-free columns
        R = (x[:, 2:] * fit[:, :, None, :]).sum(-1) / fit.sum(-1, keepdim=True).clamp_min(1)
    else:
        img = (x[:, 0] + x[:, 1])[:, None].expand(-1, 4, -1, -1).contiguous()
        img_g = rotate_batch(img, -roll_sign * rolls)      # into grism frames
        R = img_g.sum(-1)                                   # (B, 4, H)
    if row_smooth > 0:
        rr = max(1, int(np.ceil(3 * row_smooth)))
        k = torch.exp(-torch.arange(-rr, rr + 1, device=dev, dtype=R.dtype) ** 2
                      / (2 * row_smooth ** 2))
        R = F.conv1d(R.reshape(B * 4, 1, H), (k / k.sum())[None, None],
                     padding=rr).reshape(B, 4, H)
    tcol = (torch.arange(W, device=dev, dtype=R.dtype) - c0) / c0
    basis = torch.stack([tcol ** k for k in range(order + 1)], -1)   # (W, K)
    T = R[..., None, None] * basis[None, None, None]        # (B, 4, H, W, K)
    wts = fit[:, :, None, :].expand(-1, -1, H, -1).reshape(B * 4, -1)  # (B4, H*W)
    K = order + 1
    Bm = T.reshape(B * 4, H * W, K)
    tgt = x[:, 2:].reshape(B * 4, H * W)
    coef = solve_wls(Bm, tgt, wts)
    C = (T * coef.reshape(B, 4, 1, 1, K)).sum(-1)
    return C, coef.reshape(B, 4, K)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class GPUPreprocessor:
    """prepare() -> state dict of linear, aligned, masked stamps;
    flip(state) -> mirrored state; build(state) -> network inputs;
    features(state) -> hand-built features for --arch features."""

    def __init__(self, roll_sign, align_grism, pairs, asinh_scale,
                 resample_noise, rot_aug, canonicalize, vel_channels,
                 vel_decomp, vel_eps, vel_clip, vel_smooth, moment_sigma,
                 npix, device, cont_sub=False, cont_order=1,
                 cont_halfwidth=9, cont_smooth=0.7, kin_r0=1.0,
                 recenter="off", grism_mirror=False,
                 recenter_imaging=False, cont_profile="grism",
                 angle_inputs=False, kin_scalars=False):
        self.roll_sign      = float(roll_sign)
        self.recenter       = recenter if align_grism else "off"
        self.recenter_imaging = bool(recenter_imaging)
        self.cont_profile   = cont_profile
        self.angle_inputs   = bool(angle_inputs) and align_grism
        self.kin_scalars    = bool(kin_scalars) and align_grism
        self.kin_cal        = None          # set by fit_kin_calibration()
        self.feat_norm      = None          # (mu, sd) when features feed the CNN
        self.grism_mirror   = bool(grism_mirror)
        self.cont_sub       = bool(cont_sub) and align_grism and roll_sign != 0
        self.cont_order     = int(cont_order)
        self.cont_halfwidth = int(cont_halfwidth)
        self.cont_smooth    = float(cont_smooth)
        self.kin_r0         = float(kin_r0)
        self.align_grism    = align_grism
        self.pairs          = list(pairs) if align_grism else []
        self.asinh_scale    = float(asinh_scale)
        self.resample_noise = resample_noise
        self.rot_aug        = rot_aug
        self.canonicalize   = canonicalize
        self.vel_channels   = vel_channels and bool(self.pairs)
        self.vel_decomp     = vel_decomp and self.vel_channels
        self.vel_eps        = float(vel_eps)
        self.vel_clip       = float(vel_clip)
        self.noise_scale    = 1.0
        self.mask           = aperture_mask(npix, device)
        self.xx, self.yy    = pixel_coords(npix, device)
        self.W              = torch.exp(-(self.xx ** 2 + self.yy ** 2)
                                        / (2 * moment_sigma ** 2))
        self.moment_sigma   = float(moment_sigma)
        self.smooth_k       = gaussian_kernel(vel_smooth, device)
        self.dx_k = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device)
        self.dy_k = self.dx_k.transpose(-1, -2).contiguous()

    # ── channel bookkeeping ────────────────────────────────────────────────
    @property
    def n_grism_in(self) -> int:
        n = len(GRISM_CH) + len(self.pairs)
        if self.vel_channels:
            n += 2 * len(self.pairs)
        if self.vel_decomp:
            n += 2
        return n

    def channel_names(self):
        names = IMG_CH + GRISM_CH + [f"{GRISM_CH[i]}−{GRISM_CH[j]}"
                                     for i, j in self.pairs]
        if self.vel_channels:
            names += [f"v̂ {i}-{j}" for i, j in self.pairs]
            names += [f"rel {i}-{j}" for i, j in self.pairs]
        if self.vel_decomp:
            names += ["v̂ even (asym)", "v̂ odd (sym)"]
        return names

    # ── geometry ───────────────────────────────────────────────────────────
    def photometric_pa_deg(self, img_sig: torch.Tensor) -> torch.Tensor:
        """Major-axis angle (deg) of the F129+F158 coadd, in the convention of
        rotate_batch (rotating by this angle puts the major axis on +x)."""
        _, qxx, qyy, qxy = weighted_moments(img_sig.sum(1), self.xx, self.yy, self.W)
        return torch.rad2deg(0.5 * torch.atan2(2 * qxy, qxx - qyy))

    def prepare(self, noisy, clean, var, sigma, rolls, train: bool,
                alpha: float = None) -> dict:
        if train and self.resample_noise:
            noise = var.clamp_min(0).sqrt() * torch.randn_like(clean)
        else:
            noise = noisy - clean                      # stored realisation
        x = (clean + self.noise_scale * noise) / sigma[:, :, None, None]
        if self.grism_mirror:          # detector parity: flip rows of grism
            x = torch.cat([x[:, :2], torch.flip(x[:, 2:], dims=[-2])], dim=1)
        # 1. imaging: put the galaxy's light centroid at the stamp centre, so
        #    shapes, position angles and the canonical frame are measured
        #    about the galaxy, not about the stamp (the pilot stamps are
        #    often off-centre by 1-3 px)
        if self.recenter_imaging:
            c = centroid2d(x[:, 0] + x[:, 1], smooth_k=self.smooth_k)
            x = torch.cat([shift2d(x[:, :2], c[:, :1].expand(-1, 2),
                                   c[:, 1:].expand(-1, 2)), x[:, 2:]], 1)
        # 2. grism, 2-D: put each roll's line image at the stamp centre in both
        #    directions before the continuum fit, so the four rolls register
        #    with each other and with the (centred) imaging after alignment
        if self.recenter == "2d":
            B_ = x.shape[0]
            c = centroid2d(x[:, 2:].reshape(B_ * 4, *x.shape[-2:]),
                           smooth_k=self.smooth_k).reshape(B_, 4, 2)
            x = torch.cat([x[:, :2], shift2d(x[:, 2:], c[..., 0], c[..., 1])], 1)
        # 3. continuum
        if self.cont_sub:
            C, _ = continuum_model(x, rolls, self.roll_sign, self.cont_order,
                                   self.cont_halfwidth, self.cont_smooth,
                                   profile=self.cont_profile)
            if self.recenter != "off":
                # second pass: exclude the window around where the line
                # actually is, not the stamp centre
                off = line_offsets(x[:, 2:] - C, smooth_k=self.smooth_k)
                C, _ = continuum_model(x, rolls, self.roll_sign, self.cont_order,
                                       self.cont_halfwidth, self.cont_smooth,
                                       center=off.mean(1, keepdim=True)
                                       .expand(-1, 4), profile=self.cont_profile)
            x = torch.cat([x[:, :2], x[:, 2:] - C], dim=1)
        # 4. line recentring on the continuum-free stamps
        if self.recenter == "2d":
            B_ = x.shape[0]
            c = centroid2d(x[:, 2:].reshape(B_ * 4, *x.shape[-2:]),
                           smooth_k=self.smooth_k).reshape(B_, 4, 2)
            x = torch.cat([x[:, :2], shift2d(x[:, 2:], c[..., 0], c[..., 1])], 1)
        elif self.recenter != "off":
            off = line_offsets(x[:, 2:], smooth_k=self.smooth_k)
            if self.recenter == "common":
                off = off.mean(1, keepdim=True).expand(-1, off.shape[1])
            x = torch.cat([x[:, :2], shift_cols(x[:, 2:], off)], dim=1)

        B = x.shape[0]
        scene = torch.zeros(B, device=x.device)
        if self.canonicalize:
            scene = scene + self.photometric_pa_deg(x[:, :2])
        if train and self.rot_aug:
            if self.canonicalize:
                scene = scene + 180.0 * torch.randint(0, 2, (B,), device=x.device)
            else:
                scene = scene + 360.0 * torch.rand(B, device=x.device)
        if alpha is not None:
            scene = scene + float(alpha)

        ang = scene[:, None].repeat(1, len(CHANNELS))
        if self.align_grism:
            ang[:, 2:] = ang[:, 2:] + self.roll_sign * rolls
        if torch.any(ang != 0):
            x = rotate_batch(x, ang)
        x = x * self.mask

        # Dispersion direction of each grism stamp in the output frame:
        # a shift along +columns in the native stamp appears along
        # R(-angle)(1, 0) = (cos a, -sin a).
        a = torch.deg2rad(ang[:, 2:])
        dirs = torch.stack([torch.cos(a), -torch.sin(a)], dim=-1)   # (B, 4, 2)
        return {"img": x[:, :2], "gr": x[:, 2:], "dirs": dirs}

    @staticmethod
    def flip(state: dict) -> dict:
        """Mirror x -> -x.  g+ is unchanged, g× changes sign."""
        dirs = state["dirs"].clone()
        dirs[..., 0] = -dirs[..., 0]
        return {"img": torch.flip(state["img"], dims=[-1]),
                "gr": torch.flip(state["gr"], dims=[-1]), "dirs": dirs}

    # ── kinematics from roll pairs ─────────────────────────────────────────
    def pair_terms(self, state: dict, i: int, j: int):
        """D = G_i - G_j, smoothed pair mean S, and J = u·∇S with
        u = d_i - d_j.  To first order D ≈ -s v(x) J."""
        gi, gj = state["gr"][:, i], state["gr"][:, j]
        D = gi - gj
        S = conv_same(0.5 * (gi + gj), self.smooth_k)
        gx, gy = conv_same(S, self.dx_k), conv_same(S, self.dy_k)
        u = state["dirs"][:, i] - state["dirs"][:, j]                 # (B, 2)
        J = (u[:, 0, None, None] * gx + u[:, 1, None, None] * gy) * self.mask
        return D, S, J

    def velocity_maps(self, state: dict):
        vs, rs = [], []
        msum = self.mask.sum()
        for i, j in self.pairs:
            D, _, J = self.pair_terms(state, i, j)
            J2 = J * J
            eps = self.vel_eps * (J2.sum(dim=(-2, -1)) / msum)[:, None, None] + 1e-6
            vs.append((-D * J / (J2 + eps)).clamp(-self.vel_clip, self.vel_clip))
            rs.append(J2 / (J2 + eps))
        v, r = torch.stack(vs, 1) * self.mask, torch.stack(rs, 1) * self.mask
        return v, r

    # ── network inputs ─────────────────────────────────────────────────────
    def build(self, state: dict):
        img, gr = state["img"], state["gr"]
        parts = [gr]
        if self.pairs:
            parts.append(torch.stack([gr[:, i] - gr[:, j] for i, j in self.pairs], 1))
        s = self.asinh_scale
        out_img = torch.asinh(img / s)
        out_gr = torch.asinh(torch.cat(parts, 1) / s)
        if self.vel_channels:
            v, r = self.velocity_maps(state)
            extra = [v, r]
            if self.vel_decomp:
                vbar = (v * r).sum(1) / r.sum(1).clamp_min(1e-3)
                rot = torch.rot90(vbar, 2, dims=(-2, -1))
                extra.append(torch.stack([vbar + rot, vbar - rot], 1) * self.mask)
            out_gr = torch.cat([out_gr] + extra, 1)
        return out_img, out_gr

    # ── hand-built features (--arch features) ──────────────────────────────
    ANGLE_NAMES = ["disp_cos1", "disp_sin1", "disp_cos2", "disp_sin2",
                   "disp_cos4", "disp_sin4"]

    @staticmethod
    def angle_features(state: dict) -> torch.Tensor:
        """Direction of roll 0's dispersion in the frame the network sees,
        as cos/sin of 1, 2 and 4 times the angle (B, 6).  Every roll's
        dispersion follows from it (fixed roll offsets).  A PSF that is not
        round is fixed to the detector, so its imprint on the aligned stamps
        depends on this angle; with it, a model can learn and remove that
        imprint.  Mirrors correctly under flip() because it is derived from
        the (mirrored) dispersion directions."""
        d = state["dirs"][:, 0]                                  # (B, 2)
        a = torch.atan2(-d[:, 1], d[:, 0])
        return torch.stack([torch.cos(a), torch.sin(a), torch.cos(2 * a),
                            torch.sin(2 * a), torch.cos(4 * a),
                            torch.sin(4 * a)], 1)

    def joint_kinematics(self, state: dict, centre=None, search: float = 0.0,
                         step: float = 0.5):
        """Fit one rotating-disc velocity field to all four aligned rolls.

        To first order each aligned stamp is G_j = L − s·v(x)·(d_j·∇L), with
        d_j roll j's own dispersion direction, so with L ≈ mean_j G_j
            G_j − L ≈ −(d_j·∇L) · v(x),
        linear in v.  v is modelled as a flat rotation curve seen in
        projection,  v(x) = a·x/ρ + b·y/ρ + c,  ρ = sqrt(x² + y² + r0²), and
        (a, b, c) are fitted by weighted least squares over all four rolls at
        once, so imperfectly opposed rolls (3° and 23° off) are handled with
        their exact directions instead of being differenced in pairs.
        The rotation centre defaults to the stamp centre; `centre` (B, 2) px
        (x = column, y = row, in the aligned frame) moves it, and `search` > 0
        additionally tries a grid of centres within ±search px of it and keeps,
        per galaxy, the one with the smallest weighted residual (needed when
        the velocity field is not centred on the light).
        Returns (a, b, c): the kinematic axis is atan2(b, a), the amplitude
        hypot(a, b) is in pixels of line shift."""
        gr, dirs = state["gr"], state["dirs"]
        Lm = gr.mean(1)
        Ls = conv_same(Lm, self.smooth_k)
        gx, gy = conv_same(Ls, self.dx_k), conv_same(Ls, self.dy_k)
        B = gr.shape[0]
        Js, tgts = [], []
        for j in range(gr.shape[1]):
            Js.append((dirs[:, j, 0, None, None] * gx
                       + dirs[:, j, 1, None, None] * gy) * self.mask)
            tgts.append((-(gr[:, j] - Lm)).flatten(1))
        t = torch.cat(tgts, 1)
        w = (self.W * self.mask).flatten().repeat(gr.shape[1])
        c0 = (torch.zeros(B, 2, device=gr.device) if centre is None
              else centre.to(gr.dtype))

        def solve_at(c):
            X = self.xx[None] - c[:, 0, None, None]
            Y = self.yy[None] - c[:, 1, None, None]
            rho = torch.sqrt(X ** 2 + Y ** 2 + self.kin_r0 ** 2)
            bx, by = X / rho, Y / rho
            Bm = torch.cat([torch.stack([J * bx, J * by, J], -1).flatten(1, 2)
                            for J in Js], 1)
            sol = solve_wls(Bm, t, w)
            res = ((t - (Bm * sol[:, None, :]).sum(-1)) ** 2 * w).sum(1)
            return sol, res

        sol, best = solve_at(c0)
        if search > 0:
            offs = torch.arange(-search, search + 1e-6, step, device=gr.device)
            for dx in offs:
                for dy in offs:
                    if dx == 0 and dy == 0:
                        continue
                    s_, r_ = solve_at(c0 + torch.stack([dx, dy]).to(gr.dtype)[None])
                    better = r_ < best
                    sol = torch.where(better[:, None], s_, sol)
                    best = torch.where(better, r_, best)
        return sol[:, 0], sol[:, 1], sol[:, 2]

    def moment_kinematics(self, state: dict, sigma: float = 3.0):
        """Velocity gradient from how each roll's second moments differ.

        A gradient g (v = g·x) displaces line emission along roll j's
        dispersion direction d_j by s·v, which to first order changes the
        aligned image's second-moment tensor to
            Q_j = Q0 + d_j hᵀ + h d_jᵀ,      h = s·Q0·g,
        i.e. the image is stretched for g along d_j and sheared for g across
        it.  The four rolls give 12 equations for the 5 unknowns (Q0, h),
        solved by least squares per galaxy; g ∝ Q0⁻¹ h.  Moments are
        Gaussian-windowed (sigma px) about each roll's own centroid, so no
        image gradients are needed (the gradient-based joint fit misses the
        along-dispersion stretch on the pilot stamps).
        Returns (gx, gy) in the aligned frame, in px of shift per px."""
        gr, dirs = state["gr"], state["dirs"]
        B, R_, H, W = gr.shape
        g = gr.reshape(B * R_, H, W)
        c = centroid2d(g, smooth_k=self.smooth_k)
        X = self.xx[None] - c[:, 0, None, None]
        Y = self.yy[None] - c[:, 1, None, None]
        # weights from a lightly smoothed, non-negative image: robust to noise
        gs = conv_same(g, self.smooth_k).clamp_min(0)
        w = gs * torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        f = w.sum((-2, -1)).clamp_min(1e-9)
        qxx = (w * X * X).sum((-2, -1)) / f
        qyy = (w * Y * Y).sum((-2, -1)) / f
        qxy = (w * X * Y).sum((-2, -1)) / f
        # undo the Gaussian window (exact for Gaussian images):
        #   C = (Q_w^-1 − I/sigma²)^-1, otherwise the window damps a stretch
        #   more than a shear and the response becomes anisotropic
        det_w = (qxx * qyy - qxy ** 2).clamp_min(1e-6)
        ixx, iyy, ixy = qyy / det_w, qxx / det_w, -qxy / det_w
        # never remove more than 80% of the inverse width (noisy, near-flat
        # stamps would otherwise blow up to absurd sizes)
        ixx = torch.maximum(ixx - 1.0 / sigma ** 2, 0.2 * ixx)
        iyy = torch.maximum(iyy - 1.0 / sigma ** 2, 0.2 * iyy)
        det_i = (ixx * iyy - ixy ** 2).clamp_min(1e-6)
        qxx, qyy, qxy = iyy / det_i, ixx / det_i, -ixy / det_i
        Q = torch.stack([qxx, qyy, qxy], -1).reshape(B, R_, 3)        # (B, 4, 3)
        dx, dy = dirs[..., 0], dirs[..., 1]                           # (B, 4)
        z, o = torch.zeros_like(dx), torch.ones_like(dx)
        rows_xx = torch.stack([o, z, z, 2 * dx, z], -1)
        rows_yy = torch.stack([z, o, z, z, 2 * dy], -1)
        rows_xy = torch.stack([z, z, o, dy, dx], -1)
        A = torch.stack([rows_xx, rows_yy, rows_xy], 2).reshape(B, R_ * 3, 5)
        t = Q.reshape(B, R_ * 3)
        sol = solve_wls(A, t, torch.ones(R_ * 3, device=gr.device), ridge=1e-6)
        q0 = sol[:, :3]
        hx, hy = sol[:, 3], sol[:, 4]
        tr = (q0[:, 0] + q0[:, 1]).clamp_min(0.1)
        det = torch.maximum(q0[:, 0] * q0[:, 1] - q0[:, 2] ** 2, 0.05 * (tr / 2) ** 2)
        gx = (q0[:, 1] * hx - q0[:, 2] * hy) / det
        gy = (-q0[:, 2] * hx + q0[:, 0] * hy) / det
        # physical gradients are ≲ 0.5 px of shift per px; cap outliers
        return gx.clamp(-1, 1), gy.clamp(-1, 1)

    def kin_components(self, state: dict):
        """Joint-fit and moment velocity gradients, each split into the
        component along roll 0's dispersion and across it: (B, 4) =
        [joint_along, moment_along, joint_across, moment_across]."""
        a, b, _ = self.joint_kinematics(state)
        mx, my = self.moment_kinematics(state)
        d = state["dirs"][:, 0]
        e = torch.stack([-d[:, 1], d[:, 0]], 1)
        dot = lambda vx, vy, u: vx * u[:, 0] + vy * u[:, 1]
        return torch.stack([dot(a, b, d), dot(mx, my, d),
                            dot(a, b, e), dot(mx, my, e)], 1)

    def calibrated_kinematics(self, state: dict):
        """Velocity gradient (aligned frame) combining the joint fit and the
        moment method with per-component weights fitted on training truth
        (fit_kin_calibration): along the dispersion the moment method carries
        the signal, across it the joint fit does.  Returns (vx, vy)."""
        k = self.kin_components(state)
        c = self.kin_cal
        u_al = c[0] * k[:, 0] + c[1] * k[:, 1]
        u_ac = c[2] * k[:, 2] + c[3] * k[:, 3]
        d = state["dirs"][:, 0]
        e = torch.stack([-d[:, 1], d[:, 0]], 1)
        v = u_al[:, None] * d + u_ac[:, None] * e
        return v[:, 0], v[:, 1]

    def feature_names(self):
        names = ["flux_F129", "flux_F158", "img_e1", "img_e2", "img_T"]
        for i, j in self.pairs:
            t = f"p{i}{j}"
            names += [f"{t}_kin_a", f"{t}_kin_b", f"{t}_kin_c",
                      f"{t}_flux", f"{t}_e1", f"{t}_e2"]
        if self.align_grism:
            names += ["joint_kin_a", "joint_kin_b", "joint_kin_c",
                      "joint_kin_amp", "joint_kin_dpa",
                      "mom_kin_gx", "mom_kin_gy", "mom_kin_amp", "mom_kin_dpa"]
            if self.kin_cal is not None:
                names += ["cal_kin_x", "cal_kin_y", "cal_kin_amp", "cal_kin_dpa"]
        if self.angle_inputs:
            names += self.ANGLE_NAMES
        return names

    def features(self, state: dict) -> torch.Tensor:
        xx, yy, W = self.xx, self.yy, self.W
        img = state["img"]
        f129 = (img[:, 0] * W).sum(dim=(-2, -1))
        f158 = (img[:, 1] * W).sum(dim=(-2, -1))
        _, qxx, qyy, qxy = weighted_moments(img.sum(1), xx, yy, W)
        T = (qxx + qyy)
        Ts = T.abs().clamp_min(1e-3) * torch.sign(T + 1e-12)
        cols = [f129, f158, (qxx - qyy) / Ts, 2 * qxy / Ts, T]

        sx, sy = xx / self.moment_sigma, yy / self.moment_sigma
        w = (W * self.mask).flatten()
        for i, j in self.pairs:
            D, S, J = self.pair_terms(state, i, j)
            # weighted least squares:  -D ≈ J * (a x + b y + c)
            Bm = torch.stack([J * sx, J * sy, J], dim=-1).flatten(1, 2).double()
            t = (-D).flatten(1).double()
            BtW = Bm.transpose(1, 2) * w.double()
            A = BtW @ Bm
            lam = 1e-3 * A.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-9)
            A = A + lam[:, None, None] * torch.eye(3, device=A.device, dtype=A.dtype)
            sol = torch.linalg.solve(A, BtW @ t[..., None])[..., 0].float()
            fS, sxx, syy, sxy = weighted_moments(S, xx, yy, W)
            sT = (sxx + syy).abs().clamp_min(1e-3)
            cols += [sol[:, 0], sol[:, 1], sol[:, 2], fS,
                     (sxx - syy) / sT, 2 * sxy / sT]
        if self.align_grism:
            a, b, c = self.joint_kinematics(state)
            phi = 0.5 * torch.atan2(2 * qxy, qxx - qyy)      # photometric axis
            # misalignment of kinematic and photometric axes: the g× observable
            dpa = wrap_axis_t(torch.atan2(b, a) - phi)
            cols += [a, b, c, torch.hypot(a, b), dpa]
            mx, my = self.moment_kinematics(state)
            cols += [mx, my, torch.hypot(mx, my),
                     wrap_axis_t(torch.atan2(my, mx) - phi)]
            if self.kin_cal is not None:
                vx, vy = self.calibrated_kinematics(state)
                cols += [vx, vy, torch.hypot(vx, vy),
                         wrap_axis_t(torch.atan2(vy, vx) - phi)]
        if self.angle_inputs:
            cols += list(self.angle_features(state).unbind(1))
        return torch.stack(cols, dim=1)


@torch.no_grad()
def detect_roll_sign(clean, sigma, rolls, rank_key, device, n=500):
    """Find the rotation sign that brings the grism stamps into the imaging
    frame by correlating rotated noiseless grism stamps with the noiseless
    F158 stamp.  Uses only an instrument convention, not galaxy truths."""
    order = np.argsort(-rank_key)[:n] if rank_key is not None else np.arange(
        min(n, len(clean)))
    x = torch.from_numpy(clean[order] / sigma[order][:, :, None, None]).to(device)
    r = torch.from_numpy(rolls[order]).to(device)
    mask = aperture_mask(x.shape[-1], device)

    def zn(a):
        a = a * mask
        m = a.sum(dim=(-2, -1), keepdim=True) / mask.sum()
        a = (a - m) * mask
        return a / a.flatten(-2).norm(dim=-1)[..., None, None].clamp_min(1e-12)

    ref = zn(x[:, 1])
    scores = {}
    for s in (+1.0, -1.0):
        ang = torch.zeros(len(order), len(CHANNELS), device=device)
        ang[:, 2:] = s * r
        g = rotate_batch(x, ang)[:, 2:]
        scores[s] = (zn(g) * ref[:, None]).sum(dim=(-2, -1)).mean(0).cpu().numpy()

    print("Roll-sign check: mean correlation of rotated clean grism with F158")
    for s, sc in scores.items():
        print(f"  sign {s:+.0f}: " +
              "  ".join(f"{c}={v:.3f}" for c, v in zip(GRISM_CH, sc)))
    best = max(scores, key=lambda s: scores[s].mean())
    # Rolls near 0 or 180 deg barely change under either sign, so judge the
    # confidence on the roll where the two signs differ most.
    gap = float(np.max(np.abs(scores[+1.0] - scores[-1.0])))
    if gap < 0.02:
        print(f"  [WARN] signs differ by only {gap:.3f}; check input_example.png "
              f"and set --roll_sign explicitly if needed.")
    print(f"  -> using roll_sign = {best:+.0f}")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def conv_bn_act(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU())


class Heads(nn.Module):
    """Separate regression heads for g+ and g×."""

    def __init__(self, feat_dim, hidden=256, dropout=0.3):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 1))
        self.g1, self.g2 = mlp(), mlp()

    def forward(self, f):
        return torch.cat([self.g1(f), self.g2(f)], dim=1)


class SmallTwoStream(nn.Module):
    """Two-stream CNN trained from scratch (~0.6M parameters at width 1).
    Both streams keep full resolution for their first layers and meet at
    16x16.  --width scales every channel count; --dropout2d drops whole
    feature maps after fusion, a strong regulariser for small datasets."""

    def __init__(self, n_img=2, n_grism=6, n_scalar=0,
                 drop_stream="none", width=1.0, dropout=0.3, dropout2d=0.1):
        super().__init__()
        c = lambda n: max(8, int(round(n * width)))
        self.photo = nn.Sequential(conv_bn_act(n_img, c(32)), conv_bn_act(c(32), c(32)),
                                   conv_bn_act(c(32), c(48), stride=2))
        self.grism = nn.Sequential(conv_bn_act(n_grism, c(32)), conv_bn_act(c(32), c(32)),
                                   conv_bn_act(c(32), c(48)),
                                   conv_bn_act(c(48), c(48), stride=2))
        self.fuse = nn.Sequential(nn.Conv2d(2 * c(48), c(64), 1, bias=False),
                                  nn.BatchNorm2d(c(64)), nn.SiLU(),
                                  nn.Dropout2d(dropout2d))
        self.body = nn.Sequential(conv_bn_act(c(64), c(64)),
                                  conv_bn_act(c(64), c(96), stride=2),
                                  conv_bn_act(c(96), c(96)),
                                  conv_bn_act(c(96), c(128), stride=2),
                                  conv_bn_act(c(128), c(128)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = Heads(c(128) + n_scalar, hidden=128, dropout=dropout)
        self.drop_stream = drop_stream

    def forward(self, img, gr, scal=None):
        f_img, f_gr = self.photo(img), self.grism(gr)
        if self.drop_stream == "grism":
            f_gr = torch.zeros_like(f_gr)
        elif self.drop_stream == "imaging":
            f_img = torch.zeros_like(f_img)
        f = self.pool(self.body(self.fuse(torch.cat([f_img, f_gr], 1)))).flatten(1)
        if scal is not None:
            f = torch.cat([f, scal], dim=1)
        return self.heads(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss, weight averaging, calibration and checkpoint selection
# ═══════════════════════════════════════════════════════════════════════════════

class ShearLoss(nn.Module):
    """Per-component MSE (or Huber) on g / G_SCALE.  Returns the weighted
    scalar loss and the per-component means (for logging)."""

    def __init__(self, kind="mse", huber_beta=1.0, weights=(1.0, 1.0)):
        super().__init__()
        self.kind, self.beta = kind, float(huber_beta)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, y):
        if self.kind == "mse":
            per = (pred - y) ** 2
        else:
            per = F.smooth_l1_loss(pred, y, reduction="none", beta=self.beta)
        comp = per.mean(0)
        w = self.w.to(comp.device)
        return (comp * w).sum() / w.sum(), comp.detach()


class EMA:
    """Exponential moving average of all parameters and BatchNorm buffers.
    The effective decay ramps up as min(decay, (1+n)/(10+n)) so the average
    is not dominated by the random initial weights."""

    def __init__(self, model, decay):
        self.decay, self.n = float(decay), 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def copy_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fit_calibration(preds, labels, mode="shrink", min_z=3.0):
    """Per-component (a, b) with  g ≈ a * pred + b,  fitted on (validation)
    predictions.  Guarded against fitting noise:
      * a = 0 (predict 0, the prior mean) unless the val correlation r is positive
        and significant, r·sqrt(n) >= min_z.  The best epoch is picked from
        many noisy val scores, so its val r is biased high; min_z = 3 keeps
        a 2-sigma fluctuation from being amplified.
      * otherwise a = r_adj · std(true) / std(pred), with r_adj² the
        adjusted r² = r² − (1 − r²)/(n − 2), which removes the upward bias
        of r² in finite samples.
      * b centres the predictions on the known prior mean, 0.
    Without these guards, a predictor with tiny output variance and a chance
    val correlation gets a huge slope (65.7 for g× in the x10 run).
    Returns cal (n_comp, 2) and per-component (r, z) on the fitting set."""
    nc = preds.shape[1]
    cal, info = np.zeros((nc, 2)), np.zeros((nc, 2))
    for k in range(nc):
        p, t = preds[:, k].astype(np.float64), labels[:, k].astype(np.float64)
        n = len(t)
        sp, st = p.std(), t.std()
        r = float(np.corrcoef(p, t)[0, 1]) if sp > 0 and st > 0 else 0.0
        z = r * np.sqrt(n)
        info[k] = (r, z)
        if mode == "none":
            cal[k] = (1.0, 0.0)
            continue
        a = 0.0
        if r > 0 and z >= min_z and n > 2:
            r2adj = max(0.0, r * r - (1 - r * r) / (n - 2))
            a = float(np.sqrt(r2adj) * st / sp)
        # Shears are drawn from a zero-mean prior (README), so the target
        # mean is known to be 0; estimating it from val would only add
        # noise.  With a = 0 the calibrated prediction is exactly 0.
        cal[k] = (a, float(-a * p.mean()))
    return cal, info


def apply_calibration(preds, cal):
    return (preds * cal[None, :, 0] + cal[None, :, 1]).astype(np.float32)


def selection_scores(preds, labels, mode):
    """Validation score per component, lower is better, in units of the
    null MSE (predicting 0): 1.0 = no better than 0, 0.9 = 10% lower MSE.
    'calibrated' scores the predictions after the best non-negative linear
    rescaling, i.e. 1 - r² when the correlation r is positive.  Used only to
    rank epochs; the calibration applied to test is fit_calibration's."""
    if mode == "calibrated":
        out = np.empty(preds.shape[1])
        for k in range(preds.shape[1]):
            p, t = preds[:, k].astype(np.float64), labels[:, k].astype(np.float64)
            vp = np.var(p)
            a = max(0.0, float(np.mean((p - p.mean()) * (t - t.mean())) / vp)) if vp > 0 else 0.0
            q = a * (p - p.mean()) + t.mean()
            out[k] = np.mean((q - t) ** 2) / max(np.mean(t ** 2), 1e-12)
        return out
    mse = ((preds - labels) ** 2).mean(0)
    null = (labels ** 2).mean(0)
    return mse / np.maximum(null, 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Train / eval
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceData:
    """All stamps live on the device; batches are gathered by index."""

    def __init__(self, noisy, clean, var, sigma, rolls, y, scal, device):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.noisy, self.clean, self.var = t(noisy), t(clean), t(var)
        self.sigma, self.rolls, self.y   = t(sigma), t(rolls), t(y)
        self.scal    = t(scal) if scal is not None else None
        self.device  = torch.device(device)
        self.is_cuda = self.device.type == "cuda"

    def batch(self, b):
        return (self.noisy[b], self.clean[b], self.var[b], self.sigma[b],
                self.rolls[b])


def tta_alphas(prep, tta: int):
    # Averaging over rotations the network never saw in training (rotation
    # augmentation off) mixes in predictions from untrained orientations and
    # shrinks the response (slope ~0.5 for a 0/180 average), so skip it.
    if tta <= 1 or not prep.rot_aug:
        return [None]
    if prep.canonicalize:
        return [0.0, 180.0]
    return [360.0 * k / tta for k in range(tta)]


def model_forward(model, prep, state, scal, parity: bool):
    """Forward pass, optionally symmetrised under parity
    (g+ even, g× odd under a mirror flip)."""
    def scalars(st):
        extra = []
        if prep.feat_norm is not None:
            mu, sd = prep.feat_norm
            f = ((prep.features(st) - mu) / sd).clamp(-5, 5)
            return f if scal is None else torch.cat([scal, f], 1)
        if prep.angle_inputs:
            extra.append(prep.angle_features(st))
        if prep.kin_scalars:
            if prep.kin_cal is not None:
                gx, gy = prep.calibrated_kinematics(st)
                extra.append(torch.stack([gx, gy], 1).clamp(-2, 2) * 2.0)
            else:
                gx, gy = prep.moment_kinematics(st)
                extra.append(torch.stack([gx, gy], 1).clamp(-1, 1) * 5.0)
        if not extra:
            return scal
        ex = torch.cat(extra, 1)
        return ex if scal is None else torch.cat([scal, ex], 1)

    img, gr = prep.build(state)
    shear = model(img, gr, scalars(state))
    if parity:
        st_m = prep.flip(state)
        img_m, gr_m = prep.build(st_m)
        shear_m = model(img_m, gr_m, scalars(st_m))
        shear = torch.stack([(shear[:, 0] + shear_m[:, 0]) / 2,
                             (shear[:, 1] - shear_m[:, 1]) / 2], dim=1)
    return shear


def run_epoch(model, data, idx, prep, criterion, optimizer, bs, training: bool,
              parity: bool, tta: int = 1, ema=None):
    """One pass over idx.  The loss is the shear loss and nothing else."""
    model.train(training)
    idx = np.random.permutation(idx) if training else np.asarray(idx)
    alphas = [None] if training else tta_alphas(prep, tta)

    total, n, t_fwd = 0.0, 0, 0.0
    preds, labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for s in range(0, len(idx), bs):
            b = torch.from_numpy(idx[s:s + bs]).to(data.device)
            if training and len(b) < 2:          # BatchNorm needs >1 sample
                continue
            y  = data.y[b] / G_SCALE
            sc = data.scal[b] if data.scal is not None else None

            if not training and data.is_cuda:
                torch.cuda.synchronize(data.device)
            t0 = time.perf_counter()
            out = 0.0
            for a in alphas:
                state = prep.prepare(*data.batch(b), training, alpha=a)
                out = out + model_forward(model, prep, state, sc, parity)
            out = out / len(alphas)
            if not training and data.is_cuda:
                torch.cuda.synchronize(data.device)
            t_fwd += time.perf_counter() - t0

            loss, _ = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total += loss.item() * len(b)
            n     += len(b)
            preds.append(out.detach().cpu() * G_SCALE)
            labels.append(data.y[b].cpu())

    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    mse    = ((preds - labels) ** 2).mean(axis=0)
    return dict(loss=total / n, mse1=float(mse[0]), mse2=float(mse[1]),
                preds=preds, labels=labels, time=None if training else t_fwd / n)


@torch.no_grad()
def evaluate_state(model, state_dict, data, idx, prep, criterion, bs, parity,
                   tta=1):
    """Evaluate a saved set of weights without disturbing the live model."""
    live = copy_state(model)
    model.load_state_dict(state_dict)
    r = run_epoch(model, data, idx, prep, criterion, None, bs, training=False,
                  parity=parity, tta=tta)
    model.load_state_dict(live)
    return r


def plot_loss_curves(history: dict, out_dir: Path, ramp_end: int = 0, best=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["train_loss"], label="Train", color="#2E86AB")
    axes[0].plot(history["val_loss"],   label="Val (EMA)", color="#E84855")
    axes[0].set_ylabel("shear loss (g / 0.03)"); axes[0].set_title("Shear loss")
    for ax, key, name in [(axes[1], "rmse_g1", "g+"), (axes[2], "rmse_g2", "g×")]:
        ax.plot(history[f"train_{key}"], label=f"Train {name}", color="#2E86AB")
        ax.plot(history[f"val_{key}"],   label=f"Val {name}",   color="#E84855")
        ax.axhline(history[f"null_{key}"], color="k", ls="--", lw=1, label="predict 0")
        ax.set_ylabel("RMSE"); ax.set_title(f"{name} RMSE")
    axes[3].plot(history["score_g1"], color="#2E86AB", label="g+")
    axes[3].plot(history["score_g2"], color="#E84855", label="g×")
    axes[3].axhline(1.0, color="k", ls="--", lw=1, label="predict 0")
    axes[3].set_ylabel("val MSE / null MSE"); axes[3].set_title("Selection score")
    if best:
        for k, col in [(0, "#2E86AB"), (1, "#E84855")]:
            if best[k] > 0:
                axes[3].axvline(best[k] - 1, color=col, ls=":", lw=1.5)
    for ax in axes:
        if ramp_end:
            ax.axvline(ramp_end - 1, color="grey", ls=":", lw=1)
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: loss_curves.png")


def shear_metrics(preds, labels):
    """RMSE, null-predictor RMSE, skill, response m and additive bias c.

    m, c come from fitting pred = (1 + m) * true + c.  An MSE-trained
    estimator approximates the posterior mean, which shrinks noisy galaxies
    toward 0, so expect m < 0 that grows with noise.
    """
    out = {}
    for k, tag in enumerate(["g_plus", "g_cross"]):
        t, p = labels[:, k], preds[:, k]
        mse  = float(np.mean((p - t) ** 2))
        null = float(np.mean(t ** 2))
        if len(t) > 2:
            slope, icpt = np.polyfit(t, p, 1)
            r = float(np.corrcoef(t, p)[0, 1]) if np.std(p) > 0 else 0.0
            r_err = (1 - r ** 2) / np.sqrt(len(t) - 1)
        else:
            slope = icpt = r = r_err = np.nan
        out[tag] = dict(rmse=np.sqrt(mse), null_rmse=np.sqrt(null),
                        skill=1 - mse / null if null > 0 else np.nan,
                        m=float(slope - 1), c=float(icpt), r=r, r_err=float(r_err))
    return out


def print_metrics(met, header):
    print(f"\n  {header}")
    print(f"  {'':>8} {'RMSE':>8} {'null':>8} {'skill':>7} {'m':>8} "
          f"{'c':>9} {'r ± err':>14}")
    for tag, lab in [("g_plus", "g+"), ("g_cross", "g×")]:
        q = met[tag]
        print(f"  {lab:>8} {q['rmse']:8.4f} {q['null_rmse']:8.4f} "
              f"{q['skill']:7.3f} {q['m']:+8.3f} {q['c']:+9.5f} "
              f"{q['r']:6.3f} ± {q['r_err']:.3f}")
    print("  skill = 1 − MSE/MSE(predict 0);  pred = (1+m)·true + c;  "
          "r is significant if |r| >> err")



def kfold_split(groups, k, fold, seed=42):
    """Group k-fold: every catalog galaxy (pop.id) lands in exactly one fold.
    Fold `fold` is the test set, fold `fold+1` the validation set, the rest
    train."""
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = dict(zip(ug[perm], np.arange(len(ug)) % k))
    f = np.array([fold_of[g] for g in groups])
    te = np.where(f == fold)[0]
    va = np.where(f == (fold + 1) % k)[0]
    tr = np.where((f != fold) & (f != (fold + 1) % k))[0]
    return tr, va, te


def ensemble_response(preds, labels, groups=None, nboot=500, seed=0):
    """Per component: response m and additive bias c from
    mean(pred | true) = (1 + m)·true + c, and the correlation r, each with a
    bootstrap error that resamples whole catalog galaxies (pop.id), so
    repeated galaxies are not counted as independent."""
    n = len(labels)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.default_rng(seed)

    def stats(ix):
        out = []
        for k in range(labels.shape[1]):
            t, p = labels[ix, k], preds[ix, k]
            sl, ic = np.polyfit(t, p, 1) if np.std(t) > 0 else (np.nan, np.nan)
            r = np.corrcoef(t, p)[0, 1] if np.std(p) > 0 else 0.0
            out.append((sl - 1, ic, r))
        return np.array(out)                                  # (2, 3)

    base = stats(np.arange(n))
    boots = []
    for _ in range(nboot):
        pick = rng.integers(0, len(ug), len(ug))
        boots.append(stats(np.concatenate([members[k] for k in pick])))
    err = np.nanstd(np.array(boots), axis=0)
    return base, err


def print_ensemble(base, err, header):
    print(f"\n  {header}")
    print(f"  {'':>6} {'m (response − 1)':>20} {'c (additive)':>22} {'r':>16}  r/err")
    for k, lab in enumerate(["g+", "g×"]):
        m, c, r = base[k]
        em, ec, er = err[k]
        print(f"  {lab:>6} {m:+9.3f} ± {em:.3f} {c:+12.5f} ± {ec:.5f} "
              f"{r:+8.3f} ± {er:.3f}  {r / er if er > 0 else 0:5.1f}σ")
    print("  A useful shear estimator has m and c consistent with what a "
          "calibration absorbs; r/err is the detection significance.")


def plot_binned_response(preds, labels, out_path, title, nbins=10):
    """Mean prediction in quantile bins of the true shear, with errors: the
    ensemble response that weak lensing actually uses."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, k, name, col in zip(axes, [0, 1], ["g+", "g×"], ["#2E86AB", "#E84855"]):
        t, p = labels[:, k], preds[:, k]
        edges = np.quantile(t, np.linspace(0, 1, nbins + 1))
        which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
        bc, bm, be = [], [], []
        for b in range(nbins):
            m = which == b
            if m.sum() > 5:
                bc.append(t[m].mean()); bm.append(p[m].mean())
                be.append(p[m].std() / np.sqrt(m.sum()))
        ax.errorbar(bc, bm, yerr=be, fmt="o", color=col, capsize=3, label="binned mean")
        sl, ic = np.polyfit(t, p, 1)
        xs = np.array([t.min(), t.max()])
        ax.plot(xs, ic + sl * xs, color=col, lw=1.5, label=f"fit slope {sl:.3f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"mean {name} predicted")
        ax.set_title(f"{name}: ensemble response [{title}]")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pred_vs_true(preds, labels, split_name, out_dir: Path, nbins=8):
    """Scatter plus the mean prediction in bins of true shear: the binned
    response is what weak lensing uses, and it shows a weak but real signal
    that the scatter hides."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, name, color in zip(axes, [0, 1], ["g+", "g×"],
                                    ["#2E86AB", "#E84855"]):
        t, p = labels[:, col], preds[:, col]
        lim = max(np.abs(t).max(), np.abs(p).max()) * 1.1
        ax.scatter(t, p, s=6, alpha=0.35, color=color, rasterized=True)
        edges = np.quantile(t, np.linspace(0, 1, nbins + 1))
        which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1)
        bc, bm, be = [], [], []
        for k in range(nbins):
            s = which == k
            if s.sum() >= 3:
                bc.append(t[s].mean()); bm.append(p[s].mean())
                be.append(p[s].std() / np.sqrt(s.sum()))
        ax.errorbar(bc, bm, yerr=be, fmt="o", color="k", ms=5, capsize=2,
                    label="binned mean")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        if len(t) > 2:
            sl, ic = np.polyfit(t, p, 1)
            ax.plot([-lim, lim], [ic - sl * lim, ic + sl * lim], color="orange",
                    lw=1.5, label=f"fit slope {sl:.3f}")
        mse = float(np.mean((p - t) ** 2))
        ax.set_xlabel(f"{name} true"); ax.set_ylabel(f"{name} predicted")
        ax.set_title(f"{name} [{split_name}]  RMSE={np.sqrt(mse):.4f}")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.tight_layout()
    fname = out_dir / f"pred_vs_true_{split_name}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_binned_rmse(preds, labels, df_split, split_name, out_dir: Path, nbins=6):
    """RMSE of g+ and g× in quantile bins of S/N and inclination."""
    specs = [(c, l) for c, l in [("snr_eff.line_total", "realised line S/N"),
                                  ("snr_eff.F158", "realised F158 S/N"),
                                  ("truth.cosi", "cos i")]
             if c in df_split.columns]
    if not specs:
        return
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4),
                             squeeze=False)
    for ax, (col, lab) in zip(axes[0], specs):
        v = df_split[col].to_numpy()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, nbins + 1)))
        if len(edges) < 2:
            continue
        which = np.clip(np.searchsorted(edges, v, side="right") - 1,
                        0, len(edges) - 2)
        cen, r1, r2, nul = [], [], [], []
        for k in range(len(edges) - 1):
            s = which == k
            if s.sum() < 5:
                continue
            cen.append(np.median(v[s]))
            r1.append(np.sqrt(np.mean((preds[s, 0] - labels[s, 0]) ** 2)))
            r2.append(np.sqrt(np.mean((preds[s, 1] - labels[s, 1]) ** 2)))
            nul.append(np.sqrt(np.mean(labels[s] ** 2)))
        ax.plot(cen, r1, "o-", color="#2E86AB", label="g+")
        ax.plot(cen, r2, "s-", color="#E84855", label="g×")
        ax.plot(cen, nul, "k--", lw=1, label="predict 0")
        if col.startswith("snr"):
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE vs {lab} [{split_name}]")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = out_dir / f"binned_rmse_{split_name}.png"
    fig.savefig(str(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


@torch.no_grad()
def plot_input_example(data, prep, rows, df, out_dir: Path):
    """Show exactly what the network sees (eval preprocessing) for a few rows."""
    b = torch.from_numpy(np.asarray(rows)).to(data.device)
    img, gr = prep.build(prep.prepare(*data.batch(b), train=False))
    chans = torch.cat([img, gr], 1).cpu().numpy()
    names = prep.channel_names()
    nr, nc = chans.shape[0], chans.shape[1]
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc, 1.9 * nr), squeeze=False)
    for r in range(nr):
        row = df.iloc[int(rows[r])]
        for c in range(nc):
            ax, a = axes[r, c], chans[r, c]
            signed = c >= len(CHANNELS) and not names[c].startswith("rel")
            if signed:
                v = float(np.abs(a).max()) or 1.0
                ax.imshow(a, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            else:
                ax.imshow(a, origin="lower", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=6)
        info = [f"row {int(rows[r])}"]
        for k, fmt in [("truth.cosi", "cosi={:.2f}"),
                       ("snr_eff.line_total", "S/N={:.0f}"),
                       ("truth.g_cross", "g×={:+.3f}")]:
            if k in row:
                info.append(fmt.format(row[k]))
        axes[r, 0].set_ylabel("\n".join(info), fontsize=7)
    frame = "canonical (major axis along x)" if prep.canonicalize else "sky"
    fig.suptitle(f"Network inputs after preprocessing ({frame} frame, "
                 f"noise_scale={prep.noise_scale:.2f})", fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_dir / "input_example.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: input_example.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature baseline (--arch features)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_features(data, prep, idx, train, bs, alpha=None, flip=False):
    out = []
    for s in range(0, len(idx), bs):
        b = torch.from_numpy(np.asarray(idx[s:s + bs])).to(data.device)
        st = prep.prepare(*data.batch(b), train, alpha=alpha)
        if flip:
            st = prep.flip(st)
        f = prep.features(st)
        if data.scal is not None:
            f = torch.cat([f, data.scal[b]], 1)
        out.append(f.cpu())
    return torch.cat(out).numpy()


def run_feature_baseline(args, data, prep, tr, va, te, df, y, out_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = prep.feature_names() + (["z"] if data.scal is not None else [])
    print(f"\nFeature baseline with {len(names)} features: {names}")
    if not prep.canonicalize:
        print("[WARN] features are frame-dependent; --canonicalize is strongly "
              "recommended with --arch features")

    # Training set: fresh noise draws, each with its mirror (g× -> -g×)
    Xs, Ys = [], []
    for d in range(args.feat_draws):
        for flip in (False, True):
            Xs.append(collect_features(data, prep, tr, True, 512, flip=flip))
            Ys.append(y[tr] * (np.array([1.0, -1.0]) if flip else 1.0))
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    print(f"  training rows (draws x mirror): {len(X)}")

    X_va = collect_features(data, prep, va, False, 512)
    models = []
    for k, tag in enumerate(["g+", "g×"]):
        reg = HistGradientBoostingRegressor(
            max_iter=args.feat_max_iter, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=args.seed)
        reg.fit(X, Y[:, k])
        val_mse = [np.mean((p - y[va, k]) ** 2) for p in reg.staged_predict(X_va)]
        best = int(np.argmin(val_mse)) + 1
        print(f"  {tag}: best n_iter on val = {best}  "
              f"(val RMSE {np.sqrt(val_mse[best - 1]):.4f}, "
              f"null {np.sqrt(np.mean(y[va, k] ** 2)):.4f})")
        reg.set_params(max_iter=best)
        reg.fit(X, Y[:, k])
        models.append(reg)

    def predict(idx):
        acc = np.zeros((len(idx), 2))
        alphas = tta_alphas(prep, args.tta)
        for a in alphas:
            f0 = collect_features(data, prep, idx, False, 512, alpha=a)
            f1 = collect_features(data, prep, idx, False, 512, alpha=a, flip=True)
            acc[:, 0] += 0.5 * (models[0].predict(f0) + models[0].predict(f1))
            acc[:, 1] += 0.5 * (models[1].predict(f0) - models[1].predict(f1))
        return (acc / len(alphas)).astype(np.float32)

    va_pred, te_pred = predict(va), predict(te)
    met = shear_metrics(te_pred, y[te])
    print_metrics(met, "Test metrics (feature baseline)")

    feats_all = pd.DataFrame(collect_features(data, prep, np.arange(len(df)),
                                              False, 512), columns=names)
    for c in ["pop.id", "truth.g_plus", "truth.g_cross", "truth.cosi",
              "snr_eff.line_total"]:
        if c in df.columns:
            feats_all[c] = df[c].to_numpy()
    feats_all.to_csv(out_dir / "features_all.csv", index=False)
    print("  Saved: features_all.csv (eval features for every row)")
    return met, va_pred, te_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def fit_kin_calibration(prep, data, idx, df, bs=512):
    """Fit the per-component weights of calibrated_kinematics on TRAINING
    galaxies only, against the true velocity gradient
        u = vcirc·sin i·s(z)·(cos θ_int, sin θ_int),
    expressed along / across roll 0's dispersion.  Runs with the canonical
    frame switched off so the aligned frame is the sky frame (the frame of
    theta_int).  Returns the 4 weights, or None if truth columns are missing."""
    need = ["truth.vel.vcirc", "truth.cosi", "truth.z", "truth.theta_int"]
    if not all(c in df.columns for c in need):
        print("[WARN] --kin_calibrate needs " + ", ".join(need) + "; skipped")
        return None
    canon, ns = prep.canonicalize, prep.noise_scale
    prep.canonicalize = False
    K, D = [], []
    for s in range(0, len(idx), bs):
        b = torch.from_numpy(np.asarray(idx[s:s + bs])).to(data.device)
        st = prep.prepare(*data.batch(b), train=False)
        K.append(prep.kin_components(st).cpu())
        D.append(st["dirs"][:, 0].cpu())
    prep.canonicalize, prep.noise_scale = canon, ns
    K = torch.cat(K).numpy().astype(np.float64)
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(K, 1, axis=0), np.percentile(K, 99, axis=0)
    K = np.clip(K, lo, hi)
    d = torch.cat(D).numpy().astype(np.float64)
    e = np.column_stack([-d[:, 1], d[:, 0]])
    sub = df.iloc[np.asarray(idx)]
    cosi = sub["truth.cosi"].to_numpy()
    A = (sub["truth.vel.vcirc"].to_numpy() * np.sqrt(1 - cosi ** 2)
         * 656.28 * (1 + sub["truth.z"].to_numpy()) / (2.99792458e5 * 1.1))
    th = sub["truth.theta_int"].to_numpy()
    u = np.column_stack([A * np.cos(th), A * np.sin(th)])
    u_al, u_ac = (u * d).sum(1), (u * e).sum(1)
    w_al, *_ = np.linalg.lstsq(K[:, [0, 1]], u_al, rcond=None)
    w_ac, *_ = np.linalg.lstsq(K[:, [2, 3]], u_ac, rcond=None)
    r_al = np.corrcoef(K[:, [0, 1]] @ w_al, u_al)[0, 1]
    r_ac = np.corrcoef(K[:, [2, 3]] @ w_ac, u_ac)[0, 1]
    for k, nm in enumerate(["joint along", "moments along", "joint across",
                            "moments across"]):
        tgt = u_al if k < 2 else u_ac
        print(f"   kinematic input {nm:<15} corr with truth {np.corrcoef(K[:, k], tgt)[0, 1]:+.3f}")
    print(f"Kinematic calibration (train split): along = {w_al[0]:+.3f}·joint "
          f"{w_al[1]:+.3f}·moments (r {r_al:.2f});  across = {w_ac[0]:+.3f}·joint "
          f"{w_ac[1]:+.3f}·moments (r {r_ac:.2f})")
    return np.concatenate([w_al, w_ac]).astype(np.float32)


def parse_pairs(s: str):
    if s.lower() in ("", "none"):
        return []
    return [tuple(int(v) for v in tok.split("-")) for tok in s.split(",")]



def parse_args():
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="KL shear regressor for Roman "
                                            "imaging + grism stamps (v3).")
    p.add_argument("--parquet", required=True)
    p.add_argument("--meta", default=None, help="meta_pilot_v2_all.json")
    p.add_argument("--outdir", default="./kl_grism_v3")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=30,
                   help="Stop when neither component's val score has improved "
                        "for this many epochs (counted after any noise ramp)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--kfold", type=int, default=0,
                   help="K-fold cross-validation over catalog galaxies "
                        "(K >= 3, e.g. 5): trains K models, each galaxy is tested "
                        "once, and pooled out-of-fold metrics are reported")
    p.add_argument("--fold", type=int, default=-1,
                   help="Run a single fold of --kfold (default: all folds)")
    p.add_argument("--max_rows", type=int, default=0,
                   help="Use only the first N rows (quick tests)")
    # selection
    p.add_argument("--min_line_snr", type=float, default=10.0,
                   help="Keep only rows with snr_col >= this value "
                        "(applied before splitting; 0 disables the cut)")
    p.add_argument("--snr_col", default="line_snr",
                   help="Column the S/N cut is applied to. The README "
                        "recommends snr_eff.line_total (realised S/N) over "
                        "line_snr (the cruder selection value)")
    # loss, averaging, checkpoint selection, calibration
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber_beta", type=float, default=1.0,
                   help="Huber transition, in units of g / 0.03")
    p.add_argument("--component_weights", default="1,1",
                   help="Loss weights for g+,g×, e.g. '1,2' to emphasise g×")
    p.add_argument("--ema", type=float, default=0.995,
                   help="EMA decay of the weights that are evaluated and "
                        "saved (0 = off)")
    p.add_argument("--select", choices=["calibrated", "raw"], default="calibrated",
                   help="Validation score used to pick checkpoints")
    p.add_argument("--per_component_ckpt", action=B, default=True,
                   help="Keep separate best weights for g+ and g×")
    p.add_argument("--eval_weights", choices=["best", "final"], default="best",
                   help="Weights used for all predictions and plots: the "
                        "val-selected checkpoints, or the weights at the last "
                        "epoch (use 'final' for overfitting tests)")
    p.add_argument("--calibrate", choices=["shrink", "none"], default="shrink",
                   help="Fit g = a·pred + b per component on val, apply to test")
    p.add_argument("--cal_min_z", type=float, default=3.0,
                   help="Calibrate a component only if its val correlation is "
                        "at least this many sigma (r·sqrt(n)); otherwise "
                        "predict 0 (the prior mean) for it")
    # noise control
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Multiply the noise by this factor (train and eval). "
                        "0 = noiseless, 0.25 = ~4x higher S/N")
    p.add_argument("--noise_start", type=float, default=0.0,
                   help="Noise scale at epoch 1 when ramping")
    p.add_argument("--noise_ramp_epochs", type=int, default=0,
                   help="Ramp noise from --noise_start to --noise_scale over "
                        "this many epochs (0 = no ramp)")
    p.add_argument("--resample_noise", action=B, default=True,
                   help="Fresh noise realisation each training step")
    # geometry / symmetry
    p.add_argument("--align_grism", action=B, default=True,
                   help="Rotate grism stamps into the imaging frame")
    p.add_argument("--roll_sign", choices=["auto", "1", "-1"], default="-1",
                   help="kl_diagnostics.py found -1 for pilot_v2; 'auto' "
                        "re-detects it from the data")
    p.add_argument("--rolls", type=float, nargs=4, default=None,
                   help="Override roll angles (deg) for gr_roll0..3")
    p.add_argument("--pairs", default="0-3,1-2",
                   help="Roughly opposed grism pairs; 'none' to disable")
    p.add_argument("--canonicalize", action=B, default=True,
                   help="Rotate each scene so the photometric major axis is "
                        "along x")
    p.add_argument("--parity", action=B, default=True,
                   help="Parity-symmetrised predictions (g× odd, g+ even)")
    p.add_argument("--rot_aug", action=B, default=True,
                   help="Rotation augmentation (random 180° if canonical, "
                        "random angle otherwise)")
    p.add_argument("--tta", type=int, default=8,
                   help="Test-time rotations (canonical frame uses 0 and 180)")
    p.add_argument("--moment_sigma", type=float, default=3.0)
    # velocity channels
    p.add_argument("--vel_channels", action=B, default=True)
    p.add_argument("--vel_decomp", action=B, default=False)
    p.add_argument("--vel_eps", type=float, default=0.5)
    p.add_argument("--vel_clip", type=float, default=2.0)
    p.add_argument("--vel_smooth", type=float, default=1.0)
    p.add_argument("--asinh_scale", type=float, default=2.0)
    # continuum subtraction
    p.add_argument("--cont_sub", action=B, default=True,
                   help="Subtract a continuum model (imaging row profile x "
                        "polynomial along the dispersion) from each grism "
                        "stamp before alignment; test with kl_continuum_test.py")
    p.add_argument("--kin_calibrate", action=B, default=True,
                   help="Combine the joint-fit and moment velocity gradients "
                        "with weights fitted on the training split's truth "
                        "(fixes the along-dispersion blind spot of the joint fit)")
    p.add_argument("--feat_scalars", action=B, default=True,
                   help="Feed all hand-built features (the same ones the "
                        "tree baseline uses, standardised) into the CNN head")
    p.add_argument("--kin_scalars", action=B, default=True,
                   help="Give the CNN the moment-based velocity gradient "
                        "(2 numbers per galaxy) as extra scalar inputs")
    p.add_argument("--angle_inputs", action=B, default=True,
                   help="Give the model the dispersion direction relative to "
                        "each galaxy (cos/sin of 1, 2, 4x the angle), so it can "
                        "learn and remove detector-fixed PSF effects")
    p.add_argument("--cont_profile", choices=["grism", "imaging"], default="grism",
                   help="Where the continuum row profile comes from: the grism's "
                        "own line-free outer columns (default; right when the "
                        "continuum is a separate disk) or the imaging")
    p.add_argument("--cont_order", type=int, default=1,
                   help="Polynomial degree of the continuum along the dispersion")
    p.add_argument("--cont_halfwidth", type=int, default=9,
                   help="Fit the continuum on columns at least this many px "
                        "from the stamp centre")
    p.add_argument("--cont_smooth", type=float, default=0.7,
                   help="Gaussian smoothing (px) of the imaging row profile")
    p.add_argument("--recenter_imaging", action=B, default=True,
                   help="Shift each imaging stamp so the galaxy's light "
                        "centroid is at the centre before measuring shapes")
    p.add_argument("--recenter_line", choices=["off", "common", "per_roll", "2d"],
                   default="2d",
                   help="Re-register the grism line images: 2d = shift each "
                        "roll's line centroid to the stamp centre in both "
                        "directions (default); common / per_roll = along the "
                        "dispersion only; off = use stamps as delivered")
    p.add_argument("--grism_mirror", action=B, default=False,
                   help="Flip grism stamps across the dispersion axis before "
                        "alignment (only if kl_continuum_test.py finds a "
                        "parity flip)")
    # model
    p.add_argument("--arch", choices=["small", "features"], default="small")
    p.add_argument("--width", type=float, default=1.0,
                   help="Channel multiplier for the CNN (0.5 halves it)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout in the regression heads")
    p.add_argument("--dropout2d", type=float, default=0.1,
                   help="Feature-map dropout after fusion")
    p.add_argument("--use_z", action="store_true",
                   help="Append redshift as a scalar input")
    p.add_argument("--drop_stream", choices=["none", "grism", "imaging"],
                   default="none", help="Ablation: zero one stream's features")
    # feature baseline
    p.add_argument("--feat_draws", type=int, default=4)
    p.add_argument("--feat_max_iter", type=int, default=600)
    return p.parse_args()


def run_once(args):
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    comp_w = tuple(float(v) for v in args.component_weights.split(","))
    if len(comp_w) != 2:
        raise ValueError("--component_weights needs two values, e.g. '1,1'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df, meta, noisy, clean, var, sigma = load_table(args.parquet, args.meta)
    df["src_row"] = np.arange(len(df))     # parquet row, stable across S/N cuts
    if args.max_rows:
        m = args.max_rows
        df = df.iloc[:m].reset_index(drop=True)
        noisy, clean, var, sigma = noisy[:m], clean[:m], var[:m], sigma[:m]

    if args.min_line_snr > 0:
        if args.snr_col not in df.columns:
            cands = [c for c in df.columns if "snr" in c.lower()]
            raise ValueError(f"--snr_col '{args.snr_col}' not in parquet. "
                             f"S/N columns available: {cands}")
        keep = df[args.snr_col].to_numpy() >= args.min_line_snr
        print(f"Line S/N cut: {args.snr_col} >= {args.min_line_snr} keeps "
              f"{keep.sum()}/{len(keep)} rows ({keep.mean():.1%}); "
              f"{df.loc[keep, 'pop.id'].nunique() if 'pop.id' in df else '?'} "
              f"unique galaxies")
        if keep.sum() < 50:
            raise RuntimeError("Fewer than 50 rows survive the S/N cut; "
                               "lower --min_line_snr.")
        df = df[keep].reset_index(drop=True)
        noisy, clean, var, sigma = noisy[keep], clean[keep], var[keep], sigma[keep]

    check_noise_model(noisy, clean, var)
    rolls = get_roll_angles(df, meta, args.rolls)

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing target columns: {missing}")
    y = df[TARGETS].to_numpy(np.float32)

    if "pop.id" in df.columns:
        groups = df["pop.id"].to_numpy()
    else:
        print("[WARN] pop.id not found; splitting by row (galaxies may leak)")
        groups = np.arange(len(df))
    if args.kfold > 1:
        tr, va, te = kfold_split(groups, args.kfold, args.fold, args.seed)
        print(f"K-fold: fold {args.fold + 1}/{args.kfold} (test = fold "
              f"{args.fold}, val = fold {(args.fold + 1) % args.kfold})")
    else:
        tr, va, te = split_by_group(groups, args.train_frac, args.val_frac,
                                    args.seed)
    print(f"Split sizes — train: {len(tr)}  val: {len(va)}  test: {len(te)}")
    print(f"Unique galaxies — train: {len(np.unique(groups[tr]))}  "
          f"val: {len(np.unique(groups[va]))}  test: {len(np.unique(groups[te]))}")
    for name, ix in [("train", tr), ("val", va), ("test", te)]:
        df.iloc[ix].to_csv(out_dir / f"split_{name}.csv", index=False)

    scal, z_stats = None, None
    if args.use_z:
        z = df["truth.z"].to_numpy(np.float32)
        z_stats = (float(z[tr].mean()), float(z[tr].std() + 1e-8))
        scal = ((z - z_stats[0]) / z_stats[1])[:, None].astype(np.float32)

    # ── Preprocessing ────────────────────────────────────────────────────────
    align = args.align_grism
    pairs = parse_pairs(args.pairs)
    if not align:
        roll_sign = 0.0
        if pairs or args.parity or args.vel_channels:
            print("Grism alignment off -> pair differences, velocity channels "
                  "and parity symmetrisation disabled")
        pairs = []
    elif args.roll_sign == "auto":
        rank = (df["snr_eff.line_total"].to_numpy()[tr]
                if "snr_eff.line_total" in df.columns else None)
        roll_sign = detect_roll_sign(clean[tr], sigma[tr], rolls[tr], rank, device)
    else:
        roll_sign = float(args.roll_sign)
        print(f"Roll sign: {roll_sign:+.0f} (set explicitly)")
    parity = args.parity and align

    prep = GPUPreprocessor(
        roll_sign, align, pairs, args.asinh_scale,
        resample_noise=args.resample_noise, rot_aug=args.rot_aug,
        canonicalize=args.canonicalize, vel_channels=args.vel_channels,
        vel_decomp=args.vel_decomp, vel_eps=args.vel_eps, vel_clip=args.vel_clip,
        vel_smooth=args.vel_smooth, moment_sigma=args.moment_sigma,
        npix=noisy.shape[-1], device=device, cont_sub=args.cont_sub,
        cont_order=args.cont_order, cont_halfwidth=args.cont_halfwidth,
        cont_smooth=args.cont_smooth, recenter=args.recenter_line,
        grism_mirror=args.grism_mirror,
        recenter_imaging=args.recenter_imaging,
        cont_profile=args.cont_profile, angle_inputs=args.angle_inputs,
        kin_scalars=args.kin_scalars)
    ramp = args.noise_ramp_epochs if args.arch != "features" else 0
    prep.noise_scale = args.noise_start if ramp > 0 else args.noise_scale

    print(f"Grism inputs: {prep.channel_names()[2:]}")
    print(f"Continuum subtraction: {prep.cont_sub}"
          + (f" (order {args.cont_order}, fit |col − centre| ≥ "
             f"{args.cont_halfwidth} px)" if prep.cont_sub else "")
          + f"   imaging recentring: {prep.recenter_imaging}   line "
          f"recentring: {prep.recenter}   grism mirror: {prep.grism_mirror}")
    print(f"Dispersion-angle inputs: {prep.angle_inputs}")
    print(f"Canonical frame: {prep.canonicalize}   parity: {parity}   "
          f"rotation aug: {prep.rot_aug}   TTA: {tta_alphas(prep, args.tta)}")
    print(f"Noise: scale {args.noise_scale}"
          + (f" (ramp from {args.noise_start} over {ramp} epochs)" if ramp else "")
          + f"   resample: {prep.resample_noise}")

    data = DeviceData(noisy, clean, var, sigma, rolls, y, scal, device)
    del noisy, clean, var

    if args.kin_calibrate and align:
        w = fit_kin_calibration(prep, data, tr, df)
        if w is not None:
            prep.kin_cal = torch.as_tensor(w, device=device)

    if args.arch != "features" and args.feat_scalars:
        # the hand-built KL measurements (kinematic-photometric misalignment,
        # calibrated velocity gradient, moments, ...) go straight into the
        # CNN head; standardised on the training split
        ns0 = prep.noise_scale
        with torch.no_grad():
            F_tr = []
            for s0 in range(0, len(tr), 512):
                b = torch.from_numpy(np.asarray(tr[s0:s0 + 512])).to(device)
                F_tr.append(prep.features(prep.prepare(*data.batch(b), train=False)))
            F_tr = torch.cat(F_tr)
        prep.noise_scale = ns0
        mu = F_tr.median(0).values
        sd = (F_tr.quantile(0.84, 0) - F_tr.quantile(0.16, 0)).clamp_min(1e-6) / 2
        prep.feat_norm = (mu, sd)
        print(f"CNN head receives {F_tr.shape[1]} hand-built features "
              f"(--feat_scalars)")

    print("\nSaving an input example …")
    plot_input_example(data, prep, va[:4], df, out_dir)

    criterion = ShearLoss(args.loss, args.huber_beta, comp_w).to(device)
    history, best_epochs, cal = None, [-1, -1], None
    te_preds_raw = None

    if args.arch == "features":
        _, va_preds, te_preds_raw = run_feature_baseline(
            args, data, prep, tr, va, te, df, y, out_dir)
        va_labels, te_labels = y[va], y[te]
        tr_preds_raw, tr_labels = None, None
        te_time = float("nan")
    else:
        # ── Model ────────────────────────────────────────────────────────────
        model = SmallTwoStream(
            n_img=len(IMG_CH), n_grism=prep.n_grism_in,
            n_scalar=(0 if scal is None else scal.shape[1])
            + (len(prep.feature_names()) if prep.feat_norm is not None else
               (len(prep.ANGLE_NAMES) if prep.angle_inputs else 0)
               + (2 if prep.kin_scalars else 0)),
            drop_stream=args.drop_stream,
            width=args.width, dropout=args.dropout,
            dropout2d=args.dropout2d).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: small two-stream CNN (width {args.width})   "
              f"trainable parameters: {n_params:,}")
        print(f"Loss: {args.loss} on g/{G_SCALE}, component weights {comp_w}   "
              f"EMA: {args.ema or 'off'}   selection: {args.select}, "
              f"{'per component' if args.per_component_ckpt else 'joint'}   "
              f"calibration: {args.calibrate}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        ema = EMA(model, args.ema) if args.ema > 0 else None

        # ── Training loop ────────────────────────────────────────────────────
        history = {k: [] for k in ["train_loss", "val_loss", "train_rmse_g1",
                                   "val_rmse_g1", "train_rmse_g2", "val_rmse_g2",
                                   "score_g1", "score_g2"]}
        history["null_rmse_g1"] = float(np.sqrt(np.mean(y[va, 0] ** 2)))
        history["null_rmse_g2"] = float(np.sqrt(np.mean(y[va, 1] ** 2)))
        print(f"\nVal RMSE of predicting 0: g+ {history['null_rmse_g1']:.4f}  "
              f"g× {history['null_rmse_g2']:.4f}  (the number to beat)")
        print("Score = val MSE / MSE(predict 0) after selection-mode "
              "calibration; below 1 means real skill.  * = new best.\n")

        best_score = [np.inf, np.inf]
        best_state = [None, None]
        stale = 0
        ep_kw = dict(parity=parity)
        for epoch in range(1, args.epochs + 1):
            if ramp > 0:
                frac = min(1.0, (epoch - 1) / ramp)
                prep.noise_scale = (args.noise_start
                                    + (args.noise_scale - args.noise_start) * frac)

            t0 = time.time()
            r_tr = run_epoch(model, data, tr, prep, criterion, optimizer,
                             args.batch_size, training=True, ema=ema, **ep_kw)
            eval_state = ema.state_dict() if ema is not None else copy_state(model)
            r_va = evaluate_state(model, eval_state, data, va, prep, criterion,
                                  args.batch_size, parity)
            scheduler.step()

            score = selection_scores(r_va["preds"], r_va["labels"], args.select)
            for k, v in [("train_loss", r_tr["loss"]), ("val_loss", r_va["loss"]),
                         ("train_rmse_g1", np.sqrt(r_tr["mse1"])),
                         ("val_rmse_g1", np.sqrt(r_va["mse1"])),
                         ("train_rmse_g2", np.sqrt(r_tr["mse2"])),
                         ("val_rmse_g2", np.sqrt(r_va["mse2"])),
                         ("score_g1", score[0]), ("score_g2", score[1])]:
                history[k].append(float(v))

            marks = ["", ""]
            if epoch > ramp:          # val not comparable during the noise ramp
                improved = False
                if args.per_component_ckpt:
                    for k in range(2):
                        if score[k] < best_score[k]:
                            best_score[k], best_state[k] = score[k], eval_state
                            best_epochs[k], marks[k], improved = epoch, "*", True
                else:
                    joint = float(np.dot(score, comp_w) / sum(comp_w))
                    if joint < best_score[0]:
                        best_score = [joint, joint]
                        best_state = [eval_state, eval_state]
                        best_epochs = [epoch, epoch]
                        marks, improved = ["*", "*"], True
                stale = 0 if improved else stale + 1

            print(f"Epoch {epoch:3d}/{args.epochs} | loss {r_tr['loss']:.4f}/"
                  f"{r_va['loss']:.4f} | RMSE g+ {np.sqrt(r_tr['mse1']):.4f}/"
                  f"{np.sqrt(r_va['mse1']):.4f} g× {np.sqrt(r_tr['mse2']):.4f}/"
                  f"{np.sqrt(r_va['mse2']):.4f} | score g+ {score[0]:.4f}"
                  f"{marks[0]:1} g× {score[1]:.4f}{marks[1]:1} | "
                  f"noise {prep.noise_scale:.2f} | {time.time() - t0:.1f}s")

            if epoch > ramp and stale >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}: no val improvement "
                      f"in either component for {args.patience} epochs.")
                break

        if best_state[0] is None:
            raise RuntimeError("No checkpoint selected: --epochs must exceed "
                               "--noise_ramp_epochs.")
        final_state = eval_state
        torch.save({"state_final": final_state, "epoch": epoch,
                    "args": vars(args)}, out_dir / "final_model.pt")
        torch.save({"state_g_plus": best_state[0], "state_g_cross": best_state[1],
                    "best_epochs": best_epochs, "best_scores": best_score,
                    "args": vars(args), "roll_sign": roll_sign,
                    "z_stats": z_stats}, out_dir / "best_model.pt")
        print(f"\nBest val score: g+ {best_score[0]:.4f} (epoch {best_epochs[0]})"
              f"   g× {best_score[1]:.4f} (epoch {best_epochs[1]})")
        first_ok = ramp + 1
        if args.eval_weights == "best" and min(best_epochs) <= first_ok + 2:
            print(f"  [NOTE] a best epoch is within 3 epochs of the start "
                  f"(epoch {first_ok}): val never improved after that, so "
                  f"the selected weights are an almost untrained network and "
                  f"its predictions sit near 0 on every split. Use "
                  f"--eval_weights final to see what training produced.")
        if args.eval_weights == "final":
            print(f"  Using FINAL weights (epoch {epoch}) for all predictions, "
                  f"not the val-selected ones.")
            best_state = [final_state, final_state]
            best_epochs = [epoch, epoch]

        # ── Predictions: g+ from its best weights, g× from its own ──────────
        def predict(idx, tta):
            r0 = evaluate_state(model, best_state[0], data, idx, prep, criterion,
                                args.batch_size, parity, tta=tta)
            if best_state[1] is best_state[0]:
                return r0["preds"], r0["labels"], r0, r0["time"]
            r1 = evaluate_state(model, best_state[1], data, idx, prep, criterion,
                                args.batch_size, parity, tta=tta)
            p = np.stack([r0["preds"][:, 0], r1["preds"][:, 1]], axis=1)
            return p, r0["labels"], r0, r0["time"] + r1["time"]

        va_preds, va_labels, _, _ = predict(va, args.tta)
        te_preds_raw, te_labels, r_te0, te_time = predict(te, args.tta)
        # Same weights on the TRAINING galaxies (stored noise, eval mode, no
        # augmentation).  If the model memorised them, these predictions
        # follow the truth far better than on val/test.
        tr_preds_raw, tr_labels, _, _ = predict(tr, args.tta)

    # ── Calibration on val, applied to test ─────────────────────────────────
    cal, cal_info = fit_calibration(va_preds, va_labels, args.calibrate,
                                    args.cal_min_z)
    te_preds = apply_calibration(te_preds_raw, cal)
    va_preds_cal = apply_calibration(va_preds, cal)
    tr_preds = (apply_calibration(tr_preds_raw, cal)
                if tr_preds_raw is not None else None)

    print("\n── Test set evaluation ──────────────────────────────────────")
    if te_time == te_time:
        print(f"  preprocess + forward: {te_time * 1e3:.3f} ms/sample")
    met_tr = None
    if tr_preds_raw is not None:
        met_tr = shear_metrics(tr_preds_raw, tr_labels)
        print_metrics(met_tr, "TRAIN metrics, raw network output "
                              "(same weights, stored noise)")
        print("  (train r far above test r = memorisation; both ≈ 0 = the "
              "model found no signal even on galaxies it trained on)")
    met_raw = shear_metrics(te_preds_raw, te_labels)
    print_metrics(met_raw, "Test metrics, raw network output")
    if args.calibrate != "none":
        print(f"\n  Calibration fitted on val (min significance "
              f"{args.cal_min_z:.1f}σ):")
        for k, nm in enumerate(["g+", "g×"]):
            r_, z_ = cal_info[k]
            used = "used" if cal[k, 0] > 0 else "NOT significant -> predict 0"
            print(f"    {nm}: val r = {r_:+.3f} ({z_:+.1f}σ)  "
                  f"a = {cal[k, 0]:.3f}  b = {cal[k, 1]:+.5f}   [{used}]")
    met = shear_metrics(te_preds, te_labels)
    print_metrics(met, "Test metrics, calibrated (headline)")

    # ── Plots and outputs ────────────────────────────────────────────────────
    print("\nGenerating diagnostic plots …")
    if history is not None:
        plot_loss_curves(history, out_dir, ramp_end=ramp, best=best_epochs)
    plot_sets = [(va_preds_cal, va_labels, va, "val"),
                 (te_preds, te_labels, te, "test"),
                 (te_preds_raw, te_labels, te, "test_raw")]
    if tr_preds_raw is not None:
        plot_sets += [(tr_preds, tr_labels, tr, "train"),
                      (tr_preds_raw, tr_labels, tr, "train_raw")]
    for preds, labels, ix, name in plot_sets:
        plot_pred_vs_true(preds, labels, name, out_dir)
        if not name.endswith("_raw"):
            plot_binned_rmse(preds, labels, df.iloc[ix].reset_index(drop=True),
                             name, out_dir)

    keep = [c for c in ["src_row", "pop.id", "db_seed", "truth.z", "truth.cosi",
                        "truth.vel.vcirc", "snr_eff.line_total", "snr_eff.F158"]
            if c in df.columns]
    pred_df = df.iloc[te][keep].reset_index(drop=True)
    pred_df["g_plus_true"],  pred_df["g_cross_true"] = te_labels[:, 0], te_labels[:, 1]
    pred_df["g_plus_raw"],   pred_df["g_cross_raw"]  = te_preds_raw[:, 0], te_preds_raw[:, 1]
    pred_df["g_plus_pred"],  pred_df["g_cross_pred"] = te_preds[:, 0], te_preds[:, 1]
    pred_df["row"] = te
    pred_df["fold"] = args.fold if args.kfold > 1 else -1
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    g_te = df["pop.id"].to_numpy()[te] if "pop.id" in df.columns else None
    eb, ee = ensemble_response(te_preds, te_labels, g_te)
    print_ensemble(eb, ee, "Ensemble response, test (calibrated; bootstrap "
                           "over catalog galaxies)")
    eb_raw, ee_raw = ensemble_response(te_preds_raw, te_labels, g_te)
    print_ensemble(eb_raw, ee_raw, "Ensemble response, test (raw)")
    plot_binned_response(te_preds_raw, te_labels, out_dir / "binned_response_test_raw.png",
                         "test, raw")
    if history is not None:
        pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list)}
                     ).to_csv(out_dir / "history.csv", index_label="epoch0")

    summary = {
        "arch": args.arch, "width": args.width, "loss": args.loss,
        "component_weights": args.component_weights, "ema": args.ema,
        "select": args.select, "per_component_ckpt": args.per_component_ckpt,
        "calibrate": args.calibrate, "cal_min_z": args.cal_min_z,
        "val_r_g1": cal_info[0, 0], "val_r_g2": cal_info[1, 0],
        "cal_a_g1": cal[0, 0], "cal_b_g1": cal[0, 1],
        "cal_a_g2": cal[1, 0], "cal_b_g2": cal[1, 1],
        "snr_col": args.snr_col, "min_line_snr": float(args.min_line_snr),
        "noise_scale": float(prep.noise_scale), "roll_sign": roll_sign,
        "canonicalize": prep.canonicalize, "parity": parity,
        "vel_channels": prep.vel_channels, "cont_sub": prep.cont_sub, "cont_profile": prep.cont_profile,
        "angle_inputs": prep.angle_inputs, "kin_scalars": prep.kin_scalars,
        "feat_scalars": prep.feat_norm is not None,
        "kin_calibrate": prep.kin_cal is not None,
        "recenter_line": prep.recenter,
        "recenter_imaging": prep.recenter_imaging, "grism_mirror": prep.grism_mirror,
        "best_epoch_g1": best_epochs[0], "best_epoch_g2": best_epochs[1],
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }
    for m_, pre in [(met_raw, "test_raw"), (met, "test")] + (
            [(met_tr, "train_raw")] if met_tr is not None else []):
        for tag, short in [("g_plus", "g1"), ("g_cross", "g2")]:
            for k, v in m_[tag].items():
                summary[f"{pre}_{k}_{short}"] = float(v)
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    print(f"\nAll outputs saved to {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


def aggregate_kfold(args, fold_dirs):
    """Pool the out-of-fold test predictions of all folds."""
    out = Path(args.outdir)
    oof = pd.concat([pd.read_csv(d / "test_predictions.csv") for d in fold_dirs],
                    ignore_index=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    lab = oof[["g_plus_true", "g_cross_true"]].to_numpy()
    groups = oof["pop.id"].to_numpy() if "pop.id" in oof else None
    print("\n" + "═" * 79)
    print(f"K-FOLD SUMMARY: {len(oof)} out-of-fold predictions from "
          f"{len(fold_dirs)} folds")
    print("═" * 79)
    rows = []
    for name, cols in [("calibrated", ["g_plus_pred", "g_cross_pred"]),
                       ("raw", ["g_plus_raw", "g_cross_raw"])]:
        pr = oof[cols].to_numpy()
        base, err = ensemble_response(pr, lab, groups)
        print_ensemble(base, err, f"Pooled out-of-fold, {name}")
        print_metrics(shear_metrics(pr, lab), f"Pooled out-of-fold metrics, {name}")
        for k, comp in enumerate(["g1", "g2"]):
            rows.append({"preds": name, "component": comp, "m": base[k, 0],
                         "m_err": err[k, 0], "c": base[k, 1], "c_err": err[k, 1],
                         "r": base[k, 2], "r_err": err[k, 2]})
    # raw predictions of different folds have different scales; standardise
    # each fold before pooling for the binned-response plot
    pr = oof[["g_plus_raw", "g_cross_raw"]].to_numpy().copy()
    for f in oof["fold"].unique():
        m = (oof["fold"] == f).to_numpy()
        pr[m] = (pr[m] - pr[m].mean(0)) / (pr[m].std(0) + 1e-12) * lab.std(0)
    plot_binned_response(pr, lab, out / "binned_response_oof.png",
                         "out-of-fold, raw, per-fold standardised")
    print("\n  Per-fold correlations (raw):")
    for f in sorted(oof["fold"].unique()):
        m = (oof["fold"] == f).to_numpy()
        r = [np.corrcoef(lab[m, k], oof.loc[m, c])[0, 1]
             for k, c in enumerate(["g_plus_raw", "g_cross_raw"])]
        print(f"    fold {f}: n={m.sum():5d}  r(g+) {r[0]:+.3f}  r(g×) {r[1]:+.3f}")
    pd.DataFrame(rows).to_csv(out / "kfold_summary.csv", index=False)
    print(f"\nSaved oof_predictions.csv, kfold_summary.csv and "
          f"binned_response_oof.png in {out.resolve()}")


def main():
    import copy
    args = parse_args()
    if args.kfold == 2 or args.kfold < 0:
        raise SystemExit("--kfold needs at least 3 folds (one test, one "
                         "validation, the rest training); use 0 to disable")
    if args.kfold > 1 and args.fold < 0:
        fold_dirs = []
        for f in range(args.kfold):
            a = copy.deepcopy(args)
            a.fold = f
            a.outdir = str(Path(args.outdir) / f"fold{f}")
            print("\n" + "═" * 79 + f"\nFOLD {f + 1} / {args.kfold}\n" + "═" * 79)
            run_once(a)
            fold_dirs.append(Path(a.outdir))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        aggregate_kfold(args, fold_dirs)
    else:
        run_once(args)


if __name__ == "__main__":
    main()