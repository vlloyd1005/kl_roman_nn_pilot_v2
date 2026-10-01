#!/usr/bin/env python3
"""
train_kl_grism.py  (v2)
=======================
Shear regression for Roman kinematic-lensing mock stamps (imaging + slitless
grism).  Adapted from train_kl_model.py.

Inputs per galaxy (all 32 x 32, 0.11 arcsec/px)
  imaging : bb_F129, bb_F158                     sky orientation
  grism   : gr_roll0..3 at rolls 0, 10, 167, 177  sky rotated by roll_j,
                                                  dispersed along columns
Targets   : truth.g_plus, truth.g_cross (galaxy frame)

What v2 adds (every item has an on/off flag)
-------------------------------------------
Noise control     Inputs are built as clean + noise_scale * noise, so the
                  noise level is a setting rather than a row property.
                  --noise_scale 0.25 makes every galaxy ~4x higher S/N;
                  --noise_ramp_epochs ramps from --noise_start to
                  --noise_scale (curriculum).  Eval uses the stored noise
                  realisation scaled the same way (exact for Gaussian noise).
Canonical frame   Every scene is rotated so the photometric major axis
                  (Gaussian-weighted second moments of F129+F158) lies along
                  x.  Observer-legal, and the galaxy-frame targets do not
                  change.  Augmentation becomes a random 180-degree turn; TTA
                  averages 0 and 180 degrees.
Parity symmetry   Each input is also evaluated mirrored (x -> -x, dispersion
                  directions mirrored too).  g+ = mean of the two, g× = half
                  the difference, so g× is exactly parity-odd.
Velocity channels From each roughly opposed roll pair (a, b), the aligned
                  stamps satisfy D = G_a - G_b ≈ -s v(x) u·∇I with
                  u = d_a - d_b, so  v̂ = -D (u·∇S) / ((u·∇S)^2 + eps)  with
                  S the smoothed pair mean.  v̂ is a velocity map in pixels of
                  line shift; a reliability map J²/(J²+eps) is passed too.
                  --vel_decomp adds even/odd parts under 180° rotation of the
                  reliability-weighted mean v̂ (your old v_asym / v_sym).
Auxiliary heads   Rotation- and parity-invariant truths (cos i, vcirc,
                  vcirc·sin i by default) predicted from the same features,
                  weighted by --aux_weight.  Their test R² is a diagnostic:
                  if the network cannot recover cos i, it cannot do shear.
Architectures     --arch effnet    pretrained EfficientNet-B0 two-stream (v1)
                  --arch small     ~0.5M-parameter two-stream CNN from scratch
                  --arch features  hand-built moment + kinematic-fit features
                                   -> gradient-boosted trees (fast baseline;
                                   use with --canonicalize)

Preprocessing order (GPU, per batch)
  noise -> /sigma_bkg -> [photometric PA] -> one rotation per channel
  (scene angle + roll alignment for grism) -> aperture mask -> optional mirror
  -> pair differences, velocity channels -> asinh stretch.

Usage
-----
  python train_kl_grism.py --parquet pilot_v2_all.parquet \
      --meta meta_pilot_v2_all.json --outdir run01 --min_line_snr 5

Dependencies: torch timm numpy pandas pyarrow scikit-learn matplotlib
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


def build_aux_targets(df: pd.DataFrame, spec: str):
    """Rotation- and parity-invariant auxiliary truths.  'vsini' is derived
    as vcirc * sqrt(1 - cos^2 i)."""
    names, cols = [], []
    for name in [s.strip() for s in spec.split(",") if s.strip()]:
        if name == "vsini":
            if {"truth.vel.vcirc", "truth.cosi"} <= set(df.columns):
                v = df["truth.vel.vcirc"].to_numpy(np.float64)
                c = df["truth.cosi"].to_numpy(np.float64)
                cols.append(v * np.sqrt(np.clip(1 - c ** 2, 0, 1)))
                names.append("vsini")
            else:
                print("[WARN] aux 'vsini' needs truth.vel.vcirc and truth.cosi")
        elif name in df.columns:
            cols.append(df[name].to_numpy(np.float64))
            names.append(name)
        else:
            print(f"[WARN] aux column '{name}' not found; skipped")
    if not cols:
        return [], None
    return names, np.stack(cols, axis=1).astype(np.float32)


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
                 npix, device):
        self.roll_sign      = float(roll_sign)
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
    def feature_names(self):
        names = ["flux_F129", "flux_F158", "img_e1", "img_e2", "img_T"]
        for i, j in self.pairs:
            t = f"p{i}{j}"
            names += [f"{t}_kin_a", f"{t}_kin_b", f"{t}_kin_c",
                      f"{t}_flux", f"{t}_e1", f"{t}_e2"]
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
    """Separate g+ and g× heads plus an optional auxiliary head."""

    def __init__(self, feat_dim, n_aux=0, hidden=256, dropout=0.3):
        super().__init__()

        def mlp(n_out):
            return nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, n_out))
        self.g1, self.g2 = mlp(1), mlp(1)
        self.aux = mlp(n_aux) if n_aux > 0 else None

    def forward(self, f):
        shear = torch.cat([self.g1(f), self.g2(f)], dim=1)
        return shear, (self.aux(f) if self.aux is not None else None)


class TwoStreamGrismNet(nn.Module):
    """Pretrained EfficientNet-B0 photo stem (stride-1 stem for 32 px stamps)
    + light grism CNN, fused at 16x16, shared EfficientNet body."""

    def __init__(self, n_img=2, n_grism=6, n_scalar=0, n_aux=0, pretrained=True,
                 freeze_stem=False, stem_stride=1, drop_stream="none"):
        super().__init__()
        import timm
        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                     in_chans=n_img)
        backbone.conv_stem.stride = (stem_stride, stem_stride)
        self.photo_stem = nn.Sequential(
            backbone.conv_stem, backbone.bn1, backbone.blocks[0], backbone.blocks[1])
        if freeze_stem:
            for p in self.photo_stem.parameters():
                p.requires_grad = False

        n_down = 1 if stem_stride == 1 else 2
        layers = [conv_bn_act(n_grism, 32), conv_bn_act(32, 32), conv_bn_act(32, 48)]
        cin = 48
        for k in range(n_down):
            cout = 24 if k == n_down - 1 else 48
            layers.append(conv_bn_act(cin, cout, stride=2))
            cin = cout
        self.grism_stream = nn.Sequential(*layers)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=1, bias=False), nn.BatchNorm2d(24), nn.SiLU())
        self.shared_body = nn.Sequential(
            backbone.blocks[2], backbone.blocks[3], backbone.blocks[4],
            backbone.blocks[5], backbone.blocks[6], backbone.conv_head, backbone.bn2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = Heads(1280 + n_scalar, n_aux)
        self.drop_stream = drop_stream

    def forward(self, img, gr, scal=None):
        f_img, f_gr = self.photo_stem(img), self.grism_stream(gr)
        if self.drop_stream == "grism":
            f_gr = torch.zeros_like(f_gr)
        elif self.drop_stream == "imaging":
            f_img = torch.zeros_like(f_img)
        fused = self.fusion_proj(torch.cat([f_img, f_gr], dim=1))
        f = self.pool(self.shared_body(fused)).flatten(1)
        if scal is not None:
            f = torch.cat([f, scal], dim=1)
        return self.heads(f)


class SmallTwoStream(nn.Module):
    """~0.5M-parameter two-stream CNN trained from scratch.  Both streams keep
    full resolution for their first layers and meet at 16x16."""

    def __init__(self, n_img=2, n_grism=6, n_scalar=0, n_aux=0,
                 drop_stream="none"):
        super().__init__()
        self.photo = nn.Sequential(conv_bn_act(n_img, 32), conv_bn_act(32, 32),
                                   conv_bn_act(32, 48, stride=2))
        self.grism = nn.Sequential(conv_bn_act(n_grism, 32), conv_bn_act(32, 32),
                                   conv_bn_act(32, 48), conv_bn_act(48, 48, stride=2))
        self.fuse = nn.Sequential(nn.Conv2d(96, 64, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.SiLU())
        self.body = nn.Sequential(conv_bn_act(64, 64), conv_bn_act(64, 96, stride=2),
                                  conv_bn_act(96, 96), conv_bn_act(96, 128, stride=2),
                                  conv_bn_act(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = Heads(128 + n_scalar, n_aux, hidden=128, dropout=0.2)
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
# Train / eval
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceData:
    """All stamps live on the device; batches are gathered by index."""

    def __init__(self, noisy, clean, var, sigma, rolls, y, scal, aux, device):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.noisy, self.clean, self.var = t(noisy), t(clean), t(var)
        self.sigma, self.rolls, self.y   = t(sigma), t(rolls), t(y)
        self.scal    = t(scal) if scal is not None else None
        self.aux     = t(aux) if aux is not None else None
        self.device  = torch.device(device)
        self.is_cuda = self.device.type == "cuda"

    def batch(self, b):
        return (self.noisy[b], self.clean[b], self.var[b], self.sigma[b],
                self.rolls[b])


def tta_alphas(prep, tta: int):
    if tta <= 1:
        return [None]
    if prep.canonicalize:
        return [0.0, 180.0]
    return [360.0 * k / tta for k in range(tta)]


def model_forward(model, prep, state, scal, parity: bool):
    """Forward pass, optionally symmetrised under parity."""
    img, gr = prep.build(state)
    shear, aux = model(img, gr, scal)
    if parity:
        img_m, gr_m = prep.build(prep.flip(state))
        shear_m, aux_m = model(img_m, gr_m, scal)
        shear = torch.stack([(shear[:, 0] + shear_m[:, 0]) / 2,
                             (shear[:, 1] - shear_m[:, 1]) / 2], dim=1)
        if aux is not None:
            aux = (aux + aux_m) / 2
    return shear, aux


def run_epoch(model, data, idx, prep, criterion, optimizer, bs, training: bool,
              parity: bool, aux_weight: float = 0.0, tta: int = 1):
    model.train(training)
    idx = np.random.permutation(idx) if training else np.asarray(idx)
    alphas = [None] if training else tta_alphas(prep, tta)
    use_aux = data.aux is not None and aux_weight > 0

    total, n, t_fwd = 0.0, 0, 0.0
    preds, labels, aux_preds = [], [], []
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
            out, out_aux = 0.0, 0.0
            for a in alphas:
                state = prep.prepare(*data.batch(b), training, alpha=a)
                sh, ax = model_forward(model, prep, state, sc, parity)
                out = out + sh
                if ax is not None:
                    out_aux = out_aux + ax
            out = out / len(alphas)
            if not training and data.is_cuda:
                torch.cuda.synchronize(data.device)
            t_fwd += time.perf_counter() - t0

            loss = criterion(out, y)
            if use_aux:
                out_aux = out_aux / len(alphas)
                loss = loss + aux_weight * F.mse_loss(out_aux, data.aux[b])
                aux_preds.append(out_aux.detach().cpu())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item() * len(b)
            n     += len(b)
            preds.append(out.detach().cpu() * G_SCALE)
            labels.append(data.y[b].cpu())

    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    aux_p  = torch.cat(aux_preds).numpy() if aux_preds else None
    mse    = ((preds - labels) ** 2).mean(axis=0)
    return dict(loss=total / n, mse1=float(mse[0]), mse2=float(mse[1]),
                preds=preds, labels=labels, aux=aux_p,
                time=None if training else t_fwd / n)


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


def aux_r2(aux_pred, aux_true, names):
    out = {}
    for k, nm in enumerate(names):
        t, p = aux_true[:, k], aux_pred[:, k]
        out[nm] = float(1 - np.mean((p - t) ** 2) / max(np.var(t), 1e-12))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(history: dict, out_dir: Path, ramp_end: int = 0):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train", color="#2E86AB")
    axes[0].plot(history["val_loss"],   label="Val",   color="#E84855")
    axes[0].set_ylabel("loss (g / 0.03 [+ aux])"); axes[0].set_title("Total loss")
    for ax, key, name in [(axes[1], "rmse_g1", "g+"), (axes[2], "rmse_g2", "g×")]:
        ax.plot(history[f"train_{key}"], label=f"Train {name}", color="#2E86AB")
        ax.plot(history[f"val_{key}"],   label=f"Val {name}",   color="#E84855")
        ax.axhline(history[f"null_{key}"], color="k", ls="--", lw=1,
                   label="predict 0")
        ax.set_ylabel("RMSE"); ax.set_title(f"{name} RMSE")
    for ax in axes:
        if ramp_end:
            ax.axvline(ramp_end - 1, color="grey", ls=":", lw=1)
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: loss_curves.png")


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

def parse_pairs(s: str):
    if s.lower() in ("", "none"):
        return []
    return [tuple(int(v) for v in tok.split("-")) for tok in s.split(",")]


def parse_args():
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="KL shear regressor for Roman "
                                            "imaging + grism stamps (v2).")
    p.add_argument("--parquet", required=True)
    p.add_argument("--meta", default=None, help="meta_pilot_v2_all.json")
    p.add_argument("--outdir", default="./kl_grism_output")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None,
                   help="default 3e-4 (effnet) or 1e-3 (small)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25,
                   help="Early-stopping patience on val loss (counted after "
                        "any noise ramp)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
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
    p.add_argument("--roll_sign", choices=["auto", "1", "-1"], default="auto")
    p.add_argument("--rolls", type=float, nargs=4, default=None,
                   help="Override roll angles (deg) for gr_roll0..3")
    p.add_argument("--pairs", default="0-3,1-2",
                   help="Roughly opposed grism pairs, e.g. '0-3,1-2' = "
                        "(0°,177°),(10°,167°); 'none' to disable")
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
    p.add_argument("--moment_sigma", type=float, default=3.0,
                   help="Gaussian weight width (px) for moments / position angle")
    # velocity channels
    p.add_argument("--vel_channels", action=B, default=True,
                   help="Add velocity-map estimates from roll-pair differences")
    p.add_argument("--vel_decomp", action=B, default=False,
                   help="Add even/odd (180° rotation) parts of the mean v̂")
    p.add_argument("--vel_eps", type=float, default=0.5,
                   help="Regulariser for v̂, relative to the mean J²")
    p.add_argument("--vel_clip", type=float, default=2.0,
                   help="Clip v̂ to ± this many pixels of line shift")
    p.add_argument("--vel_smooth", type=float, default=1.0,
                   help="Gaussian smoothing (px) of the pair mean before ∇")
    p.add_argument("--asinh_scale", type=float, default=2.0)
    # model
    p.add_argument("--arch", choices=["effnet", "small", "features"],
                   default="effnet")
    p.add_argument("--pretrain", action=B, default=True,
                   help="ImageNet weights for --arch effnet")
    p.add_argument("--freeze_stem", action="store_true")
    p.add_argument("--stem_stride", type=int, choices=[1, 2], default=1)
    p.add_argument("--use_z", action="store_true",
                   help="Append redshift as a scalar input")
    p.add_argument("--drop_stream", choices=["none", "grism", "imaging"],
                   default="none", help="Ablation: zero one stream's features")
    p.add_argument("--aux", default="truth.cosi,truth.vel.vcirc,vsini",
                   help="Comma-separated auxiliary targets ('none' to disable)")
    p.add_argument("--aux_weight", type=float, default=0.3)
    # feature baseline
    p.add_argument("--feat_draws", type=int, default=4,
                   help="Noise draws per training galaxy (--arch features)")
    p.add_argument("--feat_max_iter", type=int, default=600)
    p.add_argument("--frac_eps", type=float, default=0.035)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.lr is None:
        args.lr = 1e-3 if args.arch == "small" else 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df, meta, noisy, clean, var, sigma = load_table(args.parquet, args.meta)
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
    tr, va, te = split_by_group(groups, args.train_frac, args.val_frac, args.seed)
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

    aux_names, aux, aux_stats = [], None, None
    if args.arch != "features" and args.aux.lower() != "none" and args.aux_weight > 0:
        aux_names, aux_raw = build_aux_targets(df, args.aux)
        if aux_raw is not None:
            mu, sd = aux_raw[tr].mean(0), aux_raw[tr].std(0) + 1e-8
            aux, aux_stats = ((aux_raw - mu) / sd).astype(np.float32), (mu, sd)
            print(f"Auxiliary targets (weight {args.aux_weight}): {aux_names}")

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
    parity = args.parity and align

    prep = GPUPreprocessor(
        roll_sign, align, pairs, args.asinh_scale,
        resample_noise=args.resample_noise, rot_aug=args.rot_aug,
        canonicalize=args.canonicalize, vel_channels=args.vel_channels,
        vel_decomp=args.vel_decomp, vel_eps=args.vel_eps, vel_clip=args.vel_clip,
        vel_smooth=args.vel_smooth, moment_sigma=args.moment_sigma,
        npix=noisy.shape[-1], device=device)
    ramp = args.noise_ramp_epochs if args.arch != "features" else 0
    prep.noise_scale = args.noise_start if ramp > 0 else args.noise_scale

    print(f"Grism inputs: {prep.channel_names()[2:]}")
    print(f"Canonical frame: {prep.canonicalize}   parity: {parity}   "
          f"rotation aug: {prep.rot_aug}   TTA: {tta_alphas(prep, args.tta)}")
    print(f"Noise: scale {args.noise_scale}"
          + (f" (ramp from {args.noise_start} over {ramp} epochs)" if ramp else "")
          + f"   resample: {prep.resample_noise}")

    data = DeviceData(noisy, clean, var, sigma, rolls, y, scal, aux, device)
    del noisy, clean, var

    print("\nSaving an input example …")
    plot_input_example(data, prep, va[:4], df, out_dir)

    criterion = nn.SmoothL1Loss()
    history, ckpt, best_val = None, None, float("nan")
    aux_test = None

    if args.arch == "features":
        met, va_preds, te_preds = run_feature_baseline(
            args, data, prep, tr, va, te, df, y, out_dir)
        te_labels, va_labels = y[te], y[va]
        te_time = float("nan")
    else:
        # ── Model ────────────────────────────────────────────────────────────
        kw = dict(n_img=len(IMG_CH), n_grism=prep.n_grism_in,
                  n_scalar=0 if scal is None else scal.shape[1],
                  n_aux=len(aux_names), drop_stream=args.drop_stream)
        if args.arch == "effnet":
            model = TwoStreamGrismNet(pretrained=args.pretrain,
                                      freeze_stem=args.freeze_stem,
                                      stem_stride=args.stem_stride, **kw)
        else:
            model = SmallTwoStream(**kw)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.arch}   trainable parameters: {n_params:,}")

        if args.arch == "effnet" and args.pretrain:
            stem_ids = {id(p) for p in model.photo_stem.parameters()}
            groups_ = [
                {"params": [p for p in model.parameters() if id(p) in stem_ids],
                 "lr": args.lr * 0.1},
                {"params": [p for p in model.parameters() if id(p) not in stem_ids],
                 "lr": args.lr}]
        else:
            groups_ = [{"params": list(model.parameters()), "lr": args.lr}]
        optimizer = optim.AdamW(groups_, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        # ── Training loop ────────────────────────────────────────────────────
        history = {k: [] for k in ["train_loss", "val_loss", "train_rmse_g1",
                                   "val_rmse_g1", "train_rmse_g2", "val_rmse_g2"]}
        history["null_rmse_g1"] = float(np.sqrt(np.mean(y[va, 0] ** 2)))
        history["null_rmse_g2"] = float(np.sqrt(np.mean(y[va, 1] ** 2)))
        print(f"\nVal RMSE of predicting 0: g+ {history['null_rmse_g1']:.4f}  "
              f"g× {history['null_rmse_g2']:.4f}  (the number to beat)")

        best_val, patience, best_path = float("inf"), 0, out_dir / "best_model.pt"
        ep_kw = dict(parity=parity, aux_weight=args.aux_weight)
        print(f"\nTraining for up to {args.epochs} epochs "
              f"(patience={args.patience}) …\n")
        for epoch in range(1, args.epochs + 1):
            if ramp > 0:
                frac = min(1.0, (epoch - 1) / ramp)
                prep.noise_scale = (args.noise_start
                                    + (args.noise_scale - args.noise_start) * frac)
            t0 = time.time()
            r_tr = run_epoch(model, data, tr, prep, criterion, optimizer,
                             args.batch_size, training=True, **ep_kw)
            r_va = run_epoch(model, data, va, prep, criterion, optimizer,
                             args.batch_size, training=False, **ep_kw)
            scheduler.step()
            for k, v in [("train_loss", r_tr["loss"]), ("val_loss", r_va["loss"]),
                         ("train_rmse_g1", np.sqrt(r_tr["mse1"])),
                         ("val_rmse_g1", np.sqrt(r_va["mse1"])),
                         ("train_rmse_g2", np.sqrt(r_tr["mse2"])),
                         ("val_rmse_g2", np.sqrt(r_va["mse2"]))]:
                history[k].append(float(v))
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {r_tr['loss']:.4f}/"
                  f"{r_va['loss']:.4f} | RMSE g+ {np.sqrt(r_tr['mse1']):.4f}/"
                  f"{np.sqrt(r_va['mse1']):.4f} | g× {np.sqrt(r_tr['mse2']):.4f}/"
                  f"{np.sqrt(r_va['mse2']):.4f} | noise {prep.noise_scale:.2f} | "
                  f"lr {scheduler.get_last_lr()[-1]:.1e} | {time.time() - t0:.1f}s")

            if epoch <= ramp:          # val loss not comparable during the ramp
                continue
            if r_va["loss"] < best_val:
                best_val, patience = r_va["loss"], 0
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "val_loss": r_va["loss"], "args": vars(args),
                            "roll_sign": roll_sign, "z_stats": z_stats,
                            "aux_names": aux_names, "aux_stats": aux_stats},
                           best_path)
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        if not best_path.exists():
            raise RuntimeError("No checkpoint saved: --epochs must exceed "
                               "--noise_ramp_epochs.")
        print(f"\nLoading best checkpoint (val loss = {best_val:.4f}) …")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # ── Test ─────────────────────────────────────────────────────────────
        print("\n── Test set evaluation ──────────────────────────────────────")
        r_te = run_epoch(model, data, te, prep, criterion, optimizer,
                         args.batch_size, training=False, tta=args.tta, **ep_kw)
        te_preds, te_labels, te_time = r_te["preds"], r_te["labels"], r_te["time"]
        print(f"  preprocess + forward: {te_time * 1e3:.3f} ms/sample")
        met = shear_metrics(te_preds, te_labels)
        print_metrics(met, f"Test metrics (noise_scale={prep.noise_scale:.2f})")
        if r_te["aux"] is not None:
            aux_test = aux_r2(r_te["aux"], aux[te], aux_names)
            print("\n  Auxiliary targets, test R² (1 = perfect, 0 = predicts mean):")
            for nm, v in aux_test.items():
                print(f"    {nm:>18}: {v:.3f}")

        r_va = run_epoch(model, data, va, prep, criterion, optimizer,
                         args.batch_size, training=False, tta=args.tta, **ep_kw)
        va_preds, va_labels = r_va["preds"], r_va["labels"]

    # ── Plots and outputs ────────────────────────────────────────────────────
    print("\nGenerating diagnostic plots …")
    if history is not None:
        plot_loss_curves(history, out_dir, ramp_end=ramp)
    for preds, labels, ix, name in [(va_preds, va_labels, va, "val"),
                                    (te_preds, te_labels, te, "test")]:
        plot_pred_vs_true(preds, labels, name, out_dir)
        plot_binned_rmse(preds, labels, df.iloc[ix].reset_index(drop=True),
                         name, out_dir)

    keep = [c for c in ["pop.id", "db_seed", "truth.z", "truth.cosi",
                        "truth.vel.vcirc", "snr_eff.line_total", "snr_eff.F158"]
            if c in df.columns]
    pred_df = df.iloc[te][keep].reset_index(drop=True)
    pred_df["g_plus_true"],  pred_df["g_plus_pred"]  = te_labels[:, 0], te_preds[:, 0]
    pred_df["g_cross_true"], pred_df["g_cross_pred"] = te_labels[:, 1], te_preds[:, 1]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {
        "arch": args.arch, "snr_col": args.snr_col,
        "min_line_snr": float(args.min_line_snr),
        "noise_scale": float(prep.noise_scale), "roll_sign": roll_sign,
        "align_grism": align, "pairs": args.pairs,
        "canonicalize": prep.canonicalize, "parity": parity,
        "vel_channels": prep.vel_channels, "vel_decomp": prep.vel_decomp,
        "aux": ",".join(aux_names), "aux_weight": args.aux_weight,
        "resample_noise": prep.resample_noise, "rot_aug": prep.rot_aug,
        "tta": args.tta, "use_z": args.use_z, "drop_stream": args.drop_stream,
        "best_epoch": int(ckpt["epoch"]) if ckpt else -1,
        "best_val_loss": float(best_val),
        "avg_eval_time_ms": float(te_time * 1e3) if te_time == te_time else np.nan,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }
    for tag, short in [("g_plus", "g1"), ("g_cross", "g2")]:
        for k, v in met[tag].items():
            summary[f"test_{k}_{short}"] = float(v)
    if aux_test:
        for nm, v in aux_test.items():
            summary[f"aux_r2_{nm}"] = v
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    print(f"\nAll outputs saved to {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()