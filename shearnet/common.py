"""Shared data loading, preprocessing and metrics for shearnet."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CHANNELS = ["bb_F129", "bb_F158", "gr_roll0", "gr_roll1", "gr_roll2", "gr_roll3"]
TARGETS = ["truth.g_plus", "truth.g_cross"]
G_SCALE = 0.1 / math.sqrt(3)  # std of U(-0.1, 0.1)

# Observable catalogue-side scalars (noisy measurements / priors the KL fit also uses).
# True values (pop.logm, pop.sigma0_kms, truth.*) are never used as inputs.
SCALAR_COLS = [
    ("fixed.z", False),                    # redshift, held fixed in the KL fit
    ("pop.prior_vcirc_mu_kms", True),      # Tully-Fisher prior mean
    ("pop.logm_obs", False),
    ("pop.f_line_obs_cgs", True),
    ("pop.flux_obs_f129_ujy", True),
    ("pop.flux_obs_f158_ujy", True),
]


def load(data_dir):
    d = Path(data_dir)
    t = pd.read_parquet(d / "table.parquet")
    noisy = np.load(d / "noisy.npy")
    clean = np.load(d / "clean.npy")
    var = np.load(d / "var.npy")
    sig = np.array([t[f"sigma_bkg.{c}"].iloc[0] for c in CHANNELS], np.float32)
    rolls = t[[f"roll_deg_{k}" for k in range(4)]].iloc[0].to_numpy(np.float32)
    return t, noisy, clean, var, sig, rolls


def scalars(t):
    cols = []
    for c, lg in SCALAR_COLS:
        v = t[c].to_numpy(np.float64)
        cols.append(np.log10(np.clip(v, 1e-30, None)) if lg else v)
    return np.column_stack(cols).astype(np.float32)


def aux_targets(t):
    """Physical auxiliary targets (scaled to O(1))."""
    th = t["truth.theta_int"].to_numpy()
    c = t["truth.cosi"].to_numpy()
    vsini = t["truth.vel.vcirc"].to_numpy() * np.sqrt(1 - c**2)
    return np.column_stack([
        (c - 0.5) / 0.26,
        np.cos(2 * th), np.sin(2 * th),
        (np.log10(vsini) - 2.0) / 0.25,
        t["truth.g1"].to_numpy() / G_SCALE, t["truth.g2"].to_numpy() / G_SCALE,
        (np.log10(t["truth.h_over_r"].to_numpy()) + 0.6) / 0.14,
    ]).astype(np.float32)


def group_split(t, seed=0, fracs=(0.7, 0.15, 0.15), fold=None, k=5):
    """Split rows by catalogue galaxy (pop.id) so test galaxies are unseen.

    With fold=None: random train/val/test by fracs.  With fold=f: galaxies are
    dealt into k folds; fold f is test, fold (f+1)%k is val, the rest train.
    """
    ids = t["pop.id"].to_numpy()
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    if fold is None:
        n1 = int(fracs[0] * len(perm))
        n2 = int((fracs[0] + fracs[1]) * len(perm))
        tr, va, te = perm[:n1], perm[n1:n2], perm[n2:]
    else:
        parts = np.array_split(perm, k)
        te, va = parts[fold], parts[(fold + 1) % k]
        tr = np.concatenate([parts[j] for j in range(k) if j not in (fold, (fold + 1) % k)])
    m = lambda s: np.where(np.isin(ids, s))[0]
    return m(tr), m(va), m(te)


def _rot_grid(angles_deg, n, device):
    a = torch.as_tensor(np.radians(angles_deg), dtype=torch.float32, device=device)
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.zeros(len(a), 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    return F.affine_grid(theta, (len(a), 1, n, n), align_corners=False)


class Preproc:
    """Stamps (B,6,32,32) in physical units -> network input (B,C,H,W).

    Channels: 2 imaging + 4 raw grism + 4 grism rotated back to the sky frame
    (roll j is the sky rotated by roll_j; rotating by -roll_j puts the galaxy
    back at its sky orientation, with dispersion now along direction roll_j).
    All in units of the background noise sigma, then asinh-compressed.
    """

    def __init__(self, sig, rolls, crop=32, align=True, soft=8.0, device="cpu"):
        self.sig = torch.as_tensor(sig).view(1, 6, 1, 1)
        self.crop = crop
        self.align = align
        self.soft = soft
        self.grid = _rot_grid(-np.asarray(rolls), 32, device) if align else None
        self.n_ch = 6 + (4 if align else 0)

    def __call__(self, x):
        x = x / self.sig
        chans = [x]
        if self.align:
            B = x.shape[0]
            g = x[:, 2:6].reshape(B * 4, 1, 32, 32)
            grid = self.grid.repeat(B, 1, 1, 1)
            al = F.grid_sample(g, grid, mode="bilinear", padding_mode="zeros",
                               align_corners=False).reshape(B, 4, 32, 32)
            chans.append(al)
        z = torch.cat(chans, 1)
        z = self.soft * torch.asinh(z / self.soft) / self.soft
        if self.crop < 32:
            o = (32 - self.crop) // 2
            z = z[:, :, o:o + self.crop, o:o + self.crop]
        return z


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def calib_metrics(pred, true):
    """Shear-pipeline metrics for one component.

    r      : Pearson correlation
    rmse   : root mean square error of the raw prediction
    m, c   : multiplicative / additive bias of pred = (1+m) g + c
    sig_eff: per-galaxy noise of the response-corrected (unbiased) estimator,
             std(pred - (1+m) g - c) / (1+m).  This is the number that sets the
             statistical power of the shear measurement: N galaxies give a mean
             shear error of sig_eff / sqrt(N).  For comparison, conventional
             weak lensing has a per-component shape noise of ~0.26.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    A = np.column_stack([true, np.ones_like(true)])
    (a, b), *_ = np.linalg.lstsq(A, pred, rcond=None)
    res = pred - a * true - b
    n = len(true)
    a_err = res.std() / (true.std() * math.sqrt(n))
    return dict(r=float(np.corrcoef(pred, true)[0, 1]),
                rmse=float(np.sqrt(np.mean((pred - true) ** 2))),
                m=float(a - 1), m_err=float(a_err), c=float(b),
                sig_eff=float(res.std() / a) if a > 0 else float("inf"))


def weighted_sig_eff(pred, true, sigma):
    """sig_eff of an inverse-variance-weighted ensemble estimate.

    With per-galaxy weights w = R^2 / sigma^2, the mean-shear error over N
    galaxies is 1/sqrt(sum w); this returns the equivalent per-galaxy noise
    sqrt(N / sum w) using the linear response R fitted on the same rows.
    """
    pred, true, sigma = map(lambda v: np.asarray(v, float), (pred, true, sigma))
    w = 1.0 / sigma**2
    A = np.column_stack([true, np.ones_like(true)]) * np.sqrt(w)[:, None]
    (a, b), *_ = np.linalg.lstsq(A, pred * np.sqrt(w), rcond=None)
    res = pred - a * true - b
    # error of the weighted response-corrected mean per galaxy-equivalent
    var_mean = np.sum(w**2 * res**2) / np.sum(w) ** 2
    return float(math.sqrt(var_mean * len(true)) / a)
