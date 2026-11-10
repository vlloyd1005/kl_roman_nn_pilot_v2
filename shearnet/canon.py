"""Canonicalisation: rotate every stamp so the measured photometric major axis
lies along +x.  Array convention: x = column index, y = row index (GalSim's,
plotted with origin='lower'); angles counter-clockwise from +x.  Grism stamp j
shows the sky rotated by -roll_j (validated against the noiseless stamps)."""
import math

import torch
import torch.nn.functional as F


def rotate(x, beta):
    """Rotate images x (B,C,H,W) by beta (B,) radians counter-clockwise about the centre."""
    c, s = torch.cos(beta), torch.sin(beta)
    th = torch.zeros(x.shape[0], 2, 3, dtype=x.dtype)
    th[:, 0, 0], th[:, 0, 1] = c, s
    th[:, 1, 0], th[:, 1, 1] = -s, c
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def moments_pa(img, win=2.5, iters=2):
    """Photometric position angle (B,) and ellipticity from windowed moments of img (B,H,W)."""
    B, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    cx = torch.full((B,), (W - 1) / 2, dtype=img.dtype)
    cy = torch.full((B,), (H - 1) / 2, dtype=img.dtype)
    for _ in range(iters):
        dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
        w = torch.exp(-0.5 * (dx**2 + dy**2) / win**2) * img
        tot = w.sum((1, 2)).clamp_min(1e-6)
        cx = cx + (w * dx).sum((1, 2)) / tot
        cy = cy + (w * dy).sum((1, 2)) / tot
        cx, cy = cx.clamp(10, W - 11), cy.clamp(10, H - 11)
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    w = torch.exp(-0.5 * (dx**2 + dy**2) / (1.5 * win) ** 2) * img
    tot = w.sum((1, 2)).clamp_min(1e-6)
    qxx = (w * dx * dx).sum((1, 2)) / tot
    qyy = (w * dy * dy).sum((1, 2)) / tot
    qxy = (w * dx * dy).sum((1, 2)) / tot
    pa = 0.5 * torch.atan2(2 * qxy, qxx - qyy)
    e = torch.sqrt((qxx - qyy) ** 2 + 4 * qxy**2) / (qxx + qyy).clamp_min(1e-6)
    return pa, e, cx, cy


class CanonPreproc:
    """(B,6,32,32) physical stamps -> (B,10,crop,crop) canonical input + (B,k) angle scalars.

    Channels: canonical F129, F158, 4 canonical grism, 4 raw grism.  All in
    units of background sigma, asinh-compressed.
    """

    def __init__(self, sig, rolls, crop=24, soft=8.0, canon=True, kin=False):
        self.sig = torch.as_tensor(sig, dtype=torch.float32).view(1, 6, 1, 1)
        self.rolls_deg = list(rolls)
        self.rolls = torch.as_tensor([math.radians(r) for r in rolls], dtype=torch.float32)
        self.crop, self.soft, self.canon, self.kin = crop, soft, canon, kin
        self.n_ch = 12 if kin else 10
        self.n_ang = 5 + (10 if kin else 0)

    def __call__(self, x):
        B = x.shape[0]
        z = x / self.sig
        img = z[:, 0] + z[:, 1]
        pa, e, _, _ = moments_pa(img.clamp_min(0))
        if not self.canon:
            pa = torch.zeros_like(pa)
        im = rotate(z[:, :2], -pa)
        # grism j shows the galaxy at PA (pa - roll_j): rotate by -(pa - roll_j)
        graw = z[:, 2:6]
        if self.kin:
            from kinfeat import continuum_subtract, kinematic_features
            kf = kinematic_features(z, self.rolls_deg)
            graw = continuum_subtract(graw)
        g = graw.reshape(B * 4, 1, 32, 32)
        ang = (-(pa[:, None] - self.rolls[None])).reshape(-1)
        gc = rotate(g, ang).reshape(B, 4, 32, 32)
        chans = [im, gc, z[:, 2:6]]
        if self.kin:
            # opposite-dispersion pairs: the velocity dipole survives, the continuum cancels
            chans.append(torch.stack([gc[:, 0] - gc[:, 3], gc[:, 1] - gc[:, 2]], 1))
        out = torch.cat(chans, 1)
        out = self.soft * torch.asinh(out / self.soft) / self.soft
        if self.crop < 32:
            o = (32 - self.crop) // 2
            out = out[:, :, o:o + self.crop, o:o + self.crop]
        # dispersion direction in the canonical frame is (roll_0 - pa)
        a = self.rolls[0] - pa
        ang_feats = torch.stack([torch.cos(a), torch.sin(a), torch.cos(2 * a), torch.sin(2 * a), e], 1)
        if self.kin:
            kf = torch.sign(kf) * torch.log1p(kf.abs())          # tame heavy tails
            ang_feats = torch.cat([ang_feats, kf], 1)
        return out, ang_feats
