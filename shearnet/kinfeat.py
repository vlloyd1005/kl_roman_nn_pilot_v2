"""Explicit kinematic summary statistics from the grism stamps (batched torch).

A line-of-sight velocity gradient G (v = G . x) displaces Halpha emission along
roll j's dispersion direction d_j by s*v, which to first order changes the
second-moment tensor of the (sky-aligned) line image to
    Q_j = Q + s (d_j h^T + h d_j^T),   h = Q G.
Rolls 0/10 and 167/177 have (nearly) opposite d_j, so the four rolls determine
Q (3 numbers) and h (2 numbers) by linear least squares.  G = Q^-1 h / s gives
the kinematic axis; its angle to the photometric major axis is the observable
that carries g_x (a sheared disk's kinematic and photometric axes are not
aligned), and |G| together with the Tully-Fisher prior carries g_+.

All moments are Gaussian-windowed about each image's own centroid.
"""
import math

import torch

from canon import moments_pa, rotate


def _moments(img, win):
    B, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    cx = torch.full((B,), (W - 1) / 2, dtype=img.dtype)
    cy = torch.full((B,), (H - 1) / 2, dtype=img.dtype)
    for _ in range(3):
        dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
        w = torch.exp(-0.5 * (dx**2 + dy**2) / win**2) * img
        tot = w.sum((1, 2)).clamp_min(1e-6)
        cx = (cx + (w * dx).sum((1, 2)) / tot).clamp(8, W - 9)
        cy = (cy + (w * dy).sum((1, 2)) / tot).clamp(8, H - 9)
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    w = torch.exp(-0.5 * (dx**2 + dy**2) / win**2) * img
    tot = w.sum((1, 2)).clamp_min(1e-6)
    return ((w * dx * dx).sum((1, 2)) / tot, (w * dy * dy).sum((1, 2)) / tot,
            (w * dx * dy).sum((1, 2)) / tot, tot, cx, cy)


def continuum_subtract(g, edge=5):
    """Grism stamps (B,4,H,W): the continuum is ~constant along each row
    (dispersion axis); estimate it from the outer columns and subtract."""
    side = torch.cat([g[..., :edge], g[..., -edge:]], -1)
    return g - side.mean(-1, keepdim=True)


def kinematic_features(z, rolls_deg, win=3.0):
    """z: (B,6,H,W) stamps in noise-sigma units.  Returns (B, 10) features:
    [e_phot, kin_along, kin_across, kin_amp, sin2dpa, cos2dpa, dpa_weight,
     line_size, line_flux, cont_level] where 'along/across' are the velocity-
    gradient components relative to the measured photometric major axis."""
    B = z.shape[0]
    img = (z[:, 0] + z[:, 1]).clamp_min(0)
    pa, e, _, _ = moments_pa(img)
    g = z[:, 2:6]
    cont = torch.cat([g[..., :5], g[..., -5:]], -1).mean((-1, -2, -3))
    L = continuum_subtract(g)
    rolls = torch.tensor([math.radians(r) for r in rolls_deg], dtype=z.dtype)
    # stamp j shows the sky rotated by -roll_j: rotate by +roll_j back to the sky
    Ls = rotate(L.reshape(B * 4, 1, *L.shape[-2:]), rolls.repeat(B)).reshape(B, 4, *L.shape[-2:])
    qs = []
    flux = 0
    for j in range(4):
        qxx, qyy, qxy, tot, _, _ = _moments(Ls[:, j].clamp_min(0), win)
        qs.append(torch.stack([qxx, qyy, qxy], 1))
        flux = flux + tot
    Qj = torch.stack(qs, 1)                               # (B,4,3)
    # design matrix: unknowns [Qxx, Qyy, Qxy, hx, hy]
    A = torch.zeros(4, 3, 5, dtype=z.dtype)
    for j in range(4):
        dx, dy = math.cos(rolls[j]), math.sin(rolls[j])
        A[j, 0, 0] = A[j, 1, 1] = A[j, 2, 2] = 1.0
        A[j, 0, 3], A[j, 0, 4] = 2 * dx, 0.0              # xx: 2 dx hx
        A[j, 1, 3], A[j, 1, 4] = 0.0, 2 * dy              # yy: 2 dy hy
        A[j, 2, 3], A[j, 2, 4] = dy, dx                   # xy: dx hy + dy hx
    A = A.reshape(12, 5)
    sol = torch.linalg.lstsq(A[None].expand(B, -1, -1), Qj.reshape(B, 12, 1)).solution[..., 0]
    Qxx, Qyy, Qxy, hx, hy = sol.unbind(1)
    det = (Qxx * Qyy - Qxy**2).clamp_min(1e-3)
    gx = (Qyy * hx - Qxy * hy) / det                     # G = Q^-1 h
    gy = (-Qxy * hx + Qxx * hy) / det
    c, s = torch.cos(pa), torch.sin(pa)
    along = c * gx + s * gy
    across = -s * gx + c * gy
    amp = torch.sqrt(gx**2 + gy**2 + 1e-12)
    kpa = torch.atan2(gy, gx)
    dpa = kpa - pa
    size = torch.sqrt((Qxx + Qyy).clamp_min(1e-3))
    return torch.stack([e, along, across, amp, torch.sin(2 * dpa), torch.cos(2 * dpa),
                        e * amp, size, torch.log10(flux.clamp_min(1)), cont], 1)
