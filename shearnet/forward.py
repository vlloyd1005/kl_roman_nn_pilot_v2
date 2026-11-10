"""Forward model of the Roman imaging + grism stamps, rebuilt to match the
kl_roman_pipe mocks (validated by chi^2 against the noiseless stamps).

Imaging : GalSim InclinedExponential, rotated by theta_int, sheared by (g1, g2),
          shifted, convolved with the Roman WFI PSF (SCA 10), drawn on 0.11" pixels.
Grism   : for roll psi, the sky is rotated by psi; the Halpha line image is
          dispersed along x with a per-pixel shift set by the line-of-sight
          velocity of a sheared, inclined disk with an arctan rotation curve,
          plus a flat continuum; then convolved with the PSF and pixelised.
"""
import math
from functools import lru_cache

import galsim
import galsim.roman as R
import numpy as np

PIX = 0.11
NPIX = 32
C_KMS = 299792.458
HA_NM = 656.461   # vacuum
DISP_NM = 1.1

# Convention switches (set by the validation fit; see validate_forward.py)
CONV = dict(roll_sign=-1, vel_sign=+1, disp_sign=+1, ha_nm=HA_NM, os=5,
            cont_halfwidth_pix=20.0, arctan=True, psf_bin=4,
            psf_nm=0.0, lsf_kms=0.0, psf_extra_sig=0.0,
            vscale=1.0, rv_scale=1.0, amp_line=0.93, amp_cont=0.95,
            v0_sign=1.0, lam_off_pix=0.0,
            los3d=False, n_los=32, rmode=0, vel_center=1)


@lru_cache(maxsize=4)
def _psf(band_or_nm, pupil_bin=4):
    if isinstance(band_or_nm, str):
        lam = R.getBandpasses()[band_or_nm].effective_wavelength
        return R.getPSF(10, band_or_nm, pupil_bin=pupil_bin, wavelength=lam)
    return R.getPSF(10, None, pupil_bin=pupil_bin, wavelength=float(band_or_nm))


_PSF_TABLE = None


def load_psf_table(path):
    """Use precomputed kernels (make_psf_table.py) instead of building GalSim PSFs."""
    global _PSF_TABLE
    z = np.load(path)
    _PSF_TABLE = {int(l): k.astype(np.float64) for l, k in zip(z["lams"], z["kernels"])}


def _psf_kernel(lam_key, os, pupil_bin):
    if _PSF_TABLE is not None and os == 5 and pupil_bin == 4 and int(lam_key) in _PSF_TABLE:
        return _PSF_TABLE[int(lam_key)]
    return _psf_kernel_galsim(lam_key, os, pupil_bin)


@lru_cache(maxsize=1024)
def _psf_kernel_galsim(lam_key, os, pupil_bin):
    """PSF drawn on the fine grid (no pixel), normalised; lam_key = rounded nm.
    The GalSim PSF object is built, drawn and discarded (they hold large pupil arrays)."""
    psf = R.getPSF(10, None, pupil_bin=pupil_bin, wavelength=float(lam_key))
    n = 24 * os + 1
    k = psf.drawImage(nx=n, ny=n, scale=PIX / os, method="no_pixel").array
    return k / k.sum()


def _disk(flux, rscale, hr, cosi, theta, g1, g2, x0, y0):
    inc = math.acos(min(max(cosi, 0.0), 1.0))
    g = galsim.InclinedExponential(inc * galsim.radians, scale_radius=rscale,
                                   scale_h_over_r=hr, flux=flux)
    return g.rotate(theta * galsim.radians).shear(g1=g1, g2=g2).shift(x0, y0)


def render_imaging(p, band):
    """p: dict of truth.* values (keys without 'truth.' prefix). band 'F129'|'F158'."""
    gb = {"F129": "J129", "F158": "H158"}[band]
    obj = _disk(p[f"{band}.flux"], p[f"{band}.rscale"], p["h_over_r"], p["cosi"],
                p["theta_int"], p["g1"], p["g2"], p[f"{band}.x0"], p[f"{band}.y0"])
    return galsim.Convolve(obj, _psf(gb, CONV["psf_bin"])).drawImage(
        nx=NPIX, ny=NPIX, scale=PIX).array


def _rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


def _fft_conv(img, ker):
    n0, n1 = img.shape
    k0, k1 = ker.shape
    s0, s1 = n0 + k0 - 1, n1 + k1 - 1
    f = np.fft.irfft2(np.fft.rfft2(img, (s0, s1)) * np.fft.rfft2(ker, (s0, s1)), (s0, s1))
    o0, o1 = (k0 - 1) // 2, (k1 - 1) // 2
    return f[o0:o0 + n0, o1:o1 + n1]


def render_grism(p, roll_deg, conv=None, components=("line", "cont")):
    cv = dict(CONV, **(conv or {}))
    os = cv["os"]
    pad = 8                                   # native pixels of padding for dispersion/PSF
    nf = (NPIX + 2 * pad) * os
    sc = PIX / os
    psi = cv["roll_sign"] * math.radians(roll_deg)
    # rotated sky: PA, shear and offsets rotate by psi
    th = p["theta_int"] + psi
    gc = (p["g1"] + 1j * p["g2"]) * np.exp(2j * psi)
    g1, g2 = gc.real, gc.imag
    z = p["z"]
    lam_c = cv["ha_nm"] * (1 + z)
    dpx = DISP_NM / os                        # nm per fine pixel along x

    # fine-grid coordinates (arcsec), centred like GalSim's drawImage
    ax = (np.arange(nf) - (nf - 1) / 2) * sc
    X, Y = np.meshgrid(ax, ax)
    out = np.zeros((nf, nf))

    if "line" in components:
        lx0, ly0 = _rot(p["Halpha.x0"], p["Halpha.y0"], psi)
        I = _disk(p["Halpha.flux"], p["Halpha.rscale"], p["h_over_r"], p["cosi"], th,
                  g1, g2, lx0, ly0).drawImage(nx=nf, ny=nf, scale=sc, method="no_pixel").array
        # sky -> (x', y') in the projected-disk frame (x' along the major axis)
        S = np.array([[1 + g1, g2], [g2, 1 - g1]]) / math.sqrt(1 - g1**2 - g2**2)
        Si = np.linalg.inv(S)
        # vel_center: 0 = Halpha centroid, 1 = stamp origin, 2 = imaging (F158) centroid
        vc = int(cv["vel_center"])
        if vc == 0:
            ox, oy = lx0, ly0
        elif vc == 1:
            ox, oy = 0.0, 0.0
        else:
            ox, oy = _rot(p["F158.x0"], p["F158.y0"], psi)
        dx, dy = X - ox, Y - oy
        ux = Si[0, 0] * dx + Si[0, 1] * dy
        uy = Si[1, 0] * dx + Si[1, 1] * dy
        xp, yp = _rot(ux, uy, -th)
        c = min(max(p["cosi"], 1e-3), 1.0)
        sn = math.sqrt(1 - c * c)
        rs_v = cv["rv_scale"] * p["vel.rscale"]

        def vrot_of(Rr):
            return cv["vscale"] * p["vel.vcirc"] * (2 / math.pi) * np.arctan(Rr / rs_v)

        sig = lam_c * math.hypot(p["Halpha.dispersion"], cv["lsf_kms"]) / C_KMS / dpx
        kv = cv["disp_sign"] * lam_c / C_KMS / dpx          # fine px per km/s
        base = kv * cv["v0_sign"] * p["vel.v0"] + cv["lam_off_pix"] * os
        if cv["los3d"]:
            # integrate along each line of sight through the thick disk:
            # disk coords (X, Y, Z) = (x', y' c, -y' s) + l (0, s, c)
            rs, hz = p["Halpha.rscale"], p["h_over_r"] * p["Halpha.rscale"]
            L = 6.0 * (rs * sn + hz * c) + 3 * hz
            ls = np.linspace(-L, L, cv["n_los"])
            Xd = xp[None]
            Yd = yp[None] * c + ls[:, None, None] * sn
            Zd = -yp[None] * sn + ls[:, None, None] * c
            Rr = np.sqrt(Xd**2 + Yd**2) + 1e-9
            w = np.exp(-Rr / rs) / np.cosh(Zd / hz) ** 2
            w = w / (w.sum(0, keepdims=True) + 1e-300)
            shifts = base + kv * cv["vel_sign"] * vrot_of(Rr) * sn * Xd / Rr
        else:
            Rr = np.hypot(xp, yp / c) + 1e-9
            Rarg = {0: Rr, 1: np.hypot(xp, yp) + 1e-9, 2: np.hypot(xp, yp * c) + 1e-9}[int(cv["rmode"])]
            w = np.ones((1,) + xp.shape)
            shifts = (base + kv * cv["vel_sign"] * vrot_of(Rarg) * sn * xp / Rr)[None]
        K = int(np.ceil(np.abs(shifts).max() + 5 * sig)) + 1
        G = np.zeros_like(I)
        for k in range(-K, K + 1):
            prof = (w * np.exp(-0.5 * ((k - shifts) / sig) ** 2)).sum(0)
            contrib = I * prof
            if k >= 0:
                G[:, k:] += contrib[:, :nf - k]
            else:
                G[:, :k] += contrib[:, -k:]
        norm = np.sqrt(2 * math.pi) * sig
        out += cv["amp_line"] * G / norm

    if "cont" in components:
        cx0, cy0 = _rot(p["Halpha.cont.x0"], p["Halpha.cont.y0"], psi)
        Ic = _disk(p["Halpha.cont.flux_per_nm"], p["Halpha.cont.rscale"], p["h_over_r"],
                   p["cosi"], th, g1, g2, cx0, cy0).drawImage(
            nx=nf, ny=nf, scale=sc, method="no_pixel").array
        # flat spectrum over lam_c +- halfwidth: a sky pixel at x' lands on x = x' + d
        hw = cv["cont_halfwidth_pix"] * os
        row = Ic.sum(axis=1)          # not used directly; do the exact box sum below
        cs = np.cumsum(np.pad(Ic, ((0, 0), (1, 0))), axis=1)
        xs = np.arange(nf)
        lo = np.clip(np.floor(xs - hw).astype(int), 0, nf)
        hi = np.clip(np.floor(xs + hw).astype(int) + 1, 0, nf)
        out += cv["amp_cont"] * (cs[:, hi] - cs[:, lo]) * dpx
        del row

    lam_key = int(round((cv["psf_nm"] or lam_c) / 5.0) * 5)
    ker = _psf_kernel(lam_key, os, cv["psf_bin"])
    if cv["psf_extra_sig"] > 0:          # extra Gaussian blur, sigma in native pixels
        s_ = cv["psf_extra_sig"] * os
        kk = np.arange(-int(4 * s_) - 1, int(4 * s_) + 2)
        g_ = np.exp(-0.5 * (kk / s_) ** 2)
        ker = _fft_conv(ker, np.outer(g_, g_) / g_.sum() ** 2)
    out = _fft_conv(out, ker)
    out = out.reshape(nf // os, os, nf // os, os).sum(axis=(1, 3))
    return out[pad:pad + NPIX, pad:pad + NPIX]


def truth_dict(row):
    return {k[6:]: float(v) for k, v in row.items() if k.startswith("truth.")}
