#!/usr/bin/env python3
"""Precompute Roman SCA-10 PSF kernels (fine grid, no pixel) on a 5 nm grid
covering Halpha at z = 0.5 - 2.0, so simulation workers never build GalSim PSFs."""
import multiprocessing as mp
import sys

import numpy as np
import galsim.roman as R

OS, BIN, PIX = 5, 4, 0.11


def one(lam):
    psf = R.getPSF(10, None, pupil_bin=BIN, wavelength=float(lam))
    n = 24 * OS + 1
    k = psf.drawImage(nx=n, ny=n, scale=PIX / OS, method="no_pixel").array
    return lam, (k / k.sum()).astype(np.float32)


if __name__ == "__main__":
    lams = np.arange(980, 1975, 5)
    with mp.Pool(4, maxtasksperchild=8) as p:
        res = dict(p.map(one, lams, chunksize=4))
    np.savez(sys.argv[1], lams=lams, kernels=np.stack([res[l] for l in lams]))
    print("saved", len(lams))
