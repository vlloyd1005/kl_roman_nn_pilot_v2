# Roman kinematic-lensing mock stamps: pilot training set

Purpose: train a network to predict the shear of a disk galaxy from its Roman
imaging and grism stamps. Inputs are six 32 x 32 images per galaxy; the targets
are the two galaxy-frame shear components `truth.g_plus` and `truth.g_cross`.

## Files

- `pilot_v2_all.parquet`: 5870 rows, 373 MB, one row per simulated galaxy.
- `meta_pilot_v2_all.json`: units, image shapes, and the settings shared by
  every row.
- `SCHEMA.md`: every column with a one-line description.
- `diagnostics/realization_N/`: truth-parameter distributions, noise checks and
  example rows (noisy / noiseless / residual per channel) for each of the two
  independent realizations the table was built from (`db_seed` column).
- `format_sample_100/`: a 100-row subset with the same layout, for testing a
  loader before touching the full table.
- `PROVENANCE.md`: code version and settings.

## What one row is

A galaxy from the COSMOS2025 catalog with a painted Halpha line, placed at a
random inclination, position angle and shear, and observed as Roman's
High Latitude Wide Area Survey would see it: two imaging bands (F129, F158) and
four grism exposures at roll angles 0, 10, 167 and 177 degrees. Redshift
0.55 to 1.9. Only galaxies whose Halpha line is detectable are kept
(realized line signal-to-noise 3 to 24 for 90% of rows, median 7, extremes 0.7 and 100), so many grism stamps
are faint; that is the real survey depth.

Galaxies are independent rows. The same catalog galaxy can appear in both
realizations with a fresh orientation, shear, rotation speed and noise, and
with small random changes (0.1 dex) to its redshift, mass, size and fluxes.
If you want validation galaxies the network has never seen, split by
`pop.id` (catalog id) rather than by row.

## Inputs

Each of the 18 image columns is a flat list of 1024 float32 values, row-major,
reshape to (32, 32). Pixel scale 0.11 arcsec.

| column | content | units |
|---|---|---|
| `img.bb_F129`, `img.bb_F158` | noisy imaging stamps | uJy per pixel |
| `img.gr_roll0` .. `img.gr_roll3` | noisy grism stamps (line + continuum) | 1e-17 erg/s/cm2 per pixel |
| `img.<channel>_var` | per-pixel noise variance | squared stamp units |
| `img.<channel>_clean` | the same stamp without noise | stamp units |

Imaging stamps are in sky orientation. Grism stamp j is the sky rotated by
`roll_deg_j`; the spectrum is dispersed along the second array axis (columns).

Noise is Gaussian with a flat background per channel plus the source's own
photon noise: `var = sigma_bkg**2 + max(clean, 0) / gain`, using the
`sigma_bkg.<channel>` and `gain.<channel>` columns. The background is the same
for every galaxy (it is the survey depth), so the noise level carries no
information about the galaxy.

## Targets and other truths

- `truth.g_plus`, `truth.g_cross`: shear along and at 45 degrees to the
  galaxy's major axis. Drawn as Gaussians of width 0.03 per component,
  truncated at |g| < 0.1. `truth.g1`, `truth.g2` are the same shear in sky
  coordinates.
- `truth.cosi`: cosine of inclination (0 edge-on, 1 face-on), uniform on
  [0.05, 0.95]. `truth.theta_int`: position angle, radians from the +x axis,
  uniform on [0, 2 pi).
- `truth.z`, `truth.vel.vcirc` (rotation speed, km/s), `truth.vel.v0`
  (systemic velocity offset, km/s), sizes in arcsec (`*.rscale`), fluxes,
  disk thickness `truth.h_over_r`. Every model parameter has a `truth.*`
  column; some are copies of a shared value (e.g. `truth.F158.h_over_r` =
  `truth.h_over_r`).
- `snr_eff.line_total`, `snr_eff.F158`, ...: realized signal-to-noise of the
  noiseless stamp against the variance map. Use these, not `line_snr` or
  `broadband_snr_*`, which are the cruder values the selection was made on.
- `pop.*`: catalog properties (mass, line flux, equivalent width, ...).

## Loading

```python
import json, numpy as np, pandas as pd

d = pd.read_parquet('pilot_v2_all.parquet')
meta = json.load(open('meta_pilot_v2_all.json'))
channels = ['bb_F129', 'bb_F158', 'gr_roll0', 'gr_roll1', 'gr_roll2', 'gr_roll3']

def stack(key):  # (n_rows, 32, 32) float32
    return np.stack(d[f'img.{key}'].to_numpy()).reshape(len(d), *meta['image_shapes'][key])

x = np.stack([stack(c) for c in channels], axis=1)            # (n, 6, 32, 32) noisy stamps
var = np.stack([stack(f'{c}_var') for c in channels], axis=1)  # matching variance maps
y = d[['truth.g_plus', 'truth.g_cross']].to_numpy()            # (n, 2) targets
```

The two imaging channels and the four grism channels have very different
pixel units; normalize per channel (for example divide by `sigma_bkg`).

## Known limitations of this pilot

- 3770 distinct catalog galaxies behind the 5870 rows; the redshift
  distribution is bimodal (peaks near 1.0 and 1.6) because the catalog is.
- Stamps are 32 pixels (3.5 arcsec). A typical stamp holds 95% of the
  galaxy's flux (the rest is PSF wings and outer disk); 2% of rows hold less
  than 89%.
- One noise realization per row.
- The forward model is our own pipeline, not an image simulator: single
  exponential disks, no bulges, no neighbors, no detector effects beyond
  pixelization and the Roman PSF.
